from __future__ import annotations

import io
import logging
from collections.abc import Mapping

import awkward as ak
from dbetto import AttrsDict
from lgdo import lh5

from reboost.iterator import GLMIterator

from . import core, utils

log = logging.getLogger(__name__)


def build_hit(
    config: Mapping | str,
    args: Mapping | AttrsDict,
    stp_files: list | str,
    glm_files: list | str,
    hit_files: list | str | None,
    *,
    start_evtid: int = 0,
    n_evtid: int | None = None,
    in_field: str = "stp",
    out_field: str = "hit",
    buffer: int = int(5e6),
) -> None | ak.Array:
    """Build the hit tier from the remage step files.

    Parameters
    ----------
    config
        dictionary or path to YAML file containing the processing chain. For example:

        .. code-block:: yaml

            # dictionary of objects useful for later computation. they are constructed with
            # auxiliary data (e.g. metadata). They can be accessed later as OBJECTS (all caps)
            objects:
             lmeta: LegendMetadata(ARGS.legendmetadata)
             geometry: pyg4ometry.load(ARGS.gdml)
             user_pars: legendmeta.TextDB(ARGS.par)
             dataprod_pars: legendmeta.TextDB(ARGS.dataprod_cycle)

            # processing chain is defined to act on a group of detectors
            processing_groups:

                # start with HPGe stuff, give it an optional name
                - name: geds

                  # this is a list of included detectors (part of the processing group)
                  output_detectors: OBJECTS.lmeta.channglmap(on=ARGS.timestamp)
                    .group('system').geds
                    .group('analysis.status').on
                    .map('name').keys()

                  # which columns we actually want to see in the output table
                  outputs:
                     - t0
                     - evtid
                     - energy
                     - r90
                     - drift_time

                  # in this section we define objects that will be instantiated at each
                  # iteration of the for loop over input tables (i.e. detectors)
                  detector_objects:
                     # The following assumes that the detector metadata is stored in the GDML file
                     pyobj: legendhpges.make_hpge(pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR))
                     phyvol: OBJECTS.geometry.physical_volume_dict[DETECTOR]
                     drift_time_map: lgdo.lh5.read(DETECTOR, ARGS.dtmap_file)

                  # this defines "hits", i.e. layout of the output hit table
                  # group steps by time and evtid with 10us window
                  hit_table_layout: reboost.shape.group_by_time(STEPS, window=10)

                  # finally, the processing chain
                  operations:

                    t0: ak.fill_none(ak.firsts(HITS.time, axis=-1), np.nan)

                    evtid: ak.fill_none(ak.firsts(HITS.__evtid, axis=-1), np.nan)

                    # distance to the nplus surface in mm
                    distance_to_nplus_surface_mm: reboost.hpge.distance_to_surface(
                        HITS.__xloc, HITS.__yloc, HITS.__zloc,
                        DETECTOR_OBJECTS.pyobj,
                        DETECTOR_OBJECTS.phyvol.position.eval(),
                        surface_type='nplus')

                    # activness based on FCCD (no TL)
                    activeness: ak.where(
                        HITS.distance_to_nplus_surface_mm <
                            lmeta.hardware.detectors.germanium.diodes[DETECTOR].characterization.combined_0vbb_fccd_in_mm.value,
                        0,
                        1
                        )

                    activeness2: reboost.math.piecewise_linear(
                        HITS.distance_to_nplus_surface_mm,
                        PARS.tlayer[DETECTOR].start_in_mm,
                        PARS.fccd_in_mm,
                        )

                    # summed energy of the hit accounting for activeness
                    energy_raw: ak.sum(HITS.__edep * HITS.activeness, axis=-1)

                    # energy with smearing
                    energy: reboost.math.sample_convolve(
                        scipy.stats.norm, # resolution distribution
                        loc=HITS.energy_raw, # parameters of the distribution (observable to convolve)
                        scale=np.sqrt(PARS.a + PARS.b * HITS.energy_raw) # another parameter
                        )

                    # this is going to return "run lengths" (awkward jargon)
                    clusters_lengths: reboost.shape.cluster.naive(
                        HITS, # can also pass the exact fields (x, y, z)
                        size=1,
                        units="mm"
                        )

                    # example of low level reduction on clusters
                    energy_clustered: ak.sum(ak.unflatten(HITS.__edep, HITS.clusters_lengths), axis=-1)

                    # example of using a reboost helper
                    steps_clustered: reboost.shape.reduction.energy_weighted_average(HITS, HITS.clusters_lengths)

                    r90: reboost.hpge.psd.r90(HITS.steps_clustered)

                    drift_time: reboost.hpge.psd.drift_time(
                        HITS.steps_clustered,
                        DETECTOR_OBJECTS.drift_time_map
                        )

                # example basic processing of steps in scintillators
                - name: lar
                  output_detectors: scintillators
                  outputs:
                    - evtid
                    - tot_edep_wlsr

                  operations:
                    tot_edep_wlsr: ak.sum(HITS[(HITS.__detuid == 0) & (HITS.__zloc < 3000)].__edep, axis=-1)

                - name: spms

                  output_detectors: OBJECTS.lmeta.channglmap(on=ARGS.timestamp)
                    .group("system").spms
                    .group("analysis.status").on
                    .map("name").keys()

                  # by default, reboost looks in the steps input table for a table with the
                  # same name as the current detector. This can be overridden for special processors
                  input_detectors_mapping:
                    scintillators: OUTPUT_DETECTORS

                  outputs:
                    - t0
                    - evtid
                    - pe_times

                  detector_objects:
                     meta: pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR)
                     optmap_lar: lgdo.lh5.read(DETECTOR, "optmaps/pen", ARGS.optmap_path)
                     optmap_pen: lgdo.lh5.read(DETECTOR, "optmaps/lar", ARGS.optmap_path)

                  hit_table_layout: reboost.shape.group_by_time(STEPS, window=10)

                  operations:
                    pe_times_lar: reboost.spms.detected_photoelectrons(
                        STEPS,
                        DETECTOR_OBJECTS.optmap_lar,
                        0
                     )

                    pe_times_pen: reboost.spms.detected_photoelectrons(
                        STEPS,
                        DETECTOR_OBJECTS.optmap_pen,
                        1
                     )

                    pe_times: ak.concatenate([HITS.pe_times_lar, HITS.pe_times_pen], axis=-1)

        args
            dictionary or :class:`legendmeta.AttrsDict` of the global arguments.
        start_evtid
            first evtid to read.
        n_evtid
            number of evtid to read, if `None` read all.
        stp_files
            list of strings or string of the stp file path.
        glm_files
            list of strings or string of the glm file path.
        hit_files
            list of strings or string of the hit file path. The `hit` file can also be `None` in which
            case the hits are returned as an `ak.Array` in memory.
        in_field
            name of the input field in the remage output.
        out_field
            name of the output field
        buffer
            buffer size for use in the `LH5Iterator`.

    """

    # extract the config file
    if isinstance(config, str):
        config = io.load_dict(config)

    # get the arguments
    if not isinstance(args, AttrsDict):
        args = AttrsDict(args)

    # get the global objects
    global_objects = utils.dict2tuple(
        core.get_global_objects(
            expressions=config.get("objects", {}).items(), local_dict={"ARGS": args}
        )
    )

    # get the input files
    files = {}
    for file_type, file_list in zip(["stp", "glm", "hit"], [stp_files, glm_files, hit_files]):
        if isinstance(file_list, str):
            files[file_type] = list(file_list)

    output_table = None

    # iterate over files
    for file_idx, (stp_file, glm_file) in enumerate(zip(files["stp"], files["glm"])):
        # loop over processing groups
        for group_idx, proc_group in enumerate(config["processing_groups"]):
            # extract the output detectors and the mapping to input detectors

            detectors_mapping = core.get_detector_mapping(
                output_expression=proc_group.get("output_detectors"),
                input_map_expression=proc_group.get("input_detector_mapping", None),
                objects=global_objects,
            )

            # loop over detectors
            for in_det_idx, (in_detector, out_detectors) in enumerate(detectors_mapping.items()):
                # get detector objects
                det_objects = core.get_detector_objects(
                    output_detectors=out_detectors,
                    args=args,
                    global_objects=global_objects,
                    proc_group=proc_group,
                )

                # begin iterating over the glm
                for out_det_idx, out_detector in enumerate(out_detectors):
                    # loop over the rows
                    glm_it = GLMIterator(
                        glm_file,
                        stp_file,
                        lh5_group=in_detector,
                        start_row=start_evtid,
                        stp_field=in_field,
                        n_rows=n_evtid,
                        read_vertices=True,
                        buffer=buffer,
                    )
                    for stps, _, chunk_idx, _ in glm_it:
                        # converting to awwkard
                        if stps is None:
                            continue

                        # produce the hit table
                        ak_obj = stps.view_as("ak")
                        hit_table = core.evaluate_hit_table_layout(
                            ak_obj, expression=proc_group["hit_table_layout"]
                        )

                        local_dict = {
                            "DETECTOR_OBJECTS": det_objects[out_detector],
                            "OBJECTS": global_objects,
                        }
                        # add fields
                        for field, expression in proc_group["operations"].items():
                            # evaluate the expression
                            col = core.evaluate_expression(
                                hit_table,
                                table_name="STEPS",
                                expression=expression,
                                local_dict=local_dict,
                            )
                            hit_table.add_field(field, col)

                        # remove unwanted fields
                        hit_table = core.remove_columns(hit_table, outputs=proc_group["outputs"])

                        # get the IO mode
                        wo_mode = utils.get_wo_mode(
                            file_idx=file_idx,
                            group_idx=group_idx,
                            in_det_idx=in_det_idx,
                            out_det_idx=out_det_idx,
                            chunk_idx=chunk_idx,
                        )

                        # now write
                        if files["hit"] is not None:
                            lh5.write(
                                hit_table,
                                f"{out_detector}/{out_field}",
                                files["hit"][file_idx],
                                wo_mode=wo_mode,
                            )

                        else:
                            output_table = (
                                hit_table.view_as("ak")
                                if output_table is None
                                else ak.concatenate((output_table, hit_table.view_as("ak")))
                            )

    # return output table or nothing
    return output_table
