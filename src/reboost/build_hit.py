"""Routines to build the `hit` tier from the `stp` tier.

A :func:`build_hit` to parse the following configuration file:

.. code-block:: yaml

    # dictionary of objects useful for later computation. they are constructed with
    # auxiliary data (e.g. metadata). They can be accessed later as OBJECTS (all caps)
    objects:
     lmeta: LegendMetadata(ARGS.legendmetadata)
     geometry: pyg4ometry.load(ARGS.gdml)
     user_pars: dbetto.TextDB(ARGS.par)
     dataprod_pars: dbetto.TextDB(ARGS.dataprod_cycle)

    # processing chain is defined to act on a group of detectors
    processing_groups:

        # start with HPGe stuff, give it an optional name
        - name: geds

          # this is a list of included detectors (part of the processing group)
          detector_mapping:
            - output: OBJECTS.lmeta.channglmap(on=ARGS.timestamp)
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
          detector_mapping:
           - output: scintillators

          outputs:
            - evtid
            - tot_edep_wlsr

          operations:
            tot_edep_wlsr: ak.sum(HITS[(HITS.__detuid == 0) & (HITS.__zloc < 3000)].__edep, axis=-1)

        - name: spms

          # by default, reboost looks in the steps input table for a table with the
          # same name as the current detector. This can be overridden for special processors

          detector_mapping:
           - output: OBJECTS.lmeta.channglmap(on=ARGS.timestamp)
            .group("system").spms
            .group("analysis.status").on
            .map("name").keys()
             input: lar

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
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Mapping

import awkward as ak
import dbetto
from dbetto import AttrsDict
from lgdo import lh5

from reboost.iterator import GLMIterator
from reboost.profile import ProfileDict

from . import core, utils

log = logging.getLogger(__name__)


def build_hit(
    config: Mapping | str,
    args: Mapping | AttrsDict,
    stp_files: str | list[str],
    glm_files: str | list[str],
    hit_files: str | list[str] | None,
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
        dictionary or path to YAML file containing the processing chain.
    args
        dictionary or :class:`dbetto.AttrsDict` of the global arguments.
    stp_files
        list of strings or string of the stp file path.
    glm_files
        list of strings or string of the glm file path.
    hit_files
        list of strings or string of the hit file path. The `hit` file can also be `None` in which
        case the hits are returned as an `ak.Array` in memory.
    start_evtid
        first evtid to read.
    n_evtid
        number of evtid to read, if `None` read all.
    in_field
        name of the input field in the remage output.
    out_field
        name of the output field
    buffer
        buffer size for use in the `LH5Iterator`.
    """
    # extract the config file
    if isinstance(config, str):
        config = dbetto.utils.load_dict(config)

    # get the arguments
    if not isinstance(args, AttrsDict):
        args = AttrsDict(args)
    time_dict = ProfileDict()

    # get the global objects
    global_objects = AttrsDict(
        core.get_global_objects(
            expressions=config.get("objects", {}), local_dict={"ARGS": args}, time_dict=time_dict
        )
    )

    # get the input files
    files = utils.get_file_dict(stp_files=stp_files, glm_files=glm_files, hit_files=hit_files)

    output_tables = {}
    # iterate over files
    for file_idx, (stp_file, glm_file) in enumerate(zip(files.stp, files.glm)):
        msg = (
            f"... starting post processing of {stp_file} to {files.hit[file_idx]} "
            if files.hit is not None
            else f"... starting post processing of {stp_file}"
        )
        log.info(msg)

        # loop over processing groups
        for group_idx, proc_group in enumerate(config["processing_groups"]):
            proc_name = proc_group.get("name", "default")
            msg = f"... starting group {proc_name}"
            log.info(msg)

            if proc_name not in time_dict:
                time_dict[proc_name] = ProfileDict()

            # extract the output detectors and the mapping to input detectors
            detectors_mapping = utils.merge_dicts(
                [
                    core.get_detectors_mapping(
                        mapping["output"],
                        input_detector_name=mapping.get("input", None),
                        objects=global_objects,
                    )
                    for mapping in proc_group.get("detector_mapping")
                ]
            )
            # loop over detectors
            for in_det_idx, (in_detector, out_detectors) in enumerate(detectors_mapping.items()):
                msg = f"... processing {in_detector} (to {out_detectors})"
                log.info(msg)

                # get detector objects
                det_objects = core.get_detector_objects(
                    output_detectors=out_detectors,
                    args=args,
                    global_objects=global_objects,
                    expressions=proc_group.get("detector_objects", {}),
                    time_dict=time_dict[proc_name],
                )

                # begin iterating over the glm
                glm_it = GLMIterator(
                    glm_file,
                    stp_file,
                    lh5_group=in_detector,
                    start_row=start_evtid,
                    stp_field=in_field,
                    n_rows=n_evtid,
                    read_vertices=True,
                    buffer=buffer,
                    time_dict=time_dict[proc_name],
                )
                for stps, _, chunk_idx, _ in glm_it:
                    # converting to awkward
                    if stps is None:
                        continue

                    # produce the hit table
                    ak_obj = stps.view_as("ak")

                    for out_det_idx, out_detector in enumerate(out_detectors):
                        # loop over the rows
                        if out_detector not in output_tables and files.hit is None:
                            output_tables[out_detector] = None

                        hit_table = core.evaluate_hit_table_layout(
                            copy.deepcopy(ak_obj),
                            expression=proc_group["hit_table_layout"],
                            time_dict=time_dict[proc_name],
                        )

                        local_dict = {
                            "DETECTOR_OBJECTS": det_objects[out_detector],
                            "OBJECTS": global_objects,
                            "DETECTOR": out_detector,
                        }
                        # add fields
                        for field, expression in proc_group["operations"].items():
                            # evaluate the expression
                            col = core.evaluate_output_column(
                                hit_table,
                                table_name="HITS",
                                expression=expression,
                                local_dict=local_dict,
                                time_dict=time_dict[proc_name],
                                name=field,
                            )
                            hit_table.add_field(field, col)

                        # remove unwanted fields
                        hit_table = core.remove_columns(hit_table, outputs=proc_group["outputs"])

                        # get the IO mode

                        wo_mode = (
                            "of"
                            if (
                                group_idx == 0
                                and out_det_idx == 0
                                and in_det_idx == 0
                                and chunk_idx == 0
                            )
                            else "append"
                        )

                        # now write
                        if files.hit is not None:
                            if time_dict is not None:
                                start_time = time.time()

                            lh5.write(
                                hit_table,
                                f"{out_detector}/{out_field}",
                                files.hit[file_idx],
                                wo_mode=wo_mode,
                            )
                            if time_dict is not None:
                                time_dict[proc_name].update_field("write", start_time)

                        else:
                            output_tables[out_detector] = core.merge(
                                hit_table, output_tables[out_detector]
                            )

    # return output table or nothing
    log.info(time_dict)

    if output_tables == {}:
        output_tables = None

    return output_tables, time_dict
