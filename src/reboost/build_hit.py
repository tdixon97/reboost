"""Routines to build the `hit` tier from the `stp` tier.

A :func:`build_hit` to parse the following configuration file:

.. code-block:: yaml

    # dictionary of objects useful for later computation. they are constructed with
    # auxiliary data (e.g. metadata). They can be accessed later as OBJECTS (all caps)
    objects:
     lmeta: legendmeta.LegendMetadata(ARGS.legendmetadata)
     geometry: pyg4ometry.load(ARGS.gdml)
     user_pars: dbetto.TextDB(ARGS.par)
     dataprod_pars: dbetto.TextDB(ARGS.dataprod_cycle)

     _spms: OBJECTS.lmeta.channelmap(on=ARGS.timestamp)
        .group("system").spms
        .map("name")
     spms: "{name: spm.daq.rawid for name, spm in OBJECTS._spms.items()}"

    # processing chain is defined to act on a group of detectors
    processing_groups:

        # start with HPGe stuff, give it an optional name
        - name: geds

          # this is a list of included detectors (part of the processing group)
          detector_mapping:
            - output: OBJECTS.lmeta.channelmap(on=ARGS.timestamp)
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
            - num_scint_ph_lar

          operations:
            tot_edep_wlsr: ak.sum(HITS.edep[np.abs(HITS.zloc) < 3000], axis=-1)

        - name: spms

          # by default, reboost looks in the steps input table for a table with the
          # same name as the current detector. This can be overridden for special processors

          detector_mapping:
           - output: OBJECTS.spms.keys()
             input: lar

          hit_table_layout: reboost.shape.group_by_time(STEPS, window=10)

          pre_operations:
            num_scint_ph_lar: reboost.spms.emitted_scintillation_photons(HITS.edep, HITS.particle, "lar")
            # num_scint_ph_pen: ...

          outputs:
            - t0
            - evtid
            - pe_times

          detector_objects:
             meta: pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR)
             spm_uid: OBJECTS.spms[DETECTOR]
             optmap_lar: reboost.spms.load_optmap(ARGS.optmap_path_pen, DETECTOR_OBJECTS.spm_uid)
             optmap_pen: reboost.spms.load_optmap(ARGS.optmap_path_lar, DETECTOR_OBJECTS.spm_uid)

          operations:
            pe_times_lar: reboost.spms.detected_photoelectrons(
                HITS.num_scint_ph_lar, HITS.particle, HITS.time, HITS.xloc, HITS.yloc, HITS.zloc,
                DETECTOR_OBJECTS.optmap_lar,
                "lar",
                DETECTOR_OBJECTS.spm_uid
             )

            pe_times_pen: reboost.spms.detected_photoelectrons(
                HITS.num_scint_ph_pen, HITS.particle, HITS.time, HITS.xloc, HITS.yloc, HITS.zloc,
                DETECTOR_OBJECTS.optmap_pen,
                "pen",
                DETECTOR_OBJECTS.spm_uid
             )

            pe_times: ak.concatenate([HITS.pe_times_lar, HITS.pe_times_pen], axis=-1)

    # can list here some lh5 objects that should just be forwarded to the
    # output file, without any processing
    forward:
      - /vtx
      - /some/dataset
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
from lgdo.lh5.exceptions import LH5EncodeError

from . import core, utils
from .iterator import GLMIterator
from .profile import ProfileDict

log = logging.getLogger(__name__)


def build_hit(
    config: Mapping | str,
    args: Mapping | AttrsDict,
    stp_files: str | list[str],
    glm_files: str | list[str] | None,
    hit_files: str | list[str] | None,
    *,
    start_evtid: int = 0,
    n_evtid: int | None = None,
    out_field: str = "hit",
    buffer: int = int(5e6),
    overwrite: bool = False,
    allow_missing_inputs=True,
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
        list of strings or string of the glm file path, if `None` will be build in memory.
    hit_files
        list of strings or string of the hit file path. The `hit` file can also be `None` in which
        case the hits are returned as an `ak.Array` in memory.
    start_evtid
        first evtid to read.
    n_evtid
        number of evtid to read, if `None` read all.
    out_field
        name of the output field
    buffer
        buffer size for use in the `LH5Iterator`.
    overwrite
        flag to overwrite the existing output.
    allow_missing_inputs
        Flag to allow an input table to be missing, generally when there were no events.
    """
    # extract the config file
    if isinstance(config, str):
        config = dbetto.utils.load_dict(config)

    # get the arguments
    if not isinstance(args, AttrsDict):
        args = AttrsDict(args)

    time_dict = ProfileDict()

    # get the global objects
    global_objects = core.get_global_objects(
        expressions=config.get("objects", {}), local_dict={"ARGS": args}, time_dict=time_dict
    )

    # get the input files
    files = utils.get_file_dict(stp_files=stp_files, glm_files=glm_files, hit_files=hit_files)

    output_tables = {}
    output_tables_names = set()

    # iterate over files
    for file_idx, (stp_file, glm_file) in enumerate(zip(files.stp, files.glm, strict=True)):
        msg = (
            f"starting processing of {stp_file} to {files.hit[file_idx]} "
            if files.hit[file_idx] is not None
            else f"starting processing of {stp_file}"
        )
        log.info(msg)

        # loop over processing groups
        for group_idx, proc_group in enumerate(config["processing_groups"]):
            proc_name = proc_group.get("name", "default")
            msg = f"starting group {proc_name}"
            log.debug(msg)

            if proc_name not in time_dict:
                time_dict[proc_name] = ProfileDict()

            # extract the output detectors and the mapping to input detectors
            detectors_mapping = core.get_detector_mapping(
                proc_group.get("detector_mapping"), global_objects, args
            )

            # loop over detectors
            for in_det_idx, (in_detector, out_detectors) in enumerate(detectors_mapping.items()):
                msg = f"processing {in_detector} (to {out_detectors})"
                log.debug(msg)

                # get detector objects
                det_objects = core.get_detector_objects(
                    output_detectors=out_detectors,
                    args=args,
                    global_objects=global_objects,
                    expressions=proc_group.get("detector_objects", {}),
                    time_dict=time_dict[proc_name],
                )

                lh5_group = proc_group.get("lh5_group", "stp")
                if lh5_group is None:
                    lh5_group = ""
                    table = in_detector
                else:
                    table = f"{lh5_group}/{in_detector}"

                # begin iterating over the glm
                # check if the in_detector is in the file
                if table in lh5.ls(stp_file, lh5_group + "/"):
                    iterator = GLMIterator(
                        glm_file,
                        stp_file,
                        lh5_group=in_detector,
                        start_row=start_evtid,
                        stp_field=lh5_group,
                        n_rows=n_evtid,
                        buffer=buffer,
                        time_dict=time_dict[proc_name],
                        reshaped_files="hit_table_layout" not in proc_group,
                    )
                elif allow_missing_inputs:
                    continue
                else:
                    msg = f"Requested input detector {in_detector} is not present in the group {lh5_group} and missing inputs were not allowed"
                    raise ValueError(msg)

                for stps, chunk_idx, _ in iterator:
                    # converting to awkward
                    if stps is None:
                        continue

                    # convert to awkward
                    if time_dict is not None:
                        start_time = time.time()

                    ak_obj = stps.view_as("ak")

                    if time_dict is not None:
                        time_dict[proc_name].update_field("conv", start_time)

                    if "hit_table_layout" in proc_group:
                        hit_table_layouted = core.evaluate_hit_table_layout(
                            copy.deepcopy(ak_obj),
                            expression=proc_group["hit_table_layout"],
                            time_dict=time_dict[proc_name],
                        )
                    else:
                        hit_table_layouted = copy.deepcopy(stps)

                    local_dict = {"OBJECTS": global_objects}
                    for field, info in proc_group.get("pre_operations", {}).items():
                        _evaluate_operation(
                            hit_table_layouted, field, info, local_dict, time_dict[proc_name]
                        )

                    # produce the hit table
                    for out_det_idx, out_detector in enumerate(out_detectors):
                        # loop over the rows
                        if out_detector not in output_tables and files.hit[file_idx] is None:
                            output_tables[out_detector] = None

                        # get the attributes
                        attrs = utils.copy_units(stps)

                        # if we have more than one output detector, make an independent copy.
                        hit_table = (
                            copy.deepcopy(hit_table_layouted)
                            if len(out_detectors) > 1
                            else hit_table_layouted
                        )

                        local_dict = {
                            "DETECTOR_OBJECTS": det_objects[out_detector],
                            "OBJECTS": global_objects,
                            "DETECTOR": out_detector,
                        }
                        # add fields
                        for field, info in proc_group.get("operations", {}).items():
                            _evaluate_operation(
                                hit_table, field, info, local_dict, time_dict[proc_name]
                            )

                        # remove unwanted fields
                        if "outputs" in proc_group:
                            hit_table = core.remove_columns(
                                hit_table, outputs=proc_group["outputs"]
                            )

                        # assign units in the output table
                        hit_table = utils.assign_units(hit_table, attrs)

                        # now write
                        if files.hit[file_idx] is not None:
                            # get modes to write with
                            wo_mode = utils.get_wo_mode(
                                group=group_idx,
                                out_det=out_det_idx,
                                in_det=in_det_idx,
                                chunk=chunk_idx,
                                new_hit_file=utils.is_new_hit_file(files, file_idx),
                                overwrite=overwrite,
                            )
                            # write the file
                            utils.write_lh5(
                                hit_table,
                                files.hit[file_idx],
                                time_dict[proc_name],
                                out_field=out_field,
                                out_detector=out_detector,
                                wo_mode=wo_mode,
                            )

                        else:
                            output_tables[out_detector] = core.merge(
                                hit_table, output_tables[out_detector]
                            )

                        output_tables_names.add(out_detector)

        # forward some data, if requested
        # possible improvement: iterate over data if it's a lot
        if "forward" in config and files.hit[file_idx] is not None:
            obj_list = config["forward"]

            if not isinstance(obj_list, list):
                obj_list = [obj_list]

            for obj in obj_list:
                try:
                    wo_mode = utils.get_wo_mode_forwarded(
                        output_tables_names, utils.is_new_hit_file(files, file_idx), overwrite
                    )
                    lh5.write(
                        lh5.read(obj, stp_file),
                        obj,
                        files.hit[file_idx],
                        wo_mode=wo_mode,
                    )
                    output_tables_names.add(obj)
                except LH5EncodeError as e:
                    msg = f"cannot forward object {obj} as it has been already processed by reboost"
                    raise RuntimeError(msg) from e

    # return output table or nothing
    log.info(time_dict)

    if output_tables == {}:
        output_tables = None

    return output_tables, time_dict


def _evaluate_operation(
    hit_table, field: str, info: str | dict, local_dict: dict, time_dict: ProfileDict
) -> None:
    if isinstance(info, str):
        expression = info
        units = None
    else:
        expression = info["expression"]
        units = info.get("units", None)

    # evaluate the expression
    col = core.evaluate_output_column(
        hit_table,
        table_name="HITS",
        expression=expression,
        local_dict=local_dict,
        time_dict=time_dict,
        name=field,
    )

    if units is not None:
        col.attrs["units"] = units

    core.add_field_with_nesting(hit_table, field, col)
