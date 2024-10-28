from __future__ import annotations

import logging

from . import processors, utils

log = logging.getLogger(__name__)


def build_hit(
    lh5_in_file: str,
    lh5_out_file: str,
    config: dict,
    buffer_len: int = int(5e6),
    gdml: str | None = None,
    macro: str | None = None,
) -> None:
    """
    Build the hit tier from the raw Geant4 output

    Parameters
    ----------
    lh5_in_file
        input file containing the raw tier
    lh5_out_file
        output file 
    config
        dictonary containg the configuration / parameters
        should contain one sub-dictonary per detector with a format like:

        .. code-block:: json
            
            "det001": {
                "reso": 1,
                ...
            }

        This can contain any parameters needed in the processing chain.
    buffer_len
        number of rows to read at once
    gdml
        path to the gdml file of the geometry
    macro
        path to the Geant4 macro used to generate the raw tier
    """    

    # get the geant4 gdml and macro

    if gdml is not None:
        pass
    
    if macro is not None:
        pass

    for idx, d in enumerate(config.keys()):
        msg = f"...running event grouping for {d}"
        log.debug(msg)
        delete_input = bool(idx == 0)

        # build the processing chain
        # TODO replace this with a config file similar to pygama.build_evt

        proc = processors.def_chain(
            [
                processors.sort_data,
                processors.group_by_time,
                processors.sum_energy,
                processors.smear_energy,
            ],
            [{}, {"window": 10}, {}, {"reso": config[d]["reso"], "energy_name": "sum_energy"}],
        )

        utils.read_write_incremental(
            lh5_out_file,
            f"hit/{d}",
            proc,
            f"hit/{d}",
            lh5_in_file,
            buffer_len,
            delete_input=delete_input,
        )
