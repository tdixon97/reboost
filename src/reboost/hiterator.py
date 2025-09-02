from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from lgdo.types import Table

from . import utils
from .iterator import GLMIterator
from .profile import ProfileDict

log = logging.getLogger(__name__)


@dataclass
class HitContext:
    # raw chunk of steps (LGDO)
    data: Table
    # mapping context
    in_detector: str
    out_detector: str
    # helper/metadata
    time_dict: ProfileDict


class Hiterator:
    def __init__(
        self,
        *,
        input_files: str | list[str],
        glm_files: str | list[str] | None,
        reshaped_files: bool = True,
        detectors_mapping: dict[str, list[str]],
        input_hdf5_group: str = "stp",
        start_hit: int = 0,
        max_n_hits: int | None = None,
        buffer_size_rows: int = 5_000_000,
        overwrite: bool = False,
    ):
        self.files = utils.get_file_dict(stp_files=input_files, glm_files=glm_files)
        self.reshaped_files = reshaped_files

        self.detectors_mapping = detectors_mapping
        self.input_hdf5_group = input_hdf5_group

        self.start_hit = start_hit
        self.max_n_hits = max_n_hits
        self.buffer_size_rows = buffer_size_rows
        self.overwrite = overwrite

        self.time_dict = ProfileDict()

    def __iter__(self):
        # loop over files
        for file_idx, (stp_file, glm_file) in enumerate(
            zip(self.files.stp, self.files.glm, strict=False)
        ):
            # some logging
            if self.files.hit[file_idx] is not None:
                msg = f"starting processing of {stp_file} to {self.files.hit[file_idx]}"
                log.info(msg)
            else:
                msg = f"starting processing of {stp_file}"
                log.info(msg)

            # now loop over detectors
            for in_detector, out_detectors in self.detectors_mapping.items():
                msg = f"processing {in_detector} (to {out_detectors})"
                log.debug(msg)

                iterator = GLMIterator(
                    glm_file,
                    stp_file,
                    lh5_group=in_detector,
                    start_row=self.start_hit,
                    stp_field=self.input_hdf5_group,
                    n_rows=self.max_n_hits,
                    buffer=self.buffer_size_rows,
                    time_dict=self.time_dict,
                    reshaped_files=self.reshaped_files,
                )

                for data, _, _ in iterator:
                    if data is None:
                        continue

                    self.time_dict.update_field("conv", time.time())

                    for out_detector in out_detectors:
                        yield HitContext(
                            data=data,
                            in_detector=in_detector,
                            out_detector=out_detector,
                            time_dict=self.time_dict,
                        )
