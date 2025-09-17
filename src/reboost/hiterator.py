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
        input_detector: str = "",
        output_detectors: str | list = "",
        input_hdf5_group: str = "stp",
        start_hit: int = 0,
        max_n_hits: int | None = None,
        buffer_size_rows: int = 5_000_000,
        overwrite: bool = False,
    ):
        self.files = utils.get_file_dict(stp_files=input_files, glm_files=glm_files)
        self.reshaped_files = reshaped_files

        self.input_hdf5_group = input_hdf5_group
        self.input_detector = input_detector
        self.output_detectors = output_detectors

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

            msg = f"processing {self.input_detector} (to {self.output_detectors})"
            log.debug(msg)

            iterator = GLMIterator(
                glm_file,
                stp_file,
                lh5_group=self.input_detector,
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

                for out_detector in self.output_detectors:
                    yield HitContext(
                        data=data,
                        in_detector=self.input_detector,
                        out_detector=out_detector,
                        time_dict=self.time_dict,
                    )
