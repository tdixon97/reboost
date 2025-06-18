from __future__ import annotations

import logging
import time
import typing

import awkward as ak
from lgdo.lh5 import LH5Store
from lgdo.types import LGDO, Table

from . import build_glm

log = logging.getLogger(__name__)


class GLMIterator:
    """A class to iterate over the rows of an event lookup map."""

    def __init__(
        self,
        glm_file: str | None,
        stp_file: str,
        lh5_group: str,
        start_row: int,
        n_rows: int | None,
        *,
        stp_field: str = "stp",
        buffer: int = 10000,
        time_dict: dict | None = None,
        reshaped_files: bool = False,
    ):
        """Constructor for the GLMIterator.

        The GLM iterator provides a way to iterate over the
        simulated geant4 evtids, extracting the number of hits or steps for
        each range in evtids. This ensures a single simulated event
        is not split between two iterations and allows to specify a
        start and an end evtid to extract.

        In case the data is already reshaped and we do not need to
        read a specific range of evtids this iterator is just loops
        over the input stp field. Otherwise if the GLM file is not provided
        this is created in memory.

        Parameters
        ----------
        glm_file
            the file containing the event lookup map, if `None` the glm will
            be created in memory if needed.
        stp_file
            the file containing the steps to read.
        lh5_group
            the name of the lh5 group to read.
        start_row
            the first row to read.
        n_rows
            the number of rows to read, if `None` read them all.
        stp_field
            name of the group.
        buffer
            the number of rows to read at once.
        time_dict
            time profiling data structure.
        reshaped_files
            flag for whether the files are reshaped.
        """
        # initialise
        self.glm_file = glm_file
        self.stp_file = stp_file
        self.lh5_group = lh5_group
        self.start_row = start_row
        self.start_row_tmp = start_row
        self.n_rows = n_rows
        self.buffer = buffer
        self.current_i_entry = 0
        self.stp_field = stp_field
        self.reshaped_files = reshaped_files

        # would be good to replace with an iterator
        self.sto = LH5Store()
        self.n_rows_read = 0
        self.time_dict = time_dict
        self.glm = None
        self.use_glm = True

        glm_n_rows = 0

        # build the glm in memory if needed
        if self.glm_file is None and (
            (self.n_rows is not None) or (self.start_row != 0) or not reshaped_files
        ):
            if self.time_dict is not None:
                time_start = time.time()

            self.glm = build_glm.build_glm(
                stp_file, None, out_table_name="glm", id_name="evtid", lh5_groups=[lh5_group]
            )

            if self.time_dict is not None:
                self.time_dict.update_field("read/glm", time_start)

            glm_n_rows = len(self.glm)

        elif self.glm_file is None:
            self.use_glm = False
        else:
            glm_n_rows = self.sto.read_n_rows(f"glm/{self.lh5_group}", self.glm_file)

        # get the number of stp rows
        stp_n_rows = self.sto.read_n_rows(f"{self.stp_field}/{self.lh5_group}", self.stp_file)

        # heuristics for a good buffer length
        if self.use_glm:
            self.buffer = int(buffer * glm_n_rows / (1 + stp_n_rows))
            msg = f"Number of stp rows {stp_n_rows}, number of glm rows {glm_n_rows} changing buffer from {buffer} to {self.buffer}"
            log.debug(msg)

    def __iter__(self) -> typing.Iterator:
        self.current_i_entry = 0
        self.n_rows_read = 0
        self.start_row_tmp = self.start_row
        return self

    def get_n_rows(self):
        """Get the number of rows to read."""
        # get the number of rows to read
        if self.time_dict is not None:
            time_start = time.time()

        if self.n_rows is not None:
            rows_left = self.n_rows - self.n_rows_read
            n_rows = self.buffer if (self.buffer > rows_left) else rows_left
        else:
            n_rows = self.buffer

        glm_rows = None
        start = 0
        n = 0

        if self.use_glm:
            if self.glm_file is not None:
                glm_rows = self.sto.read(
                    f"glm/{self.lh5_group}",
                    self.glm_file,
                    start_row=self.start_row_tmp,
                    n_rows=n_rows,
                )
                n_rows_read = len(glm_rows.view_as("ak"))

            else:
                # get the maximum row to read
                max_row = self.start_row_tmp + n_rows
                max_row = min(len(self.glm[self.lh5_group]), max_row)

                if max_row != self.start_row_tmp:
                    glm_rows = Table(self.glm[self.lh5_group][self.start_row_tmp : max_row])

                n_rows_read = max_row - self.start_row_tmp

            if self.time_dict is not None:
                self.time_dict.update_field("read/glm", time_start)

            self.n_rows_read += n_rows_read
            self.start_row_tmp += n_rows_read

            # view our glm as an awkward array
            if glm_rows is not None:
                glm_ak = glm_rows.view_as("ak")

                # remove empty rows
                glm_ak = glm_ak[glm_ak.n_rows > 0]

                if len(glm_ak) > 0:
                    # extract range of stp rows to read
                    start = glm_ak.start_row[0]
                    n = ak.sum(glm_ak.n_rows)

        else:
            start = self.start_row_tmp
            n = n_rows
            n_rows_read = n
            self.start_row_tmp += n

        return start, n, n_rows_read

    def __next__(self) -> tuple[LGDO, int, int]:
        """Read one chunk.

        Returns
        -------
        a tuple of:
            - the steps
            - the chunk index
            - the number of steps read
        """
        # read the glm rows]
        start, n, n_rows_read = self.get_n_rows()

        if self.time_dict is not None:
            time_start = time.time()

        try:
            stp_rows = self.sto.read(
                f"{self.stp_field}/{self.lh5_group}",
                self.stp_file,
                start_row=int(start),
                n_rows=int(n),
            )
            n_steps = len(stp_rows.view_as("ak"))

        except OverflowError:
            raise StopIteration from None

        if n_rows_read == 0 or n_steps == 0:
            raise StopIteration

        # save time
        if self.time_dict is not None:
            self.time_dict.update_field("read/stp", time_start)

        self.current_i_entry += 1

        return (stp_rows, self.current_i_entry - 1, n_steps)
