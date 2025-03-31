from __future__ import annotations

import logging
import time
import typing

from lgdo.lh5 import LH5Store
from lgdo.types import LGDO

log = logging.getLogger(__name__)


class GLMIterator:
    """A class to iterate over the rows of an event lookup map."""

    def __init__(
        self,
        glm_file: str,
        stp_file: str,
        lh5_group: str,
        start_row: int,
        n_rows: int | None,
        *,
        stp_field: str = "stp",
        read_vertices: bool = False,
        buffer: int = 10000,
        time_dict: dict | None = None,
    ):
        """Constructor for the glmIterator.

        Parameters
        ----------
        glm_file
            the file containing the event lookup map.
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
        read_vertices
            whether to read also the vertices table.
        buffer
            the number of rows to read at once.
        time_dict
            time profiling data structure.
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
        self.read_vertices = read_vertices
        self.stp_field = stp_field

        # would be good to replace with an iterator
        self.sto = LH5Store()
        self.n_rows_read = 0
        self.time_dict = time_dict

    def __iter__(self) -> typing.Iterator:
        self.current_i_entry = 0
        self.n_rows_read = 0
        self.start_row_tmp = self.start_row
        return self

    def __next__(self) -> tuple[LGDO, LGDO | None, int, int]:
        # get the number of rows to read
        if self.n_rows is not None:
            rows_left = self.n_rows - self.n_rows_read
            n_rows = self.buffer if (self.buffer > rows_left) else rows_left
        else:
            n_rows = self.buffer

        if self.time_dict is not None:
            time_start = time.time()

        # read the glm rows
        glm_rows, n_rows_read = self.sto.read(
            f"glm/{self.lh5_group}", self.glm_file, start_row=self.start_row_tmp, n_rows=n_rows
        )
        if self.time_dict is not None:
            self.time_dict.update_field("read/glm", time_start)

        self.n_rows_read += n_rows_read
        self.start_row_tmp += n_rows_read

        if n_rows_read == 0:
            raise StopIteration

        # view our glm as an awkward array
        glm_ak = glm_rows.view_as("ak")

        # remove empty rows
        glm_ak = glm_ak[glm_ak.n_rows > 0]

        if len(glm_ak) > 0:
            # extract range of stp rows to read
            start = glm_ak.start_row[0]
            n = sum(glm_ak.n_rows)

            if self.time_dict is not None:
                time_start = time.time()

            stp_rows, n_steps = self.sto.read(
                f"{self.stp_field}/{self.lh5_group}", self.stp_file, start_row=start, n_rows=n
            )

            # save time
            if self.time_dict is not None:
                self.time_dict.update_field("read/stp", time_start)

            self.current_i_entry += 1

            if self.read_vertices:
                vert_rows, _ = self.sto.read(
                    f"{self.stp_field}/vertices",
                    self.stp_file,
                    start_row=self.start_row,
                    n_rows=n_rows,
                )
            else:
                vert_rows = None
            # vertex table should have same structure as glm

            return (stp_rows, vert_rows, self.current_i_entry - 1, n_steps)
        return (None, None, self.current_i_entry, 0)
