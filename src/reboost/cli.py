from __future__ import annotations

import argparse
import logging

import dbetto

from .build_glm import build_glm
from .build_hit import build_hit
from .log_utils import setup_log
from .utils import _check_input_file, _check_output_file, get_file_list

log = logging.getLogger(__name__)


def cli(args=None) -> None:
    parser = argparse.ArgumentParser(
        prog="reboost",
        description="%(prog)s command line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="""Increase the program verbosity""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # glm parser
    glm_parser = subparsers.add_parser("build-glm", help="build glm file from remage stp file")

    glm_parser.add_argument(
        "--stp-file",
        "-s",
        required=True,
        type=str,
        help="Path to the stp file, if multithreaded this will be appended with _t$idx.",
    )
    glm_parser.add_argument(
        "--glm-file",
        "-g",
        required=True,
        type=str,
        help="Path to the glm file, if multithreaded this will be appended with _t$idx. ",
    )

    # optional args
    glm_parser.add_argument(
        "--out-table-name", "-n", type=str, default="glm", help="Output table name."
    )
    glm_parser.add_argument("--id-name", "-i", type=str, default="g4_evtid", help="ID column name.")
    glm_parser.add_argument(
        "--evtid-buffer", "-e", type=int, default=int(1e7), help="event id buffer size."
    )
    glm_parser.add_argument(
        "--stp-buffer", "-b", type=int, default=int(1e7), help="stp buffer size."
    )
    glm_parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite the input file if it exists."
    )
    glm_parser.add_argument(
        "--threads",
        "-t",
        required=False,
        default=None,
        type=int,
        help="Number of threads used for remage",
    )

    # hit parser
    hit_parser = subparsers.add_parser("build-hit", help="build hit file from remage stp file")

    hit_parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file ."
    )
    hit_parser.add_argument("--args", type=str, required=True, help="Path to args file.")
    hit_parser.add_argument(
        "--stp-file",
        type=str,
        required=True,
        help="stp file to process, if multithreaded this will be appended with _t$idx",
    )
    hit_parser.add_argument(
        "--glm-file",
        type=str,
        required=False,
        default=None,
        help="glm file to process, if multithreaded this will be appended with _t$idx",
    )
    hit_parser.add_argument(
        "--hit-file",
        type=str,
        required=True,
        help="hit file to produce, if multithreaded this will be appended with _t$idx",
    )

    # optional args
    hit_parser.add_argument("--start-evtid", type=int, default=0, help="Start event id.")
    hit_parser.add_argument(
        "--n-evtid", type=int, default=None, help="Number of event id to process."
    )
    hit_parser.add_argument("--out-field", type=str, default="hit", help="Output field name.")
    hit_parser.add_argument("--buffer", type=int, default=int(5e6), help="Buffer size.")

    hit_parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite the input file if it exists."
    )
    hit_parser.add_argument(
        "--threads",
        "-t",
        required=False,
        default=None,
        type=int,
        help="Number of threads used for remage",
    )

    args = parser.parse_args(args)

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    if args.command == "build-glm":
        # catch some cases
        glm_files = get_file_list(args.glm_file, threads=args.threads)
        stp_files = get_file_list(args.stp_file, threads=args.threads)

        _check_input_file(parser, stp_files)

        if args.overwrite is False:
            _check_output_file(parser, glm_files)

        msg = "Running build_glm with arguments: \n"
        msg += f"    glm file:       {glm_files}\n"
        msg += f"    stp file:       {stp_files}\n"
        msg += f"    out_table_name: {args.out_table_name}\n"
        msg += f"    evtid_name:     {args.id_name}\n"
        msg += f"    evtid_buffer:   {args.evtid_buffer}\n"
        msg += f"    stp_buffer:     {args.stp_buffer}"

        log.info(msg)

        build_glm(
            stp_files,
            glm_files,
            out_table_name=args.out_table_name,
            id_name=args.id_name,
            evtid_buffer=args.evtid_buffer,
            stp_buffer=args.stp_buffer,
        )

    elif args.command == "build-hit":
        glm_files = get_file_list(args.glm_file, threads=args.threads)
        stp_files = get_file_list(args.stp_file, threads=args.threads)
        hit_files = get_file_list(args.hit_file, threads=args.threads)

        _check_input_file(parser, stp_files)

        if args.glm_file is not None:
            _check_input_file(parser, glm_files)

        if args.overwrite is False:
            _check_output_file(parser, hit_files)

        msg = "Running build_hit with arguments: \n"
        msg += f"    config:         {args.config}\n"
        msg += f"    args:           {args.args}\n"
        msg += f"    glm files:      {glm_files}\n"
        msg += f"    stp files:      {stp_files}\n"
        msg += f"    hit files:      {hit_files}\n"
        msg += f"    start_evtid:    {args.start_evtid}\n"
        msg += f"    n_evtid:        {args.n_evtid}\n"
        msg += f"    out_field:      {args.out_field}\n"
        msg += f"    buffer:         {args.buffer} \n"
        msg += f"    overwrite:      {args.overwrite} \n"

        log.info(msg)

        build_hit(
            config=args.config,
            args=dbetto.AttrsDict(dbetto.utils.load_dict(args.args)),
            stp_files=stp_files,
            glm_files=glm_files,
            hit_files=hit_files,
            start_evtid=args.start_evtid,
            n_evtid=args.n_evtid,
            out_field=args.out_field,
            buffer=args.buffer,
            overwrite=args.overwrite,
        )
