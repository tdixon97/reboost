from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import colorlog


def hpge_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost-hpge",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )

    parser.add_argument(
        "--bufsize",
        action="store",
        type=int,
        default=int(5e6),
        help="""Row count for input table buffering (only used if applicable). default: %(default)e""",
    )

    # step 1: hit tier
    subparsers = parser.add_subparsers(dest="command", required=True)
    hit_parser = subparsers.add_parser("hit", help="build hit file from remage raw file")

    hit_parser.add_argument(
        "--detectors",
        help="file that contains a list of detector ids that are part of the input file",
        required=True,
    )
    hit_parser.add_argument("input", help="input hit LH5 file", metavar="INPUT_HIT")
    hit_parser.add_argument("output", help="output evt LH5 file", metavar="OUTPUT_EVT")

    # parse arguments
    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )
    logger = logging.getLogger("reboost.hpge")
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.command == "hit":
        # is the import here a good idea?
        from reboost.hpge.hit import build_hit

        with Path.open(Path(args.detectors)) as detectors_f:
            detectors = json.load(detectors_f)

        build_hit(args.input, args.output, detectors, args.bufsize)
