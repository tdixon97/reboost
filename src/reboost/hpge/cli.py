from __future__ import annotations

import argparse
import logging
from pathlib import Path

import colorlog
import yaml


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
        "--proc_chain",
        help="YAML file that contains the processing chain",
        required=True,
    )
    hit_parser.add_argument(
        "--pars",
        help="YAML file that contains the pars",
        required=True,
    )
    hit_parser.add_argument(
        "--gdml",
        help="GDML file used for Geant4",
        required=False,
    )
    hit_parser.add_argument(
        "--macro",
        help="Gean4 macro file used to generate raw tier",
        required=False,
    )
    hit_parser.add_argument("input", help="input hit LH5 file", metavar="INPUT_HIT")
    hit_parser.add_argument("output", help="output evt LH5 file", metavar="OUTPUT_EVT")

    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )
    logger = logging.getLogger("reboost.hpge")
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if args.command == "hit":
        # is the import here a good idea?
        logger.info("...running raw->hit tier")
        from reboost.hpge.hit import build_hit

        with Path.open(Path(args.pars)) as config_f:
            pars = yaml.safe_load(config_f)

        with Path.open(Path(args.proc_chain)) as config_f:
            proc_config = yaml.safe_load(config_f)

        # check the processing chain
        for req_field in ["channels", "outputs", "step_group", "operations"]:
            if req_field not in proc_config:
                msg = f"error proc chain config must contain the field {req_field}"
                raise ValueError(msg)

        build_hit(
            args.input,
            args.output,
            proc_config,
            pars,
            args.bufsize,
            gdml=args.gdml,
            macro=args.macro,
        )
