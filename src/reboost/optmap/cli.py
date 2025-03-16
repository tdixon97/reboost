from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..log_utils import setup_log
from ..utils import _check_input_file, _check_output_file

log = logging.getLogger(__name__)


def optical_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost-optical",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="""Increase the program verbosity""",
    )

    parser.add_argument(
        "--bufsize",
        action="store",
        type=int,
        default=int(5e6),
        help="""Row count for input table buffering (only used if applicable). default: %(default)e""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # STEP 1: build evt file from hit tier
    evt_parser = subparsers.add_parser("evt", help="build evt file from remage hit file")
    evt_parser.add_argument(
        "--detectors",
        help="file that contains a list of detector ids that are part of the input file",
        required=True,
    )
    evt_parser.add_argument("input", help="input hit LH5 file", metavar="INPUT_HIT")
    evt_parser.add_argument("output", help="output evt LH5 file", metavar="OUTPUT_EVT")

    # STEP 2a: build map file from evt tier
    map_parser = subparsers.add_parser("createmap", help="build optical map from evt file(s)")
    map_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    map_parser.add_argument(
        "--detectors",
        help="file that contains a list of detector ids that will be produced as additional output maps.",
    )
    map_parser.add_argument(
        "--n-procs",
        "-N",
        type=int,
        default=1,
        help="number of worker processes to use. default: %(default)e",
    )
    map_parser.add_argument(
        "--check",
        action="store_true",
        help="""Check map statistics after creation. default: %(default)s""",
    )
    map_parser.add_argument("input", help="input evt LH5 file", metavar="INPUT_EVT", nargs="+")
    map_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")

    # STEP 2b: view maps
    mapview_parser = subparsers.add_parser("viewmap", help="view optical map")
    mapview_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")
    mapview_parser.add_argument(
        "--channel",
        action="store",
        default="all",
        help="default: %(default)s",
    )
    mapview_parser.add_argument(
        "--hist",
        choices=("nr_gen", "nr_det", "p_det", "p_det_err", "p_det_err_rel"),
        action="store",
        default="p_det",
        help="select optical map histogram to show. default: %(default)s",
    )
    mapview_parser.add_argument(
        "--divide",
        action="store",
        help="default: none",
    )
    mapview_parser.add_argument(
        "--min",
        default=1e-4,
        type=(lambda s: s if s == "auto" else float(s)),
        help="colormap min value. default: %(default)e",
    )
    mapview_parser.add_argument(
        "--max",
        default=1e-2,
        type=(lambda s: s if s == "auto" else float(s)),
        help="colormap max value. default: %(default)e",
    )
    mapview_parser.add_argument("--title", help="title of figure. default: stem of filename")

    # STEP 2c: merge maps
    mapmerge_parser = subparsers.add_parser("mergemap", help="merge optical maps")
    mapmerge_parser.add_argument(
        "input", help="input map LH5 files", metavar="INPUT_MAP", nargs="+"
    )
    mapmerge_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")
    mapmerge_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    mapmerge_parser.add_argument(
        "--n-procs",
        "-N",
        type=int,
        default=1,
        help="number of worker processes to use. default: %(default)e",
    )
    mapmerge_parser.add_argument(
        "--check",
        action="store_true",
        help="""Check map statistics after creation. default: %(default)s""",
    )

    # STEP 2d: check map
    checkmap_parser = subparsers.add_parser("checkmap", help="check optical maps")
    checkmap_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")

    # STEP 3: convolve with hits from non-optical simulations
    convolve_parser = subparsers.add_parser(
        "convolve", help="convolve non-optical hits with optical map"
    )
    convolve_parser.add_argument(
        "--material",
        action="store",
        choices=("lar", "pen", "fib"),
        default="lar",
        help="default: %(default)s",
    )
    convolve_parser.add_argument(
        "--map",
        action="store",
        required=True,
        metavar="INPUT_MAP",
        help="input map LH5 file",
    )
    convolve_parser.add_argument(
        "--edep",
        action="store",
        required=True,
        metavar="INPUT_EDEP",
        help="input non-optical LH5 hit file",
    )
    convolve_parser.add_argument(
        "--edep-lgdo",
        action="store",
        required=True,
        metavar="LGDO_PATH",
        help="path to LGDO inside non-optical LH5 hit file (e.g. /hit/detXX)",
    )
    convolve_parser.add_argument("--output", help="output hit LH5 file", metavar="OUTPUT_HIT")

    # STEP X: rebin maps
    rebin_parser = subparsers.add_parser("rebin", help="rebin optical maps")
    rebin_parser.add_argument("input", help="input map LH5 files", metavar="INPUT_MAP")
    rebin_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")
    rebin_parser.add_argument("--factor", type=int, help="integer scale-down factor")

    args = parser.parse_args()

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    # STEP 1: build evt file from hit tier
    if args.command == "evt":
        from reboost.optmap.evt import build_optmap_evt

        _check_input_file(parser, args.detectors)
        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)

        # load detector ids from a JSON array
        with Path.open(Path(args.detectors)) as detectors_f:
            detectors = json.load(detectors_f)

        build_optmap_evt(args.input, args.output, detectors, args.bufsize)

    # STEP 2a: build map file from evt tier
    if args.command == "createmap":
        from reboost.optmap.create import create_optical_maps

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        with Path.open(Path(args.settings)) as settings_f:
            settings = json.load(settings_f)

        chfilter = ()
        if args.detectors is not None:
            # load detector ids from a JSON array
            with Path.open(Path(args.detectors)) as detectors_f:
                chfilter = json.load(detectors_f)

        create_optical_maps(
            args.input,
            settings,
            args.bufsize,
            chfilter=chfilter,
            output_lh5_fn=args.output,
            check_after_create=args.check,
            n_procs=args.n_procs,
        )

    # STEP 2b: view maps
    if args.command == "viewmap":
        from reboost.optmap.mapview import view_optmap

        _check_input_file(parser, args.input)
        if args.divide is not None:
            _check_input_file(parser, args.divide)
        view_optmap(
            args.input,
            args.channel,
            args.divide,
            cmap_min=args.min,
            cmap_max=args.max,
            title=args.title,
            histogram_choice=args.hist,
        )

    # STEP 2c: merge maps
    if args.command == "mergemap":
        from reboost.optmap.create import merge_optical_maps

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        with Path.open(Path(args.settings)) as settings_f:
            settings = json.load(settings_f)

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)
        merge_optical_maps(
            args.input, args.output, settings, check_after_create=args.check, n_procs=args.n_procs
        )

    # STEP 2d: check maps
    if args.command == "checkmap":
        from reboost.optmap.create import check_optical_map

        _check_input_file(parser, args.input)
        check_optical_map(args.input)

    # STEP 3: convolve with hits from non-optical simulations
    if args.command == "convolve":
        from reboost.optmap.convolve import convolve

        _check_input_file(parser, [args.map, args.edep])
        _check_output_file(parser, args.output)
        convolve(args.map, args.edep, args.edep_lgdo, args.material, args.output, args.bufsize)

    # STEP X: rebin maps
    if args.command == "rebin":
        from reboost.optmap.create import rebin_optical_maps

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)
        rebin_optical_maps(args.input, args.output, args.factor)
