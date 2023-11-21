#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2022 UT-BATTELLE, LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The Kolmogorov-Smirnov Test:
This tests the null hypothesis that the reference (n) and modified (m) model
Short Independent Simulation Ensembles (SISE) represent the same climate
state, based on the equality of distribution of each variable's annual global
average in the standard monthly model output between the two simulations.

The (per variable, per grid cell) null hypothesis uses the non-parametric, two-sample
(n and m) Kolmogorov-Smirnov test as the univariate test of equality of distribution of
global means. The test statistic (t) is the number of variables that reject the
(per variable) null hypothesis of equality of distribution at a 95% confidence
level. The (overall) null hypothesis is rejected if t > α, where α is some
critical number of rejecting variables. The critical value, α, is obtained from
an empirically derived approximate null distribution of t using resampling
techniques.
"""
import argparse
import os
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import livvkit
import numpy as np
import pandas as pd
from evv4esm import EVVException, human_color_names
from evv4esm.ensembles import e3sm
from evv4esm.ensembles.tools import prob_plot
from evv4esm.utils import bib2html
from livvkit import elements as el
from livvkit.util import functions as fn
from livvkit.util.LIVVDict import LIVVDict
from scipy import stats
from statsmodels.stats import multitest as smm


def variable_set(name):
    var_sets = fn.read_json(os.path.join(os.path.dirname(__file__), "ocean_vars.json"))
    try:
        the_set = var_sets[name.lower()]
        return set(the_set)
    except KeyError as _err:
        raise argparse.ArgumentTypeError(
            f"Unknown variable set! Known sets are {var_sets.keys()}"
        ) from _err


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c",
        "--config",
        type=fn.read_json,
        help=(
            "A JSON config file containing a `ks` dictionary defining "
            "the options. NOTE: command line options will override file options."
        ),
    )

    parser.add_argument("--test-case", default="default", help="Name of the test case.")

    parser.add_argument(
        "--test-dir",
        default=os.path.join(os.getcwd(), "archive"),
        help="Location of the test case run files.",
    )

    parser.add_argument(
        "--ref-case", default="fast", help="Name of the reference case."
    )

    parser.add_argument(
        "--ref-dir",
        default=os.path.join(os.getcwd(), "archive"),
        help="Location of the reference case run files.",
    )

    parser.add_argument(
        "--var-set",
        default="default",
        type=variable_set,
        help="Name of the variable set to analyze.",
    )

    parser.add_argument(
        "--ninst",
        default=30,
        type=int,
        help="The number of instances (should be the same for both cases).",
    )

    parser.add_argument(
        "--critical",
        default=0,
        type=float,
        help="The critical value (desired significance level) for rejecting the "
        + "null hypothesis.",
    )

    parser.add_argument("--img-dir", default=os.getcwd(), help="Image output location.")

    parser.add_argument(
        "--img-fmt", default="png", type=str, help="Format for output images"
    )

    parser.add_argument(
        "--component", default="mpaso", help="Model component name (e.g. eam, cam, ...)"
    )

    parser.add_argument(
        "--hist-name",
        default="hist.am.timeSeriesStatsClimatology",
        help=(
            "History file output component "
            "(e.g. h0, hist.am.timeSeriesStatsClimatology)"
        ),
    )

    parser.add_argument(
        "--alpha", default=0.05, type=float, help="Alpha threshold for pass / fail"
    )
    args, _ = parser.parse_known_args(args)

    # use config file arguments, but override with command line arguments
    if args.config:
        default_args = parser.parse_args([])

        for key, val in vars(args).items():
            if val != vars(default_args)[key]:
                args.config["ks"][key] = val

        config_arg_list = []
        _ = [
            config_arg_list.extend(["--" + key, str(val)])
            for key, val in args.config["ks"].items()
            if key != "config"
        ]
        args, _ = parser.parse_known_args(config_arg_list)

    return args


def col_fmt_ff(dat):
    """Format results for table output."""
    if dat is not None:
        try:
            _out = "{:.3e}, {:.3e}".format(*dat)
        except TypeError:
            _out = dat
    else:
        _out = "-"
    return _out


def col_fmt_ip(dat):
    """Format results for table output."""
    if dat is not None:
        try:
            _out = "{}, {:.1f}".format(*dat)
        except TypeError:
            _out = dat
    else:
        _out = "-"
    return _out


def run(name, config):
    """
    Runs the analysis.

    Args:
        name: The name of the test
        config: A dictionary representation of the configuration file

    Returns:
       The result of elements.page with the list of elements to display
    """

    config_arg_list = []
    [config_arg_list.extend(["--" + key, str(val)]) for key, val in config.items()]

    args = parse_args(config_arg_list)

    args.img_dir = os.path.join(livvkit.output_dir, "validation", "imgs", name)
    fn.mkdir_p(args.img_dir)

    details, img_gal = main(args)

    table_data = pd.DataFrame(details).T
    _hdrs = [
        "h0",
        f"Pre-Correction (N, %) < {args.alpha}",
        f"Post-Correction (N, %) < {args.alpha}",
        "mean (test case, ref. case)",
        "std (test case, ref. case)",
    ]
    table_data = table_data[_hdrs]
    for _hdr in _hdrs[1:]:
        if "(N, %)" in _hdr:
            table_data[_hdr] = table_data[_hdr].apply(col_fmt_ip)
        else:
            table_data[_hdr] = table_data[_hdr].apply(col_fmt_ff)

    tables = [
        el.Table("Rejected", data=table_data[table_data["h0"] == "reject"]),
        el.Table("Accepted", data=table_data[table_data["h0"] == "accept"]),
        el.Table("Null", data=table_data[~table_data["h0"].isin(["accept", "reject"])]),
    ]

    bib_html = bib2html(os.path.join(os.path.dirname(__file__), "ks.bib"))

    tabs = el.Tabs(
        {"Figures": img_gal, "Details": tables, "References": [el.RawHTML(bib_html)]}
    )
    rejects = [var for var, dat in details.items() if dat["h0"] == "reject"]

    results = el.Table(
        title="Results",
        data=OrderedDict(
            {
                "Test status": ["pass" if len(rejects) <= args.critical else "fail"],
                "Variables analyzed": [len(details.keys())],
                "Rejecting": [len(rejects)],
                "Critical value": [int(args.critical)],
                "Ensembles": [
                    "statistically identical"
                    if len(rejects) <= args.critical
                    else "statistically different"
                ],
            }
        ),
    )

    page = el.Page(name, __doc__.replace("\n\n", "<br><br>"), elements=[results, tabs])
    return page


def case_files(args):
    # ensure unique case names for the dictionary
    key1 = args.test_case
    key2 = args.ref_case
    if args.test_case == args.ref_case:
        key1 += "1"
        key2 += "2"

    f_sets = {
        key1: e3sm.component_monthly_files(
            args.test_dir,
            args.component,
            args.ninst,
            hist_name=args.hist_name,
            date_style="med",
        ),
        key2: e3sm.component_monthly_files(
            args.ref_dir,
            args.component,
            args.ninst,
            hist_name=args.hist_name,
            date_style="med",
        ),
    }

    output_fsets = {key1: [], key2: []}
    for key in f_sets:
        # Only require the last climatology file
        if any([not f_sets[key] for key in f_sets]):
            raise EVVException(
                f"Could not find all the required case files for case: {key}"
            )

        for iinst in f_sets[key]:
            output_fsets[key].append(f_sets[key][iinst][-1])

    return output_fsets, key1, key2


def print_summary(summary):
    print(f"    Kolmogorov-Smirnov Test: {summary['']['Case']}")
    print(f"      Variables analyzed: {summary['']['Variables analyzed']}")
    print(f"      Rejecting: {summary['']['Rejecting']}")
    print(f"      Critical value: {summary['']['Critical value']}")
    print(f"      Ensembles: {summary['']['Ensembles']}")
    print(f"      Test status: {summary['']['Test status']}\n")


def print_details(details):
    for set_ in details:
        print("-" * 80)
        print(set_)
        print("-" * 80)
        pprint(details[set_])


def summarize_result(results_page):
    summary = {"Case": results_page.title}

    for elem in results_page.elements:
        if isinstance(elem, el.Table) and elem.title == "Results":
            summary["Test status"] = elem.data["Test status"][0]
            summary["Variables analyzed"] = elem.data["Variables analyzed"][0]
            summary["Rejecting"] = elem.data["Rejecting"][0]
            summary["Critical value"] = elem.data["Critical value"][0]
            summary["Ensembles"] = elem.data["Ensembles"][0]
            break

    return {"": summary}


def populate_metadata():
    """
    Generates the metadata responsible for telling the summary what
    is done by this module's run method
    """

    metadata = {
        "Type": "ValSummary",
        "Title": "Validation",
        "TableTitle": "Kolmogorov-Smirnov test",
        "Headers": [
            "Test status",
            "Variables analyzed",
            "Rejecting",
            "Critical value",
            "Ensembles",
        ],
    }
    return metadata


def main(args):
    ens_files, key1, key2 = case_files(args)
    if args.test_case == args.ref_case:
        args.test_case = key1
        args.ref_case = key2

    # Right now climatology has five computed climatologies (JFM, AMJ, JAS, OND, ANN)
    # We need to format the variable list so it's has the key we want (_5), and is also
    # formatted to have the correct averaging prefix (timeClimatology_avg)
    var_prefix = "timeClimatology_avg"
    var_suffix = ""
    test_vars = [
        f"{var_prefix}_{test_var.format(var_suffix)}" for test_var in args.var_set
    ]
    # test_data = e3sm.load_mpas_climatology_ensemble(ens_files[key1], field_name)

    # Vectorize the ks test function. The signature maps a vector (array) to a
    # scalar (float) for this to work, the data must have the axis on which we are
    # performing the test (across ensemble members) to be the last dimension
    # (e.g. [nCells, nLevels, nEns]) this is why load_mpas_climatology_ensemble
    # returns data in this way
    ks_test = np.vectorize(stats.mstats.ks_2samp, signature="(n),(n)->(),()")

    images = {"accept": [], "reject": [], "-": []}
    details = LIVVDict()
    for var in sorted(test_vars):

        var_1 = e3sm.load_mpas_climatology_ensemble(ens_files[key1], var)
        var_2 = e3sm.load_mpas_climatology_ensemble(ens_files[key2], var)

        annuals_1 = var_1["data"]
        annuals_2 = var_2["data"]
        if isinstance(annuals_1, np.ma.MaskedArray) and isinstance(
            annuals_2, np.ma.MaskedArray
        ):
            _, p_val = ks_test(annuals_1.filled(), annuals_2.filled())
        else:
            _, p_val = ks_test(annuals_1, annuals_2)

        null_reject_pre_correct = np.sum(np.where(p_val <= args.alpha, 1, 0))
        _, p_val = smm.fdrcorrection(
            p_val.flatten(), alpha=args.alpha, method="n", is_sorted=False
        )
        null_reject_post_correct = np.sum(np.where(p_val <= args.alpha, 1, 0))

        test_result = "reject"
        # Only check the post-correction for pass / fail result. Pre-correction
        # total is just for information purposes
        if null_reject_post_correct <= args.critical:
            test_result = "accept"

        details[var][f"Pre-Correction (N, %) < {args.alpha}"] = (
            null_reject_pre_correct,
            100 * null_reject_pre_correct / np.prod(p_val.shape),
        )

        details[var][f"Post-Correction (N, %) < {args.alpha}"] = (
            null_reject_post_correct,
            100 * null_reject_post_correct / np.prod(p_val.shape),
        )

        # For output, mask out missing data, can't do this before the K-S test because
        # it's vectorised and will raise a ValueError if a particular cell is masked for
        # all ensemble members, which is fine because it's all equal
        mask_value = -0.9999e33
        annuals_1 = np.ma.masked_less(annuals_1, mask_value)
        annuals_2 = np.ma.masked_less(annuals_2, mask_value)
        details[var]["mean (test case, ref. case)"] = (
            annuals_1.mean(),
            annuals_2.mean(),
        )

        details[var]["max (test case, ref. case)"] = (annuals_1.max(), annuals_2.max())
        details[var]["min (test case, ref. case)"] = (annuals_1.min(), annuals_2.min())
        details[var]["std (test case, ref. case)"] = (annuals_1.std(), annuals_2.std())
        details[var]["h0"] = test_result

        img_file = os.path.relpath(
            os.path.join(args.img_dir, f"{var}.{args.img_fmt}"), os.getcwd()
        )

        # Plot ensemble histogram / q-q, p-p plot of
        # global means (mean over all but ensemble axis)
        prob_plot(
            annuals_1.mean(axis=tuple(range(annuals_1.ndim - 1))),
            annuals_2.mean(axis=tuple(range(annuals_2.ndim - 1))),
            20,
            img_file,
            test_name=args.test_case,
            ref_name=args.ref_case,
            pf=details[var]["h0"],
            combine_hist=True,
        )

        img_desc = (
            f"Mean annual global average of {var}{var_1['desc']} for "
            f"<em>{args.test_case}</em> is "
            f"{details[var]['mean (test case, ref. case)'][0]:.4e} and for "
            f"<em>{args.ref_case}</em> is "
            f"{details[var]['mean (test case, ref. case)'][1]:.4e}. "
            f"Pass (fail) is indicated by {human_color_names['fail'][0]} "
            f"({human_color_names['pass'][0]}) coloring of the plot markers and bars."
        )

        img_link = Path(*Path(args.img_dir).parts[-2:], Path(img_file).name)

        # Trim timeClimatology_avg so image captions in gallery are not super long
        if "timeClimatology_avg" in var:
            img_title = "_".join(var.split("_")[2:])
        else:
            img_title = var

        _img = el.Image(
            img_title, img_desc, img_link, relative_to="", group=details[var]["h0"]
        )
        images[details[var]["h0"]].append(_img)

    gals = []
    for group in ["reject", "accept", "-"]:
        _group_name = {
            "reject": "Failed variables",
            "accept": "Passed variables",
            "-": "Null variables",
        }
        if images[group]:
            gals.append(el.Gallery(_group_name[group], images[group]))

    return details, gals
