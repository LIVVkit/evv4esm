#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2018-2023 UT-BATTELLE, LLC
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

"""General tools for working with ensembles."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from evv4esm import pf_color_picker, light_pf_color_picker


def monthly_to_annual_avg(var_data, cal="ignore"):
    if len(var_data) != 12:
        raise ValueError(
            "Error! There are 12 months in a year; "
            "you passed in {} monthly averages.".format(len(var_data))
        )

    if cal == "ignore":
        # weight each month equally
        avg = np.average(var_data)
    else:
        # TODO: more advanced calendar handling
        raise NotImplementedError
    return avg


def prob_plot(
    test,
    ref,
    n_q,
    img_file,
    test_name="Test",
    ref_name="Ref.",
    thing="annual global averages",
    pf=None,
    combine_hist=False,
):
    q = np.linspace(0, 100, n_q + 1)
    all_ = np.concatenate((test, ref))
    min_ = np.min(all_)
    max_ = np.max(all_)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    axes = [ax1, ax2, ax3, ax4]

    _var = os.path.split(img_file)[-1].split(".")[0]
    fig.suptitle(_var)

    plt.rc("font", family="serif")

    ax1.set_title("Q-Q Plot")
    ax1.set_xlabel("{} pdf".format(ref_name))
    ax1.set_ylabel("{} pdf".format(test_name))

    # NOTE: Axis switched here from Q-Q plot because cdf reflects about the 1-1 line
    ax2.set_title("P-P Plot")
    ax2.set_xlabel("{} cdf".format(test_name))
    ax2.set_ylabel("{} cdf".format(ref_name))

    norm_rng = [0.0, 1.0]
    ax1.plot(norm_rng, norm_rng, "gray", zorder=1)
    ax1.set_xlim(tuple(norm_rng))
    ax1.set_ylim(tuple(norm_rng))
    ax1.autoscale()

    ax2.plot(norm_rng, norm_rng, "gray", zorder=1)
    ax2.set_xlim(tuple(norm_rng))
    ax2.set_ylim(tuple(norm_rng))
    ax2.autoscale()

    if combine_hist:
        ax3.set_title("Ensemble histogram")
        ax4.set_title("Ensemble CDF")
    else:
        ax3.set_title("{} pdf".format(ref_name))

        ax4.set_title("{} pdf".format(test_name))
        ax4.set_xlabel("Unity-based normalization of {}".format(thing))
        ax4.set_ylabel("Frequency")
        ax4.set_xlim(tuple(norm_rng))
        ax4.autoscale()

    ax3.set_ylabel("Frequency")
    ax3.set_xlabel("Unity-based normalization of {}".format(thing))

    ax3.set_xlim(tuple(norm_rng))
    ax3.autoscale()

    ax4.set_ylabel("N Ensemble members")
    ax4.set_xlabel("Unity-based normalization of {}".format(thing))

    # NOTE: Produce unity-based normalization of data for the Q-Q plots because
    #       matplotlib can't handle small absolute values or data ranges. See
    #          https://github.com/matplotlib/matplotlib/issues/6015
    bnds = np.linspace(min_, max_, n_q)
    if not np.allclose(
        bnds, bnds[0], rtol=np.finfo(bnds[0]).eps, atol=np.finfo(bnds[0]).eps
    ):
        norm_ref = (ref - min_) / (max_ - min_)
        norm_test = (test - min_) / (max_ - min_)

        # Create P-P plot
        ax1.scatter(
            np.percentile(norm_ref, q),
            np.percentile(norm_test, q),
            color=pf_color_picker.get(pf, "#1F77B4"),
            zorder=2,
        )
        if combine_hist:
            # Plot joint histogram (groups test / ref side-by-side for each bin)
            freq, bins, _ = ax3.hist(
                [norm_ref, norm_test],
                bins=n_q,
                edgecolor="k",
                label=[ref_name, test_name],
                color=[
                    pf_color_picker.get(pf, "#1F77B4"),
                    light_pf_color_picker.get(pf, "#B55D1F"),
                ],
                zorder=5,
            )
            ax3.legend()

            cdf = freq.cumsum(axis=1)

            ax4.plot(
                bins,
                [0, *cdf[0]],
                color=pf_color_picker.get(pf, "#1F77B4"),
                label=ref_name,
            )
            ax4.plot(
                bins,
                [0, *cdf[1]],
                color=light_pf_color_picker.get(pf, "#B55D1F"),
                label=test_name,
            )
            ax4.set_xlim(tuple(norm_rng))
            ax4.legend()

        else:
            ax3.hist(
                norm_ref, bins=n_q, color=pf_color_picker.get(pf, "#1F77B4"), edgecolor="k"
            )
            ax4.hist(
                norm_test, bins=n_q, color=pf_color_picker.get(pf, "#1F77B4"), edgecolor="k"
            )

            # Check if these distributions are wildly different. If they are, use different
            # colours for the bottom axis? Otherwise set the scales to be the same [0, 1]
            if abs(norm_ref.mean() - norm_test.mean()) >= 0.5:
                ax3.tick_params(axis="x", colors="C0")
                ax3.spines["bottom"].set_color("C0")

                ax4.tick_params(axis="x", colors="C1")
                ax4.spines["bottom"].set_color("C1")
            else:
                ax4.set_xlim(tuple(norm_rng))

        ax3.set_xlim(tuple(norm_rng))

        # bin both series into equal bins and get cumulative counts for each bin
        ppxb = pd.cut(ref, bnds)
        ppyb = pd.cut(test, bnds)

        ppxh = ppxb.value_counts().sort_index(ascending=True) / len(ppxb)
        ppyh = ppyb.value_counts().sort_index(ascending=True) / len(ppyb)

        ppxh = np.cumsum(ppxh)
        ppyh = np.cumsum(ppyh)

        ax2.scatter(
            ppyh.values, ppxh.values, color=pf_color_picker.get(pf, "#1F77B4"), zorder=2
        )
    else:
        # Define a text box if the data are not plottable
        const_axis_text = {
            "x": 0.5,
            "y": 0.5,
            "s": f"CONSTANT FIELD\nMIN: {min_:.4e}\nMAX: {max_:.4e}",
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "backgroundcolor": ax1.get_facecolor(),
        }
        ax1.text(**const_axis_text)
        ax2.text(**const_axis_text)
        if combine_hist:
            ax3.hist(
                [test, ref],
                bins=n_q,
                edgecolor="k",
                label=[test_name, ref_name],
                color=[
                    pf_color_picker.get(pf, "#1F77B4"),
                    light_pf_color_picker.get(pf, "#B55D1F"),
                ],
                zorder=5,
            )
            ax3.legend()
            ax4.legend()
        else:
            ax3.hist(
                test,
                bins=n_q,
                edgecolor="k",
                color=pf_color_picker.get(pf, "#1F77B4"),
                zorder=5,
            )
            ax4.hist(
                ref,
                bins=n_q,
                edgecolor="k",
                label=[test_name, ref_name],
                color=pf_color_picker.get(pf, "#1F77B4"),
                zorder=5,
            )

    for axis in axes:
        axis.grid(visible=True, ls="--", lw=0.5, zorder=-1)

    plt.tight_layout()

    plt.savefig(img_file, bbox_inches="tight")

    plt.close(fig)

    return img_file
