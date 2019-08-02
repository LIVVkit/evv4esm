#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2018 UT-BATTELLE, LLC
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evv4esm import pf_color_picker

def monthly_to_annual_avg(var_data, cal='ignore'):
    if len(var_data) != 12:
        raise ValueError('Error! There are 12 months in a year; '
                         'you passed in {} monthly averages.'.format(len(var_data)))

    if cal == 'ignore':
        # weight each month equally
        avg = np.average(var_data)
    else:
        # TODO: more advanced calendar handling
        raise NotImplementedError
    return avg


def prob_plot(test, ref, n_q, img_file, test_name='Test', ref_name='Ref.',
              thing='annual global averages', pf=None):
    # NOTE: Following the methods described in
    #       https://stackoverflow.com/questions/43285752
    #       to create the Q-Q and P-P plots
    q = np.linspace(0, 100, n_q + 1)
    all_ = np.concatenate((test, ref))
    min_ = np.min(all_)
    max_ = np.max(all_)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    plt.rc('font', family='serif')

    ax1.set_title('Q-Q Plot')
    ax1.set_xlabel('{} pdf'.format(ref_name))
    ax1.set_ylabel('{} pdf'.format(test_name))

    # NOTE: Axis switched here from Q-Q plot because cdf reflects about the 1-1 line
    ax2.set_title('P-P Plot')
    ax2.set_xlabel('{} cdf'.format(test_name))
    ax2.set_ylabel('{} cdf'.format(ref_name))

    ax3.set_title('{} pdf'.format(ref_name))
    ax3.set_xlabel('Unity-based normalization of {}'.format(thing))
    ax3.set_ylabel('Frequency')

    ax4.set_title('{} pdf'.format(test_name))
    ax4.set_xlabel('Unity-based normalization of {}'.format(thing))
    ax4.set_ylabel('Frequency')

    norm_rng = [0.0, 1.0]
    ax1.plot(norm_rng, norm_rng, 'gray', zorder=1)
    ax1.set_xlim(tuple(norm_rng))
    ax1.set_ylim(tuple(norm_rng))
    ax1.autoscale()

    ax2.plot(norm_rng, norm_rng, 'gray', zorder=1)
    ax2.set_xlim(tuple(norm_rng))
    ax2.set_ylim(tuple(norm_rng))
    ax2.autoscale()

    ax3.set_xlim(tuple(norm_rng))
    ax3.autoscale()

    ax4.set_xlim(tuple(norm_rng))
    ax4.autoscale()

    # NOTE: Produce unity-based normalization of data for the Q-Q plots because
    #       matplotlib can't handle small absolute values or data ranges. See
    #          https://github.com/matplotlib/matplotlib/issues/6015
    if not np.allclose(min_, max_, atol=np.finfo(max_).eps):
        norm1 = (ref - min_) / (max_ - min_)
        norm2 = (test - min_) / (max_ - min_)

        ax1.scatter(np.percentile(norm1, q), np.percentile(norm2, q),
                    color=pf_color_picker.get(pf, '#1F77B4'), zorder=2)
        ax3.hist(norm1, bins=n_q, color=pf_color_picker.get(pf, '#1F77B4'))
        ax4.hist(norm2, bins=n_q, color=pf_color_picker.get(pf, '#1F77B4'))

    # bin both series into equal bins and get cumulative counts for each bin
    bnds = np.linspace(min_, max_, n_q)
    if not np.allclose(bnds, bnds[0], atol=np.finfo(bnds[0]).eps):
        ppxb = pd.cut(ref, bnds)
        ppyb = pd.cut(test, bnds)

        ppxh = ppxb.value_counts().sort_index(ascending=True) / len(ppxb)
        ppyh = ppyb.value_counts().sort_index(ascending=True) / len(ppyb)

        ppxh = np.cumsum(ppxh)
        ppyh = np.cumsum(ppyh)

        ax2.scatter(ppyh.values, ppxh.values,
                    color=pf_color_picker.get(pf, '#1F77B4'), zorder=2)

    plt.tight_layout()
    plt.savefig(img_file, bbox_inches='tight')

    plt.close(fig)

    return img_file
