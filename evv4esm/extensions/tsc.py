#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2015-2022 UT-BATTELLE, LLC
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

"""The Time Step Convergence Test:
This tests the null hypothesis that the convergence of the time stepping error
for a set of key atmospheric variables is the same for a reference ensemble and
a test ensemble. Both the reference and test ensemble are generated with a
two-second time step, and for each variable the RMSD between each ensemble and
a truth ensemble, generated with a one-second time step, is calculated. RMSD is
calculated globally and over two domains, the land and the ocean. The land/ocean
domains contain just the atmosphere points that are over land/ocean cells.

At each 10 second interval during the 10 minute long simulations, the difference
in the reference and test RMSDs for each variable, each ensemble member, and each
domain are calculated and these ΔRMSDs should be zero for identical climates. A
one sided (due to self convergence) Student's T Test is used to test the null
hypothesis that the ensemble mean ΔRMSD is statistically zero at the {}%
confidence level. A rejection of the null hypothesis (mean ΔRMSD is not
statistically zero) at any time step for any variable will cause this test to
fail.
"""

import argparse
import glob
import os
from collections import OrderedDict
from itertools import groupby
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from scipy import stats
from netCDF4 import Dataset

import livvkit
from livvkit import elements as el
from livvkit.util import functions as fn
from livvkit.util.LIVVDict import LIVVDict

from evv4esm.ensembles import e3sm
from evv4esm.utils import bib2html
from evv4esm import pf_color_picker, light_pf_color_picker, human_color_names


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        type=fn.read_json,
                        default='test/tsc_pc0101123.json',
                        help='A JSON config file containing a `tsc` dictionary defining ' +
                             'the options.')

    args = parser.parse_args(args)
    name = args.config.keys()[0]
    config = args.config[name]

    return name, config


def run(name, config, print_details=False):
    """
    Runs the extension.

    Args:
        name: The name of the extension
        config: The test's config dictionary
        print_details: Whether to print the analysis details to stdout
                       (default: False)

    Returns:
       A LIVVkit page element containing the LIVVkit elements to display on a webpage
    """

    # FIXME: move into a config to NameSpace function
    test_args = OrderedDict([(k.replace('-', '_'), v) for k, v in config.items()])
    test_args = argparse.Namespace(**test_args)

    test_args.img_dir = os.path.join(livvkit.output_dir, 'validation', 'imgs', name)
    fn.mkdir_p(test_args.img_dir)
    details, img_gal = main(test_args)

    if print_details:
        _print_details(details)

    domain_headers = ['Null hypothesis', 'T test (t, P)']
    global_tbl_el = {'Type': 'V-H Table',
                     'Title': 'Validation',
                     'TableTitle': 'Analyzed variables',
                     'Headers': domain_headers,
                     'Data': details['global'],
                     }
    land_tbl_el = {'Type': 'V-H Table',
                   'Title': 'Validation',
                   'TableTitle': 'Analyzed variables',
                   'Headers': domain_headers,
                   'Data': details['land'],
                   }
    ocean_tbl_el = {'Type': 'V-H Table',
                    'Title': 'Validation',
                    'TableTitle': 'Analyzed variables',
                    'Headers': domain_headers,
                    'Data': details['ocean'],
                    }
    bib_html = bib2html(os.path.join(os.path.dirname(__file__), 'tsc.bib'))
    detail_tables = {
        "global": [],
        "land": [],
        "ocean": []
    }
    table_headers = ["Variable", *domain_headers]
    for level in detail_tables:
        for _time in details[level]:
            tbl_data = {_hdr: [] for _hdr in table_headers}
            for _var in details[level][_time]:
                tbl_data["Variable"].append(_var)
                for _hdr, val in details[level][_time][_var].items():
                    tbl_data[_hdr].append(val)

            detail_tables[level].append(
                el.Table(
                    title=f"{level.capitalize()}, {_time}",
                    data=tbl_data
                )
            )

    tabs = el.Tabs(
        {
            'Figures': [img_gal],
            'Global_details': detail_tables["global"],
            'Land_details': detail_tables["land"],
            'Ocean_details': detail_tables["ocean"],
            'References': [el.RawHTML(bib_html)]
        }
    )

    results = el.Table(
        title="Results",
        data=OrderedDict(
            {
                'Test status': [details['overall']],
                'Global': [details['domains']['delta_l2_global']],
                'Land': [details['domains']['delta_l2_land']],
                'Ocean': [details['domains']['delta_l2_ocean']],
                'Ensembles': ['statistically identical' if details['overall'] == 'Pass' else 'statistically different'],
            }
        )
    )

    # FIXME: Put into a ___ function
    doc_text = __doc__.format((1 - test_args.p_threshold) * 100).replace('\n\n', '<br><br>')
    page = el.Page(name, doc_text, elements=[results, tabs])
    return page


def main(args):
    if args.test_case == args.ref_case:
        args.test_case += '1'
        args.ref_case += '2'

    file_search_glob = '{d}/*{c}_????.h0.0001-01-01-?????.nc.{s}'
    truth_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.ref_dir, c=args.component, s='DT0001'))),
            key=lambda f: e3sm.component_file_instance(args.component, f))}

    ref_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.ref_dir, c=args.component, s='DT0002'))),
            key=lambda f: e3sm.component_file_instance(args.component, f))}

    test_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.test_dir, c=args.component, s='DT0002'))),
            key=lambda f: e3sm.component_file_instance(args.component, f))}

    # So, we want a pandas dataframe that will have the columns :
    #     (test/ref, ensemble, seconds, l2_global, l2_land, l2_ocean)
    # But, building a pandas dataframe row by row is sloooow, so append a list of list and convert
    data = []
    for instance, truth_files in truth_ens.items():
        times = [int(e3sm.file_date_str(ff, style='full').split('-')[-1]) for ff in truth_files]
        for tt, time in enumerate(times):
            with Dataset(truth_ens[instance][tt]) as truth, \
                    Dataset(ref_ens[instance][tt]) as ref, \
                    Dataset(test_ens[instance][tt]) as test:

                truth_plt, truth_ps = pressure_layer_thickness(truth)
                ref_plt, ref_ps = pressure_layer_thickness(ref)
                test_plt, test_ps = pressure_layer_thickness(test)

                ref_avg_plt = (truth_plt + ref_plt) / 2.0
                test_avg_plt = (truth_plt + test_plt) / 2.0

                ref_avg_ps = np.expand_dims((truth_ps + ref_ps) / 2.0, 0)
                test_avg_ps = np.expand_dims((truth_ps + test_ps) / 2.0, 0)

                ref_area = np.expand_dims(ref.variables['area'][...], 0)
                test_area = np.expand_dims(test.variables['area'][...], 0)

                ref_landfrac = np.expand_dims(ref.variables['LANDFRAC'][0, ...], 0)
                test_landfrac = np.expand_dims(test.variables['LANDFRAC'][0, ...], 0)

                for var in args.variables:
                    truth_var = truth.variables[var][0, ...]
                    ref_var = ref.variables[var][0, ...]
                    test_var = test.variables[var][0, ...]

                    ref_int = (ref_var - truth_var) ** 2 * ref_avg_plt
                    test_int = (test_var - truth_var) ** 2 * test_avg_plt

                    ref_global_l2 = np.sqrt((ref_int * ref_area).sum()
                                            / (ref_avg_ps * ref_area).sum())
                    ref_land_l2 = np.sqrt((ref_int * (ref_area * ref_landfrac)).sum()
                                          / (ref_avg_ps * (ref_area * ref_landfrac)).sum())
                    ref_ocean_l2 = np.sqrt((ref_int * (ref_area * (1 - ref_landfrac))).sum()
                                           / (ref_ps * (ref_area * (1 - ref_landfrac))).sum())

                    test_global_l2 = np.sqrt((test_int * test_area).sum()
                                             / (test_avg_ps * test_area).sum())
                    test_land_l2 = np.sqrt((test_int * (test_area * test_landfrac)).sum()
                                           / (test_avg_ps * (test_area * test_landfrac)).sum())
                    test_ocean_l2 = np.sqrt((test_int * (test_area * (1 - test_landfrac))).sum()
                                            / (test_ps * (test_area * (1 - test_landfrac))).sum())

                    # (case, ensemble, seconds, var, l2_global, l2_land, l2_ocean)
                    data.append([args.ref_case, instance, time, var,
                                 ref_global_l2, ref_land_l2, ref_ocean_l2])
                    data.append([args.test_case, instance, time, var,
                                 test_global_l2, test_land_l2, test_ocean_l2])

    tsc_df = pd.DataFrame(data, columns=['case', 'instance', 'seconds', 'variable',
                                         'l2_global', 'l2_land', 'l2_ocean'])

    # NOTE: This is what we're actually going to apply the one-sided t test to
    test_columns = ['l2_global', 'l2_land', 'l2_ocean']
    delta_columns = ['delta_' + t for t in test_columns]
    ref_columns = ['ref_' + t for t in test_columns]
    delta_rmsd = tsc_df[tsc_df['case'] == args.ref_case].copy()
    delta_rmsd[delta_columns] = tsc_df[tsc_df['case'] == args.test_case][test_columns].values \
                                - delta_rmsd[test_columns]
    delta_rmsd.rename(columns={t: r for t, r in zip(test_columns, ref_columns)}, inplace=True)

    # check for bit-for-bit results; nothing to plot if true b/c everything will be zero
    if not np.allclose(delta_rmsd[delta_columns], 0.0, atol=1e-12):
        # NOTE: This mess is because we want to "normalize" delta l2 by the mean
        # _reference_ l2 for each instance and variable.
        delta_rmsd[['norm_' + d for d in delta_columns]] = \
            delta_rmsd.groupby(['instance', 'variable']).apply(
                lambda g: g[delta_columns] / g[ref_columns].mean().values)

        testee = delta_rmsd.query(' seconds >= @args.time_slice[0] & seconds <= @args.time_slice[-1]')
        ttest = testee.groupby(['seconds', 'variable']).agg(stats.ttest_1samp, popmean=0.0).drop(columns='instance')

        # H0: enemble_mean_ΔRMSD_{t,var} is (statistically) zero and therefore, the simulations are identical
        null_hypothesis = ttest.applymap(lambda x: 'Reject' if x[1] < args.p_threshold else 'Accept')

        domains = (
            # True for rejection of null_hypothesis for each variable at each time, by comparing
            # index 1 (x[1]) of each column of tuples, which corresponds to the p-value to the
            # threshold for p-values
            ttest.applymap(lambda x: x[1] < args.p_threshold)
            # Select only times in the inspection window
            .query(' seconds >= @args.inspect_times[0] & seconds <= @args.inspect_times[-1]')
            # Create groups of all variables at each time step in the window
            .groupby("seconds")
            # Are any variables failing at each time step in the inspection window?
            .any()
            # Since True -> 1 False -> 0, .sum() gets the number of timesteps for which
            # the null hypothesis is rejected
            .sum()
            # If two or more time steps are failing then the domain [glob, lnd, ocn] is failing
            .transform(lambda x: "Fail" if x >= 2 else "Pass")
        )
        overall = 'Fail' if domains[delta_columns].apply(lambda x: x == 'Fail').any() else 'Pass'

        ttest.reset_index(inplace=True)
        null_hypothesis.reset_index(inplace=True)

        land_data = LIVVDict()
        ocean_data = LIVVDict()
        global_data = LIVVDict()
        for sec in ttest['seconds'].unique():
            for var in ttest['variable'].unique():
                sec_str = 'Time: {:03d}s'.format(sec)
                t_data = ttest.loc[(ttest['seconds'] == sec) & (ttest['variable'] == var)]
                h0_data = null_hypothesis.loc[(null_hypothesis['seconds'] == sec) & (null_hypothesis['variable'] == var)]
                global_data[sec_str][var]['Null hypothesis'] = h0_data['delta_l2_global'].values[0]
                land_data[sec_str][var]['Null hypothesis'] = h0_data['delta_l2_land'].values[0]
                ocean_data[sec_str][var]['Null hypothesis'] = h0_data['delta_l2_ocean'].values[0]

                global_data[sec_str][var]['T test (t, P)'] = '({:.3f}, {:.4f})'.format(*t_data['delta_l2_global'].values[0])
                land_data[sec_str][var]['T test (t, P)'] = '({:.3f}, {:.4f})'.format(*t_data['delta_l2_land'].values[0])
                ocean_data[sec_str][var]['T test (t, P)'] = '({:.3f}, {:.4f})'.format(*t_data['delta_l2_ocean'].values[0])

        details = {'ocean': ocean_data, 'land': land_data, 'global': global_data,
                   'domains': domains, 'overall': overall}

        fail_timeline_img = os.path.relpath(os.path.join(args.img_dir, 'failing_timeline.png'), os.getcwd())
        pmin_timeline_img = os.path.relpath(os.path.join(args.img_dir, 'pmin_timeline.png'), os.getcwd())
        rmsd_img_format = os.path.relpath(os.path.join(args.img_dir, '{}_rmsd_{{}}s.png'), os.getcwd())

        img_list = [plot_failing_variables(args, null_hypothesis, fail_timeline_img),
                    plot_pmin(args, ttest, pmin_timeline_img)]
        e_img_list = errorbars_delta_rmsd(args, delta_rmsd[delta_rmsd['seconds'].isin(args.inspect_times)],
                                          null_hypothesis, rmsd_img_format.format('distribution'))

        b_img_list = boxplot_delta_rmsd(args, delta_rmsd[delta_rmsd['seconds'].isin(args.inspect_times)],
                                        null_hypothesis, rmsd_img_format.format('boxplot'))

        img_list.extend([img for pair in zip(e_img_list, b_img_list) for img in pair])

    else:
        details = {'overall': 'Pass',
                   'global': {'Time: ALL': {'Vars: ALL': {'Null hypothesis': 'Accept',
                                                          'T test (t, P)': '(-, 1)'}}},
                   'land': {'Time: ALL': {'Vars: ALL': {'Null hypothesis': 'Accept',
                                                        'T test (t, P)': '(-, 1)'}}},
                   'ocean':  {'Time: ALL': {'Vars: ALL': {'Null hypothesis': 'Accept',
                                                          'T test (t, P)': '(-, 1)'}}},
                   'domains': {
                       'delta_l2_global': 'bit-for-bit',
                       'delta_l2_land': 'bit-for-bit',
                       'delta_l2_ocean': 'bit-for-bit',
                   }}
        img_list = plot_bit_for_bit(args)

    img_gallery = el.Gallery('Time step convergence', img_list)
    return details, img_gallery


def pressure_layer_thickness(dataset):
    p0 = dataset.variables['P0'][...]
    ps = dataset.variables['PS'][0, ...]
    hyai = dataset.variables['hyai'][...]
    hybi = dataset.variables['hybi'][...]

    da = hyai[1:] - hyai[0:-1]
    db = hybi[1:] - hybi[0:-1]

    dp = np.expand_dims(da * p0, 1) + (np.expand_dims(db, 1) * np.expand_dims(ps, 0))
    return dp, ps

def plot_thresholds(args, axis, x_thr=None, y_thr=None):
    """Add vertical / horiziontal lines on axis indicating certain thresholds."""
    if x_thr is None:
        x_thr = [args.inspect_times[0], args.inspect_times[-1]]
    if y_thr is None:
        y_thr = args.p_threshold * 100

    axis.axhline(y_thr, linestyle="--", linewidth=1, color="grey", zorder=-1)
    for _time in x_thr:
        axis.axvline(_time, linestyle="--", linewidth=1, color="grey", zorder=-1)


def plot_bit_for_bit(args):
    failing_img_file = os.path.relpath(os.path.join(args.img_dir, 'failing_timeline.png'), os.getcwd())
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', family='serif')

    xx = np.arange(0, args.time_slice[1] + args.time_slice[0],  args.time_slice[0])
    yy = np.zeros(xx.shape)
    ax.plot(xx, yy, linestyle='-', marker='o', color=pf_color_picker.get('pass'))
    plot_thresholds(args, ax, y_thr=1)

    ax.set_ybound(-1, 20)
    ax.set_yticks(np.arange(0, 24, 4))
    ax.set_yticks(np.arange(0, 21, 1), minor=True)

    ax.set_ylabel('Number of failing variables')
    ax.set_xlabel('Integration time (s)')

    plt.tight_layout()
    plt.savefig(failing_img_file, bbox_inches='tight')
    plt.close(fig)

    failing_img_caption = (
        "The number of failing variables across both domains (land and ocean) as a "
        "function of model integration time. The dashed horizontal line represents the "
        "failing threshold of 1 variable, the dashed vertical lines represent the inspection "
        f"window of {args.inspect_times[0]} - {args.inspect_times[-1]} s."
    )
    failing_img_link = Path(*Path(args.img_dir).parts[-2:], Path(failing_img_file).name)
    failing_img = el.Image(
        'Timeline of failing variables',
        failing_img_caption,
        failing_img_link,
        height=300,
        relative_to="",
        group="Timelines"
    )

    pmin_img_file = os.path.relpath(os.path.join(args.img_dir, 'pmin_timeline.png'), os.getcwd())
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', family='serif')

    ax.semilogy(xx, yy + 1.0, linestyle='-', marker='o', color=pf_color_picker.get('pass'))
    plot_thresholds(args, ax)

    ax.text(np.mean(args.time_slice), 10 ** -1, 'Fail', fontsize=15, color=pf_color_picker.get('fail'),
            horizontalalignment='center')
    ax.text(np.mean(args.time_slice), 0.5 * 10 ** 1, 'Pass', fontsize=15, color=pf_color_picker.get('pass'),
            horizontalalignment='center')

    ax.set_ybound(100, 10 ** -15)
    locmaj = tkr.LogLocator(numticks=18)
    ax.yaxis.set_major_locator(locmaj)
    locmin = tkr.LogLocator(subs=(0.2, 0.4, 0.6, 0.8), numticks=18)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(tkr.NullFormatter())
    ax.set_ylabel('P_{min} (%)')
    ax.set_xlabel('Integration time (s)')

    plt.tight_layout()
    plt.savefig(pmin_img_file, bbox_inches='tight')
    plt.close(fig)

    pmin_img_caption = (
        "The minimum P value of all variables in both domains (land and ocean) as a "
        "function of model integration time plotted with a logarithmic y-scale. The "
        "dashed horizontal grey line indicates the threshold for assigning an overall "
        "pass or fail to a test ensemble; the dashed vertical lines represent the "
        f"inspection window of {args.inspect_times[0]} - {args.inspect_times[-1]} s. "
        "see Wan et al. (2017) eqn. 8"
    )
    pmin_img_link = Path(*Path(args.img_dir).parts[-2:], Path(pmin_img_file).name)
    pmin_img = el.Image(
        'Timeline of P_{min}', pmin_img_caption, pmin_img_link, height=300, relative_to="", group="Timelines"
    )

    return [failing_img, pmin_img]


def plot_failing_variables(args, null_hypothesis, img_file):
    null_hypothesis[['n_fail_land', 'n_fail_ocean']] = \
        null_hypothesis[['delta_l2_land', 'delta_l2_ocean']].transform(lambda x: x == 'Reject').astype('int')

    pdata = null_hypothesis[['seconds', 'n_fail_land', 'n_fail_ocean']].groupby('seconds').sum().sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', family='serif')

    pdata.plot(linestyle='-', marker='o', color=pf_color_picker.get('fail'))
    ax.set_ybound(0, 20)
    ax.set_yticks(np.arange(0, 24, 4))
    ax.set_yticks(np.arange(0, 21, 1), minor=True)

    ax.set_ylabel('Number of failing variables')
    ax.set_xlabel('Integration time (s)')
    plot_thresholds(args, ax, y_thr=1)

    plt.tight_layout()
    plt.savefig(img_file, bbox_inches='tight')
    plt.close(fig)

    img_caption = (
        "The number of failing variables across both domains (land and ocean) as a "
        "function of model integration time. The dashed horizontal line represents the "
        "failing threshold of 1 variable, the dashed vertical lines represent the "
        f"inspection window of {args.inspect_times[0]} - {args.inspect_times[-1]} s."
    )
    img_link = Path(*Path(args.img_dir).parts[-2:], Path(img_file).name)
    img = el.Image(
        'Timeline of failing variables', img_caption, img_link, height=300, relative_to="", group="Timelines"
    )
    return img


def plot_pmin(args, ttest, img_file):
    ttest[['p_land', 'p_ocean']] = ttest[['delta_l2_land', 'delta_l2_ocean']].applymap(lambda x: x[1])
    pdata = ttest[['seconds', 'p_land', 'p_ocean']].groupby(['seconds']).min().min(axis=1) * 100  # to %

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', family='serif')

    fails = pdata[pdata.lt(args.p_threshold * 100)]
    passes = pdata[pdata.ge(args.p_threshold * 100)]

    if passes.empty:
        fails.plot(logy=True, linestyle='-', marker='o', color=pf_color_picker.get('fail'))
    elif fails.empty:
        passes.plot(logy=True, linestyle='-', marker='o', color=pf_color_picker.get('pass'))
    else:
        pdata.plot(logy=True, linestyle="-", color="black")
        passes.plot(logy=True, linestyle="None", marker="o", color=pf_color_picker.get("pass"))
        fails.plot(logy=True, linestyle="None", marker="o", color=pf_color_picker.get("fail"))

    plot_thresholds(args, ax)

    inspect_window = [args.inspect_times[0], args.inspect_times[-1]]
    ax.text(np.mean(inspect_window), 10 ** -1, 'Fail', fontsize=15, color=pf_color_picker.get('fail'),
            horizontalalignment='center')
    ax.text(np.mean(inspect_window), 10 ** 0, 'Pass', fontsize=15, color=pf_color_picker.get('pass'),
            horizontalalignment='center')

    ax.set_ybound(100, 10 ** -15)
    locmaj = tkr.LogLocator(numticks=18)
    ax.yaxis.set_major_locator(locmaj)
    locmin = tkr.LogLocator(subs=(0.2, 0.4, 0.6, 0.8), numticks=18)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(tkr.NullFormatter())
    ax.set_ylabel('P_{min} (%)')
    ax.set_xlabel('Integration time (s)')

    plt.tight_layout()
    plt.savefig(img_file, bbox_inches='tight')
    plt.close(fig)

    img_caption = (
        "The minimum P value of all variables in both domains (land and ocean) as a "
        "function of model integration time plotted with a logarithmic y-scale. The "
        "dashed horizontal grey line indicates the threshold for assigning an overall "
        "pass or fail to a test ensemble; the dashed vertical lines represent the "
        f"inspection window of {args.inspect_times[0]} - {args.inspect_times[-1]} s. "
        "see Wan et al. (2017) eqn. 8"
    )

    # img_link = os.path.join(os.path.basename(args.img_dir), os.path.basename(img_file))
    img_link = Path(*Path(args.img_dir).parts[-2:], Path(img_file).name)
    img = el.Image('Timeline of P_{min}', img_caption, img_link, height=300, relative_to="", group="Timelines")
    return img


def boxplot_delta_rmsd(args, delta_rmsd, null_hypothesis, img_file_format):
    img_list = []
    columns = ['instance', 'variable', 'norm_delta_l2_land', 'norm_delta_l2_ocean']
    for time in delta_rmsd['seconds'].unique():
        img_file = img_file_format.format(time)
        pdata = delta_rmsd[delta_rmsd['seconds'] == time][columns]

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex='all')
        plt.rc('font', family='serif')

        bp1 = pdata.boxplot(column='norm_delta_l2_land', by='variable', ax=ax1, vert=False, grid=False,
                            patch_artist=True, return_type='dict', showmeans=True,
                            meanprops={'markerfacecolor': 'black',
                                       'markeredgecolor': 'black',
                                       'marker': '.'})
        bp2 = pdata.boxplot(column='norm_delta_l2_ocean', by='variable', ax=ax2, vert=False, grid=False,
                            patch_artist=True, return_type='dict', showmeans=True,
                            meanprops={'markerfacecolor': 'black',
                                       'markeredgecolor': 'black',
                                       'marker': '.'})

        for box1, box2, var1, var2 in zip(bp1['norm_delta_l2_land']['boxes'],
                                          bp2['norm_delta_l2_ocean']['boxes'],
                                          list(ax1.get_yticklabels()),
                                          list(ax2.get_yticklabels())):

            land_var_color = pf_color_picker[
                null_hypothesis[(null_hypothesis['seconds'] == time)
                                & (null_hypothesis['variable'] == var1.get_text())]['delta_l2_land'].values[0]]
            ocean_var_color = pf_color_picker[
                null_hypothesis[(null_hypothesis['seconds'] == time)
                                & (null_hypothesis['variable'] == var2.get_text())]['delta_l2_ocean'].values[0]]

            var1.set_color(land_var_color)
            box1.set_color(land_var_color)
            box1.set_alpha(0.5)

            var2.set_color(ocean_var_color)
            box2.set_color(ocean_var_color)
            box2.set_alpha(0.5)

        for artist in ['fliers', 'medians', 'means', 'whiskers', 'caps']:
            for a1, a2 in zip(bp1['norm_delta_l2_land'][artist],
                              bp2['norm_delta_l2_ocean'][artist]):
                a1.set_color('black')
                a2.set_color('black')

        st = fig.suptitle("Ensemble ΔRMSD at t={}s".format(time), size=16)

        ax1.set_xlabel('Normalized ensemble ΔRMSD')
        ax1.set_title('Land')

        ax2.set_xlabel('Normalized ensemble ΔRMSD')
        ax2.set_title('Ocean')

        plt.savefig(img_file, bbox_extra_artists=[st], bbox_inches='tight')
        plt.close(fig)

        img_caption = 'Boxplot of ensemble ΔRMSD at t={time}s, where the dots show ' \
                      'the ensemble mean, ther verical lines show the ensemble median, ' \
                      'the filled boxes cover the interquartile range (IQR), and ' \
                      'the whiskers cover 1.5*IQR. All values here have been ' \
                      'normalized by the mean RMSD of the reference ensemble. ' \
                      '{cfail} and {cpass} indicate fail and pass, respectively, ' \
                      'according to the criterion described in ' \
                      'Wan et al. (2017), eq. 6.'.format(time=time,
                                                         cfail=human_color_names['fail'][0].capitalize(),
                                                         cpass=human_color_names['pass'][0])
        img_link = Path(*Path(args.img_dir).parts[-2:], Path(img_file).name)
        img_list.append(el.Image('Boxplot of normalized ensemble ΔRMSD at {}s'.format(time),
                                 img_caption, img_link, height=300, relative_to="", group="Boxplots"))
    return img_list


def errorbars_delta_rmsd(args, delta_rmsd, null_hypothesis, img_file_format):
    img_list = []
    columns = ['instance', 'variable', 'norm_delta_l2_land', 'norm_delta_l2_ocean']

    N = len(delta_rmsd['instance'].unique()) - 1
    tval_crit = stats.t.ppf(1 - args.p_threshold, df=N - 1)

    # NOTE: σ/√N is the t-test scaling parameter, see:
    #          https://en.wikipedia.org/wiki/Student%27s_t-test
    #       In the Wan et al. (2017) fig. 5, and the corresponding article text,
    #       indicates that these plots show ±2σ with the filled boxes, BUT they
    #       actually show ±2σ/√N (verified in old NCL code as well).
    scale_std = 1/np.sqrt(N)

    for time in delta_rmsd['seconds'].unique():
        img_file = img_file_format.format(time)
        pdata = delta_rmsd[delta_rmsd['seconds'] == time][columns].groupby('variable').agg(
                ['mean', np.std])

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex='all')
        plt.rc('font', family='serif')

        yvals = pdata.reset_index().index.values + 1
        ylims = [yvals[0] - 0.5, yvals[-1] + 0.5]
        land_colors = [pf_color_picker[
                           null_hypothesis[(null_hypothesis['seconds'] == time)
                                           & (null_hypothesis['variable'] == var)]['delta_l2_land'].values[0]]
                       for var in pdata.index.values]
        l_land_colors = [light_pf_color_picker[lc] for lc in land_colors]
        ocean_colors = [pf_color_picker[
                           null_hypothesis[(null_hypothesis['seconds'] == time)
                                           & (null_hypothesis['variable'] == var)]['delta_l2_ocean'].values[0]]
                        for var in pdata.index.values]
        l_ocean_colors = [light_pf_color_picker[oc] for oc in ocean_colors]

        et1 = ax1.errorbar(pdata['norm_delta_l2_land']['mean'].values, yvals,
                           xerr=np.stack([tval_crit * scale_std * pdata['norm_delta_l2_land']['std'].values,
                                          2 * scale_std * pdata['norm_delta_l2_land']['std'].values]),
                           fmt='k.', elinewidth=20, ecolor='lightblue')
        es1 = ax1.errorbar(pdata['norm_delta_l2_land']['mean'].values, yvals,
                           xerr=2 * scale_std * pdata['norm_delta_l2_land']['std'].values,
                           fmt='k.', elinewidth=20)

        et2 = ax2.errorbar(pdata['norm_delta_l2_ocean']['mean'].values, yvals,
                           xerr=np.stack([tval_crit * scale_std * pdata['norm_delta_l2_ocean']['std'].values,
                                          2 * scale_std * pdata['norm_delta_l2_ocean']['std'].values]),
                           fmt='k.', elinewidth=20, ecolor='lightblue')
        es2 = ax2.errorbar(pdata['norm_delta_l2_ocean']['mean'].values, yvals,
                           xerr=2 * scale_std * pdata['norm_delta_l2_ocean']['std'].values,
                           fmt='k.', elinewidth=20)

        _, _, (elines1,) = et1.lines
        elines1.set_color(l_land_colors)
        elines1.set_alpha(0.5)

        _, _, (elines1,) = es1.lines
        elines1.set_color(land_colors)
        elines1.set_alpha(0.5)

        ax1.plot([0, 0], ylims, 'k-', alpha=0.5)
        ax1.set_ylim(*ylims)
        ax1.set_yticks(yvals)
        ax1.set_yticklabels(pdata.index.values)

        _, _, (elines2,) = et2.lines
        elines2.set_color(l_ocean_colors)
        elines2.set_alpha(0.5)

        _, _, (elines2,) = es2.lines
        elines2.set_color(ocean_colors)
        elines2.set_alpha(0.5)

        ax2.plot([0, 0], ylims, 'k-', alpha=0.5)
        ax2.set_ylim(*ylims)
        ax2.set_yticks(yvals)
        ax2.set_yticklabels(pdata.index.values)

        for ii, var1, var2 in zip(yvals - 1,
                                  list(ax1.get_yticklabels()),
                                  list(ax2.get_yticklabels())):

            var1.set_color(land_colors[ii])
            var2.set_color(ocean_colors[ii])

        st = fig.suptitle("Ensemble ΔRMSD at t={}s".format(time), size=16)

        ax1.set_xlabel('Normalized ensemble ΔRMSD')
        ax1.set_title('Land')

        ax2.set_xlabel('Normalized ensemble ΔRMSD')
        ax2.set_title('Ocean')

        plt.savefig(img_file, bbox_extra_artists=[st], bbox_inches='tight')
        plt.close(fig)

        img_caption = 'Ensemble-mean ΔRMSD at t={time}s (dots) and the ±2σ/√N range of ' \
                      'the mean (dark filled boxes), where σ denotes the standard ' \
                      'deviation, N the ensemble size, and σ/√N is the t-test scaling ' \
                      'parameter. The left end of the light filled boxes shows the ' \
                      'threshold value corresponding to the critical P {pthres}% in the ' \
                      'one-sided t-test. All values here have been ' \
                      'normalized by the mean RMSD of the reference ensemble. ' \
                      '{cfail} and {cpass} indicate fail and pass, respectively, ' \
                      'according to the criterion described in ' \
                      'Wan et al. (2017), eq. 6.'.format(time=time,
                                                         pthres=args.p_threshold * 100,
                                                         cfail=human_color_names['fail'][0].capitalize(),
                                                         cpass=human_color_names['pass'][0])
        img_link = Path(*Path(args.img_dir).parts[-2:], Path(img_file).name)
        img_list.append(el.Image('Distribution of the ensemble ΔRMSD at {}s'.format(time),
                                 img_caption, img_link, height=300, relative_to="", group="Boxplots"))
    return img_list


def _print_details(details):
    for set_ in details:
        print('-' * 80)
        print(set_)
        print('-' * 80)
        pprint(details[set_])


def print_summary(summary):
    print('    Time step convergence test: {}'.format(summary['']['Case']))
    print('      Global: {}'.format(summary['']['Global']))
    print('      Land: {}'.format(summary['']['Land']))
    print('      Ocean: {}'.format(summary['']['Ocean']))
    print('      Ensembles: {}'.format(summary['']['Ensembles']))
    print('      Test status: {}\n'.format(summary['']['Test status']))


def summarize_result(results_page):
    summary = {'Case': results_page.title}
    for elem in results_page.elements:
        if isinstance(elem, el.Table) and elem.title == "Results":
            for group in ["Global", "Land", "Ocean", "Ensembles", "Test status"]:
                summary[group] = elem.data[group][0]
            break
    return {'': summary}


def populate_metadata():
    """
    Generates the metadata responsible for telling the summary what
    is done by this module's run method
    """

    metadata = {'Type': 'ValSummary',
                'Title': 'Validation',
                'TableTitle': 'Time step convergence test',
                'Headers': ['Test status', 'Global', 'Land', 'Ocean', 'Ensembles']}
    return metadata


if __name__ == '__main__':
    test_name, test_config = parse_args()
    run(test_name, test_config, print_details=True)
