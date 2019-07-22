#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2015,2016, UT-BATTELLE, LLC
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
This tests...
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import os
from collections import OrderedDict
from itertools import groupby
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from netCDF4 import Dataset

import livvkit
from livvkit.util import elements as el
from livvkit.util import functions as fn
from livvkit.util.LIVVDict import LIVVDict

from evv4esm.ensembles import e3sm


# FIXME: CONSTANTS that probably aren't needed
T_THRESHOLD = 3.106
P_THRESHOLD = 0.005


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

    land_tbl_el = {'Type': 'V-H Table',
                   'Title': 'Validation',
                   'TableTitle': 'Analyzed variables',
                   'Headers': test_args.variables,
                   'Data': {'': details['land']}
                   }
    ocean_tbl_el = {'Type': 'V-H Table',
                    'Title': 'Validation',
                    'TableTitle': 'Analyzed variables',
                    'Headers': test_args.variables,
                    'Data': {'': details['ocean']}
                    }
    tab_list = [el.tab('Gallery', element_list=[img_gal]),
                el.tab('Land_table', element_list=[land_tbl_el]),
                el.tab('Ocean_table', element_list=[ocean_tbl_el])]

    page = el.page(name, __doc__, tab_list=tab_list)
    page['domains'] = details['domains'].to_dict()
    page['overall'] = details['overall']

    return page


def main(args):
    if args.test_case == args.ref_case:
        args.test_case += '1'
        args.ref_case += '2'

    file_search_glob = '{d}/*.cam_????.h0.0001-01-01-?????.nc.{s}'
    truth_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.ref_dir, s='DT0001'))),
            key=lambda f: e3sm.component_file_instance('cam', f))}

    ref_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.ref_dir, s='DT0002'))),
            key=lambda f: e3sm.component_file_instance('cam', f))}

    test_ens = {instance: list(files) for instance, files in groupby(
            sorted(glob.glob(file_search_glob.format(d=args.test_dir, s='DT0002'))),
            key=lambda f: e3sm.component_file_instance('cam', f))}

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

    # NOTE: This mess is because we want to "normalize" both the test l2 and
    # reference l2 by the mean _reference_ l2 for each instance and variable.
    tsc_df[['norm_l2_global', 'norm_l2_land', 'norm_l2_ocean']] = \
        tsc_df.groupby(['instance', 'variable']).apply(
            lambda g: g[['l2_global', 'l2_land', 'l2_ocean']]
                      / g[g['case'] == args.ref_case][['l2_global', 'l2_land', 'l2_ocean']].mean())

    # NOTE: This is what we're actually going to apply the one-sided t test to
    test_columns = ['l2_global', 'l2_land', 'l2_ocean', 'norm_l2_global', 'norm_l2_land', 'norm_l2_ocean']
    delta_rmsd = tsc_df[tsc_df['case'] == args.test_case].copy()
    delta_rmsd[test_columns] = delta_rmsd[test_columns] - tsc_df[tsc_df['case'] == args.ref_case][test_columns].values
    delta_rmsd.rename(columns={c: 'delta_' + c for c in test_columns}, inplace=True)

    testee = delta_rmsd.query(' seconds >= @args.time_slice[0] & seconds <= @args.time_slice[1]')
    ttest = testee.groupby(['seconds', 'variable']).agg(ttest_1samp, popmean=0.0).drop(columns='instance')

    # H0: enemble_mean_ΔRMSD_{t,var} is (statistically) zero and therefore, the simulations are identical
    null_hypothesis = ttest.applymap(lambda x: 'Reject' if x[1] < P_THRESHOLD else 'Accept')

    domains = null_hypothesis.applymap(lambda x: x == 'Reject').any().transform(lambda x: 'Fail' if x is True else 'Pass')
    overall = 'Fail' if domains.apply(lambda x: x == 'Fail').any() else 'Pass'

    ttest.reset_index(inplace=True)
    null_hypothesis.reset_index(inplace=True)

    land_data = LIVVDict()
    ocean_data = LIVVDict()
    for sec in ttest['seconds'].unique():
        for var in ttest['variable'].unique():
            t_data = ttest.loc[(ttest['seconds'] == sec) & (ttest['variable'] == var)]
            h0_data = null_hypothesis.loc[(null_hypothesis['seconds'] == sec) & (null_hypothesis['variable'] == var)]
            land_data[sec][var] = 'h0: {}, T test (t, p): ({}, {})'.format(
                    h0_data['delta_l2_land'].values[0], *t_data['delta_l2_land'].values[0])
            ocean_data[sec][var] = 'h0: {}, T test (t, p): ({}, {})'.format(
                    h0_data['delta_l2_ocean'].values[0], *t_data['delta_l2_ocean'].values[0])


    details = {'ocean': ocean_data, 'land': land_data,
               'domains': domains, 'overall': overall}

    img_gallery = el.gallery('Time step convergence', [])

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
    print('      Ensembles: {}\n'.format(summary['']['Ensembles']))


def summarize_result(results_page):
    summary = {'Case': results_page['Title']}
    summary['Global'] = results_page['domains']['delta_l2_global']
    summary['Land'] = results_page['domains']['delta_l2_land']
    summary['Ocean'] = results_page['domains']['delta_l2_ocean']
    summary['Ensembles'] = 'identical' if results_page['overall'] == 'Pass' else 'distinct'
    return {'': summary}


def populate_metadata():
    """
    Generates the metadata responsible for telling the summary what
    is done by this module's run method
    """

    metadata = {'Type': 'ValSummary',
                'Title': 'Validation',
                'TableTitle': 'Time step convergence test',
                'Headers': ['Global', 'Land', 'Ocean', 'Ensembles']}
    return metadata


if __name__ == '__main__':
    test_name, test_config = parse_args()
    run(test_name, test_config, print_details=True)
