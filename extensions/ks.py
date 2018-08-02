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

"""The Kolmogorov-Smirnov Test:
This tests the null hypothesis that the baseline (n) and modified (m) model
Short Independent Simulation Ensembles (SISE) represent the same climate
state, based on the equality of distribution of each variable's annual global
average in the standard monthly model output between the two simulations.

The (per variable) null hypothesis uses the non-parametric, two-sample (n and m)
Kolmogorov-Smirnov test as the univariate test of of equality of distribution of
global means. The test statistic (t) is the number of variables that reject the
(per variable) null hypothesis of equality of distribution at a 95% confidence
level. The (overall) null hypothesis is rejected if t > α, where α is some
critical number of rejecting variables. The critical value, α, is obtained from
an empirically derived approximate null distribution of t using resampling
techniques.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import re
import glob
import argparse

from pprint import pprint
from collections import OrderedDict
    
import numpy as np
import pandas as pd
from scipy import stats
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import livvkit
from livvkit.util import elements as el
from livvkit.util import functions as fn
from livvkit.util.LIVVDict import LIVVDict


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        type=fn.read_json,
                        help='A JSON config file containing a `ks` dictionary defining ' +
                             'the options. NOTE: command line options will override file options.')

    parser.add_argument('--case1',
                        default='default',
                        help='Name of case 1.')

    parser.add_argument('--dir1',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of case 1 files.')

    parser.add_argument('--case2',
                        default='fast',
                        help='Name of case 2.')

    parser.add_argument('--dir2',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of case 2 files.')

    # noinspection PyTypeChecker
    parser.add_argument('--ninst',
                        default=30, type=int,
                        help='The number of instances in both cases.')

    parser.add_argument('--critical',
                        default=13, type=float,
                        help='The critical value (desired significance level) for rejecting the ' +
                        'null hypothesis.')
   
    parser.add_argument('--img-dir',
                        default=os.getcwd(),
                        help='Image output location.')

    args, _ = parser.parse_known_args(args)

    # use config file arguments, but override with command line arguments
    if args.config:
        default_args = parser.parse_args([])

        for key, val, in vars(args).items():
            if val != vars(default_args)[key]:
                args.config['ks'][key] = val

        config_arg_list = []
        [config_arg_list.extend(['--'+key, str(val)]) for key, val in args.config['ks'].items()
         if key != 'config']
        args, _ = parser.parse_known_args(config_arg_list)

    return args


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
    [config_arg_list.extend(['--'+key, str(val)]) for key, val in config.items()]

    args = parse_args(config_arg_list)

    args.img_dir = os.path.join(livvkit.output_dir, 'validation', 'imgs', name)
    fn.mkdir_p(args.img_dir)

    details, img_gal = main(args)

    tbl_data = OrderedDict(sorted(details.items()))
    
    tbl_el = {'Type': 'V-H Table',
              'Title': 'Validation',
              'TableTitle': 'Analyzed variables',
              'Headers': ['h0', 'K-S test (D, p)', 'T test (t, p)'],
              'Data': {'': tbl_data}
              }
    
    tl = [el.tab('Table', element_list=[tbl_el]), el.tab('Gallery', element_list=[img_gal])]

    page = el.page(name, __doc__, tab_list=tl)
    page['critical'] = args.critical

    return page


def monthly_to_annual_avg(var_data, cal='ignore'):
    if len(var_data) != 12:
        raise ValueError('Error! There are 12 months in a year; '
                         'you passed in {} monthly averages.'.format(len(var_data)))
    
    # TODO: more advanced calendar handling
    if cal == 'ignore':
        # weight each month equally
        avg = np.sum(var_data) / 12.
    else:
        avg = None
    return avg


def file_instance(case_file):
    return int(re.search(r'cam_[0-9]+', case_file).group(0).replace('cam_', ''))


def file_date_str(case_file):
    return re.search(r'h0\.[0-9]+-[0-9]+.nc', case_file).group(0).replace('h0.', '').replace('.nc', '')


def case_files(args):
    # ensure unique case names for the dictionary
    key1 = args.case1
    key2 = args.case2
    if args.case1 == args.case2:
        key1 += '1'
        key2 += '2'
    
    f_sets = {}
    for key, case, dir_ in zip([key1, key2], [args.case1, args.case2], [args.dir1, args.dir2]):
        f = []
        base = '{d}/*cam_????.h0.????-??.nc'.format(d=dir_)
        glb = os.path.normpath(base)
        f.extend(sorted(glob.glob(glb)))
        f_i = OrderedDict()
        for ii in range(1, args.ninst+1):
            f_i[ii] = sorted(filter(lambda x: file_instance(x) == ii, f), key=file_date_str)
            if len(f_i[ii]) > 12:
                f_i[ii] = f_i[ii][-12:]

        f_sets[key] = f_i

    return f_sets, key1, key2


def prob_plot(args, var, averages, n_q, img_file):
    # NOTE: Following the methods described in
    #       https://stackoverflow.com/questions/43285752
    #       to create the Q-Q and P-P plots
    q = np.linspace(0, 100, n_q+1)
    avgs1 = averages[args.case1][var]['annuals']
    avgs2 = averages[args.case2][var]['annuals']
    avgs_all = np.concatenate((avgs1, avgs2))
    avgs_min = np.min(avgs_all)
    avgs_max = np.max(avgs_all)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    plt.rc('font', family='serif')
  
    ax1.set_title('Q-Q Plot')
    ax1.set_xlabel('{} pdf'.format(args.case1))
    ax1.set_ylabel('{} pdf'.format(args.case2))

    # NOTE: Axis switched here from Q-Q plot because cdf reflects about the 1-1 line
    ax2.set_title('P-P Plot')
    ax2.set_xlabel('{} cdf'.format(args.case2))
    ax2.set_ylabel('{} cdf'.format(args.case1))
    
    ax3.set_title('{} pdf'.format(args.case1))
    ax3.set_xlabel('Unity-based normalization of annual global averages')
    ax3.set_ylabel('Frequency')

    ax4.set_title('{} pdf'.format(args.case2))
    ax4.set_xlabel('Unity-based normalization of annual global averages')
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
    if not np.allclose(avgs_min, avgs_max, atol=np.finfo(avgs_max).eps):
        norm1 = (avgs1 - avgs_min)/(avgs_max - avgs_min)
        norm2 = (avgs2 - avgs_min)/(avgs_max - avgs_min)
        
        ax1.scatter(np.percentile(norm1, q), np.percentile(norm2, q), zorder=2)
        ax3.hist(norm1, bins=n_q)
        ax4.hist(norm2, bins=n_q)
    
    # bin both series into equal bins and get cumulative counts for each bin
    bnds = np.linspace(avgs_min, avgs_max, n_q)
    if not np.allclose(bnds, bnds[0], atol=np.finfo(bnds[0]).eps):
        ppxb = pd.cut(avgs1, bnds)
        ppyb = pd.cut(avgs2, bnds)

        ppxh = ppxb.value_counts().sort_index(ascending=True)/len(ppxb)
        ppyh = ppyb.value_counts().sort_index(ascending=True)/len(ppyb)

        ppxh = np.cumsum(ppxh)
        ppyh = np.cumsum(ppyh)

        ax2.scatter(ppyh, ppxh, zorder=2)
    
    plt.tight_layout()
    plt.savefig(img_file, bbox_inches='tight')

    plt.close(fig)

    return img_file


def print_summary(summary):
    print('    Kolmogorov-Smirnov Test: {}'.format(summary['']['Case']))
    print('      Variables analyzed: {}'.format(summary['']['Variables Analyzed']))
    print('      Rejecting: {}'.format(summary['']['Rejecting']))
    print('      Critical value: {}'.format(summary['']['Critical Value']))
    print('      Ensembles: {}\n'.format(summary['']['Ensembles']))


def print_details(details):
    for set_ in details:
        print('-'*80)
        print(set_)
        print('-'*80)
        pprint(details[set_])


def summarize_result(results_page):
    summary = {'Case': results_page['Title']}
    for tab in results_page['Data']['Tabs']:
        for elem in tab['Elements']:
            if elem['Type'] == 'V-H Table':
                summary['Variables Analyzed'] = len(elem['Data'][''].keys())
                rejects = [var for var, dat in elem['Data'][''].items() if dat['h0'] == 'reject']
                summary['Rejecting'] = len(rejects)
                summary['Critical Value'] = results_page['critical']
                summary['Ensembles'] = 'identical' if len(rejects) < results_page['critical'] else 'distinct'
                break
        else:
            continue
    return {'': summary}


def populate_metadata():
    """
    Generates the metadata responsible for telling the summary what
    is done by this module's run method
    """
    
    metadata = {'Type': 'ValSummary',
                'Title': 'Validation',
                'TableTitle': 'Kolmogorov-Smirnov',
                'Headers': ['Variables Analyzed', 'Rejecting', 'Critical Value', 'Ensembles']}
    return metadata
    

def main(args):
    ens_files, key1, key2 = case_files(args)
    if args.case1 == args.case2:
        args.case1 = key1
        args.case2 = key2

    averages = LIVVDict()
    details = LIVVDict()
    for case, inst_dict in six.iteritems(ens_files):
        for inst, i_files in six.iteritems(inst_dict):
            # Get monthly averages from files
            for file_ in i_files:
                date_str = file_date_str(file_)

                data = None
                try:
                    data = Dataset(file_)
                except OSError as E:
                    six.raise_from(BaseException('Could not open netCDF dataset: {}'.format(file_)), E)

                for var in data.variables.keys():
                    if len(data.variables[var].shape) < 2 or var in ['time_bnds', 'date_written', 'time_written']:
                        continue
                    elif 'ncol' not in data.variables[var].dimensions:
                        continue
                    elif len(data.variables[var].shape) == 3:
                        averages[case][var]['{:04}'.format(inst)][date_str] = np.mean(data.variables[var][0, :, :])
                    elif len(data.variables[var].shape) == 2:
                        averages[case][var]['{:04}'.format(inst)][date_str] = np.mean(data.variables[var][0, :])

        # calculate annual averages from data structure
        for var, instances in six.iteritems(averages[case]):
            for inst in instances:
                months = [instances[inst][date] for date in instances[inst]]
                averages[case][var][inst]['annual'] = monthly_to_annual_avg(months)

        # array of annual averages for
        for var in averages[case]:
                averages[case][var]['annuals'] = np.array(
                        [averages[case][var][m]['annual'] for m in sorted(six.iterkeys(averages[case][var]))])

    # now, we got the data, so let's get some stats
    var_set1 = set([var for var in averages[args.case1]])
    var_set2 = set([var for var in averages[args.case2]])
    common_vars = list(var_set1 & var_set2)

    img_list = []
    for var in sorted(common_vars):
        details[var]['T test (t, p)'] = stats.ttest_ind(averages[args.case1][var]['annuals'],
                                                        averages[args.case2][var]['annuals'],
                                                        equal_var=False, nan_policy=str('omit'))
        if np.isnan(details[var]['T test (t, p)']).any() or np.isinf(details[var]['T test (t, p)']).any():
            details[var]['T test (t, p)'] = (None, None)

        details[var]['K-S test (D, p)'] = stats.ks_2samp(averages[args.case1][var]['annuals'],
                                                         averages[args.case2][var]['annuals'])

        details[var]['mean (case 1, case 2)'] = (np.mean(averages[args.case1][var]['annuals']),
                                                 np.mean(averages[args.case2][var]['annuals']))

        details[var]['max (case 1, case 2)'] = (np.max(averages[args.case1][var]['annuals']),
                                                np.max(averages[args.case2][var]['annuals']))

        details[var]['min (case 1, case 2)'] = (np.min(averages[args.case1][var]['annuals']),
                                                np.min(averages[args.case2][var]['annuals']))

        details[var]['std (case 1, case 2)'] = (np.std(averages[args.case1][var]['annuals']),
                                                np.std(averages[args.case2][var]['annuals']))

        if details[var]['T test (t, p)'][0] is None:
            details[var]['h0'] = '-'
        elif details[var]['K-S test (D, p)'][1] < 0.05:
            details[var]['h0'] = 'reject'
        else:
            details[var]['h0'] = 'accept'

        img_file = os.path.relpath(os.path.join(args.img_dir, var + '.png'), os.getcwd())
        prob_plot(args, var, averages, 20, img_file)
        
        img_desc = 'Mean annual global average of {} for <em>{}</em> is {:.3e} and for <em>{}</em> is {:.3e}'.format(
                        var, args.case1, details[var]['mean (case 1, case 2)'][0],
                        args.case2, details[var]['mean (case 1, case 2)'][1])

        img_link = os.path.join(os.path.basename(args.img_dir), os.path.basename(img_file))
        img_list.append(el.image(var, img_desc, img_link))
        
    img_gal = el.gallery('Analyzed variables', img_list)

    return details, img_gal


if __name__ == '__main__':
    print_details(main(parse_args()))
