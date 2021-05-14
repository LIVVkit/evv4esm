#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2018-2021 UT-BATTELLE, LLC
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

The (per variable) null hypothesis uses the non-parametric, two-sample (n and m)
Kolmogorov-Smirnov test as the univariate test of equality of distribution of
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
import argparse

from pprint import pprint
from collections import OrderedDict
    
import numpy as np
from scipy import stats

import livvkit
from livvkit.util import elements as el
from livvkit.util import functions as fn
from livvkit.util.LIVVDict import LIVVDict

from evv4esm.ensembles import e3sm
from evv4esm.ensembles.tools import monthly_to_annual_avg, prob_plot
from evv4esm.utils import bib2html
from evv4esm import human_color_names, EVVException


def variable_set(name):
    var_sets = fn.read_json(os.path.join(os.path.dirname(__file__),
                                         'ks_vars.json'))
    try:
        the_set = var_sets[name.lower()]
        return set(the_set)
    except KeyError as e:
        six.raise_from(argparse.ArgumentTypeError(
                'Unknown variable set! Known sets are {}'.format(
                        var_sets.keys()
                )), e)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        type=fn.read_json,
                        help='A JSON config file containing a `ks` dictionary defining ' +
                             'the options. NOTE: command line options will override file options.')

    parser.add_argument('--test-case',
                        default='default',
                        help='Name of the test case.')

    parser.add_argument('--test-dir',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of the test case run files.')

    parser.add_argument('--ref-case',
                        default='fast',
                        help='Name of the reference case.')

    parser.add_argument('--ref-dir',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of the reference case run files.')

    parser.add_argument('--var-set',
                        default='default', type=variable_set,
                        help='Name of the variable set to analyze.')

    parser.add_argument('--ninst',
                        default=30, type=int,
                        help='The number of instances (should be the same for '
                             'both cases).')

    parser.add_argument('--critical',
                        default=13, type=float,
                        help='The critical value (desired significance level) for rejecting the ' +
                        'null hypothesis.')
   
    parser.add_argument('--img-dir',
                        default=os.getcwd(),
                        help='Image output location.')

    parser.add_argument('--component',
                        default='eam',
                        help='Model component name (e.g. eam, cam, ...)')

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
    bib_html = bib2html(os.path.join(os.path.dirname(__file__), 'ks.bib'))
    tl = [el.tab('Figures', element_list=[img_gal]),
          el.tab('Details', element_list=[tbl_el]),
          el.tab('References', element_list=[el.html(bib_html)])]

    rejects = [var for var, dat in tbl_data.items() if dat['h0'] == 'reject']
    results = {'Type': 'Table',
               'Title': 'Results',
               'Headers': ['Test status', 'Variables analyzed', 'Rejecting', 'Critical value', 'Ensembles'],
               'Data': {'Test status': 'pass' if len(rejects) < args.critical else 'fail',
                        'Variables analyzed': len(tbl_data.keys()),
                        'Rejecting': len(rejects),
                        'Critical value': args.critical,
                        'Ensembles': 'statistically identical' if len(rejects) < args.critical else 'statistically different'}
               }

    # FIXME: Put into a ___ function
    page = el.page(name, __doc__.replace('\n\n', '<br><br>'), element_list=[results], tab_list=tl)
    return page


def case_files(args):
    # ensure unique case names for the dictionary
    key1 = args.test_case
    key2 = args.ref_case
    if args.test_case == args.ref_case:
        key1 += '1'
        key2 += '2'

    f_sets = {key1: e3sm.component_monthly_files(args.test_dir, args.component, args.ninst),
              key2: e3sm.component_monthly_files(args.ref_dir, args.component, args.ninst)}

    for key in f_sets:
        # Require case files for at least the last 12 months.
        if any(list(map(lambda x: x == [], f_sets[key].values()))[-12:]):
            raise EVVException('Could not find all the required case files for case: {}'.format(key))

    return f_sets, key1, key2


def print_summary(summary):
    print('    Kolmogorov-Smirnov Test: {}'.format(summary['']['Case']))
    print('      Variables analyzed: {}'.format(summary['']['Variables analyzed']))
    print('      Rejecting: {}'.format(summary['']['Rejecting']))
    print('      Critical value: {}'.format(summary['']['Critical value']))
    print('      Ensembles: {}'.format(summary['']['Ensembles']))
    print('      Test status: {}\n'.format(summary['']['Test status']))


def print_details(details):
    for set_ in details:
        print('-'*80)
        print(set_)
        print('-'*80)
        pprint(details[set_])


def summarize_result(results_page):
    summary = {'Case': results_page['Title']}
    for elem in results_page['Data']['Elements']:
        if elem['Type'] == 'Table' and elem['Title'] == 'Results':
            summary['Test status'] = elem['Data']['Test status']
            summary['Variables analyzed'] = elem['Data']['Variables analyzed']
            summary['Rejecting'] = elem['Data']['Rejecting']
            summary['Critical value'] = elem['Data']['Critical value']
            summary['Ensembles'] = elem['Data']['Ensembles']
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
                'TableTitle': 'Kolmogorov-Smirnov test',
                'Headers': ['Test status', 'Variables analyzed', 'Rejecting', 'Critical value', 'Ensembles']}
    return metadata
    

def main(args):
    ens_files, key1, key2 = case_files(args)
    if args.test_case == args.ref_case:
        args.test_case = key1
        args.ref_case = key2

    monthly_avgs = e3sm.gather_monthly_averages(ens_files, args.var_set)
    annual_avgs = monthly_avgs.groupby(['case', 'variable', 'instance']
                                       ).monthly_mean.aggregate(monthly_to_annual_avg).reset_index()

    # now, we got the data, so let's get some stats
    test_set = set(monthly_avgs[monthly_avgs.case == args.test_case].variable.unique())
    ref_set = set(monthly_avgs[monthly_avgs.case == args.ref_case].variable.unique())
    common_vars = list(test_set & ref_set)
    if not common_vars:
        raise EVVException('No common variables between {} and {} to analyze!'.format(args.test_case, args.ref_case))

    img_list = []
    details = LIVVDict()
    for var in sorted(common_vars):
        annuals_1 = annual_avgs.query('case == @args.test_case & variable == @var').monthly_mean.values
        annuals_2 = annual_avgs.query('case == @args.ref_case & variable == @var').monthly_mean.values

        details[var]['T test (t, p)'] = stats.ttest_ind(annuals_1, annuals_2,
                                                        equal_var=False, nan_policy=str('omit'))
        if np.isnan(details[var]['T test (t, p)']).any() or np.isinf(details[var]['T test (t, p)']).any():
            details[var]['T test (t, p)'] = (None, None)

        details[var]['K-S test (D, p)'] = stats.ks_2samp(annuals_1, annuals_2)

        details[var]['mean (test case, ref. case)'] = (annuals_1.mean(), annuals_2.mean())

        details[var]['max (test case, ref. case)'] = (annuals_1.max(), annuals_2.max())

        details[var]['min (test case, ref. case)'] = (annuals_1.min(), annuals_2.min())

        details[var]['std (test case, ref. case)'] = (annuals_1.std(), annuals_2.std())

        if details[var]['T test (t, p)'][0] is None:
            details[var]['h0'] = '-'
        elif details[var]['K-S test (D, p)'][1] < 0.05:
            details[var]['h0'] = 'reject'
        else:
            details[var]['h0'] = 'accept'

        img_file = os.path.relpath(os.path.join(args.img_dir, var + '.png'), os.getcwd())
        prob_plot(annuals_1, annuals_2, 20, img_file, test_name=args.test_case, ref_name=args.ref_case,
                  pf=details[var]['h0'])
        
        img_desc = 'Mean annual global average of {var} for <em>{testcase}</em> ' \
                   'is {testmean:.3e} and for <em>{refcase}</em> is {refmean:.3e}. ' \
                   'Pass (fail) is indicated by {cpass} ({cfail}) coloring of the ' \
                   'plot markers and bars.'.format(var=var,
                                                   testcase=args.test_case,
                                                   testmean=details[var]['mean (test case, ref. case)'][0],
                                                   refcase=args.ref_case,
                                                   refmean=details[var]['mean (test case, ref. case)'][1],
                                                   cfail=human_color_names['fail'][0],
                                                   cpass=human_color_names['pass'][0])

        img_link = os.path.join(os.path.basename(args.img_dir), os.path.basename(img_file))
        img_list.append(el.image(var, img_desc, img_link))
        
    img_gal = el.gallery('Analyzed variables', img_list)

    return details, img_gal


if __name__ == '__main__':
    print_details(main(parse_args()))
