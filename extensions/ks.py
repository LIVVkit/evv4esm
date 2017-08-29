#!/usr/bin/env python

"""
The K-S test.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import sys
import glob
import json
#import calendar
import argparse

#FIXME: Temporary! Remove before int. into eve.
import matplotlib
matplotlib.use('Agg')
    
import numpy as np
import pandas as pd
from scipy import stats
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from livvkit.util import elements as EL
from livvkit.util import functions as FN
from livvkit.util.LIVVDict import LIVVDict

_DEBUG = False

# from livvkit.util.cats import json_file
def json_file(f):
    with open(f) as jf:
        j = json.load(jf)
    return j


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        type=json_file,
                        help='A JSON config file containing a `ks` dictionary defining ' +
                             'the options. NOTE: command line options will override file options.')

    parser.add_argument('--case1',
                        default='default',
                        help='Name of case 1.')

    parser.add_argument('--dir1',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of case 1 files.')

    parser.add_argument('--set1-file',
                        default=os.path.join(os.getcwd(), 'input_files', 'list_default.txt'),
                        help='File containing set of ensemble members to bused for case 1.')

    parser.add_argument('--case2',
                        default='fast',
                        help='Name of case 2.')

    parser.add_argument('--dir2',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of case 2 files.')

    parser.add_argument('--set2-file',
                        default=os.path.join(os.getcwd(), 'input_files', 'list_fast.txt'),
                        help='File containing set of ensemble members to bused for case 2.')

    parser.add_argument('--critical',
                        default=13, type=float,
                        help='The critical value (desired significance level) for rejecting the ' +
                        'null hypothesis.')
    
    args, _ = parser.parse_known_args(args)

    # FIXME: Test more
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
    FN.mkdir_p(args.img_dir)

    details, img_gal = main(args)

    tbl_data = OrderedDict(sorted(details.items()))
    
    tbl_el = EL.vtable('Results', ['h0', 'KS', 'Welch', 'avg', 'std', 'max', 'min'], tbl_data)
   
    tl = [EL.tab('Table', element_list=[table_el]), EL.tab('Gallery', element_list=[img_gal])]

    # TODO: Put your analysis here
    return EL.page(name, utils.format_doc(__doc__), tab_list=tl)


def monthly_to_annual_avg(var_data, cal):
    if len(var_data) != 12:
        raise ValueError('Error! There are 12 months in a year; you passed in {} monthly averages.'.format(len(var_data)))
    
    # TODO: more advanced calendar handling
    if cal == 'ignore':
        # weight each month equally
        avg = np.sum(var_data) / 12.
    else:
        # for ii in range(0,12):
        #     _, days = calendar.monthrange(2017, ii+1)
        avg = None
    return avg


def case_files(args):
    ens_set1 = np.genfromtxt(args.set1_file, dtype=int)
    ens_set2 = np.genfromtxt(args.set2_file, dtype=int)
    
    # ensure unique case names for the dictionary
    key1 = args.case1
    key2 = args.case2
    if args.case1 == args.case2:
        key1 += '1'
        key2 += '2'
    
    f_sets = {}
    for key, case, dir_, set_ in zip([key1, key2], [args.case1, args.case2], [args.dir1, args.dir2], [ens_set1, ens_set2]):
        f = []
        for n in set_:
            base = '{c}_{n:04}/run/{c}_{n:04}.cam.h0.0001-12.nc'.format(c=case, n=n)
            f.append(os.path.join(dir_, base))
            glb = os.path.join(dir_, base.replace('0001-12', '0002-*'))
            f.extend(sorted(glob.glob(glb)))
        f_sets[key] = f

    if _DEBUG:
        for case, f_list in f_sets.items():
            with open(case+'.txt', 'w') as f:
                for line in f_list:
                    f.write('{}\n'.format(line))

    return f_sets


def prob_plot(args, var, averages, n_q, img_file):
    #NOTE: Following the methods described in
    #      https://stackoverflow.com/questions/43285752
    #      to create the Q-Q and P-P plots
    q = np.linspace(0,100,n_q+1)
    avgs1 = averages[args.case1][var]['annuals']
    avgs2 = averages[args.case2][var]['annuals']
    avgs_all = np.concatenate((avgs1, avgs2))
    avgs_min = np.min(avgs_all)
    avgs_max = np.max(avgs_all)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    plt.rc('font', family='serif')
  
    ax1.set_title('Q-Q Plot')
    ax1.set_xlabel('{} pdf'.format(args.case1))
    ax1.set_ylabel('{} pdf'.format(args.case2))

    #NOTE: Axis switched here from Q-Q plot because cdf reflects about the 1-1 line
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

    # Produce unity-based normalization of data for the Q-Q plots because
    # matplotlib can't handle small absolute values or data ranges. See 
    #     https://github.com/matplotlib/matplotlib/issues/6015
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
    """
    Print out a summary generated by this module's summarize_result method
    """
    raise NotImplementedError


def print_details(details):
    for set_ in details:
        print('-'*80)
        print(set_)
        print('-'*80)
        from pprint import pprint
        pprint(details[set_])


def summarize_result(result_page):
    """
    Provides a snapshot of the results of the analysis to be provided
    to the sumamry as well as being printed out in this module's
    print_summary method
    """
    from pprint import pprint
    pprint(result_page)
    
    raise NotImplementedError


def populate_metadata():
    """
    Generates the metadata responsible for telling the summary what
    is done by this module's run method
    """
    raise NotImplementedError


def main(args):
    ens_files = case_files(args)
    averages = LIVVDict()
    details = LIVVDict()
    for case, c_files in six.iteritems(ens_files):
        # Get monthly averages from files
        for file_ in c_files:
            member, date = [os.path.basename(file_).split('.')[s] for s in [0,-2]]
            month = int(date.split('-')[-1])

            try:
                data = Dataset(file_, 'r')
            except OSError as E:
                six.raise_from(BaseException('Could not open netCDF dataset: {}'.format(file_)), E)
            
            for var in data.variables.keys():
                if len(data.variables[var].shape) < 2 or var in ['time_bnds', 'date_written', 'time_written']:
                    continue
                elif 'ncol' not in data.variables[var].dimensions:
                    continue
                elif len(data.variables[var].shape) == 3:
                    averages[case][var][member][month] = np.mean(data.variables[var][0,:,:])
                elif len(data.variables[var].shape) == 2:
                    averages[case][var][member][month] = np.mean(data.variables[var][0,:])
        
        # calculate annual averages from data structure
        for var, members in six.iteritems(averages[case]):
            for m in members:
                monthly = [members[m][i] for i in range(1,13)]
                averages[case][var][m]['annual'] = monthly_to_annual_avg(monthly, cal='ignore')
        
        # array of annual averages for 
        for var in averages[case]:
            averages[case][var]['annuals'] = np.array([averages[case][var][m]['annual'] for m in sorted(six.iterkeys(averages[case][var]))])

    # now, we got the data, so let's get some stats
    var_set1 = set([var for var in averages[args.case1]])
    var_set2 = set([var for var in averages[args.case2]])
    common_vars = list( var_set1 & var_set2 ) 

    img_list = []
    for var in sorted(common_vars):
        details[var]['Welch'] = stats.ttest_ind(averages[args.case1][var]['annuals'], 
                                                 averages[args.case2][var]['annuals'], 
                                                 equal_var=False, nan_policy='omit')

        details[var]['KS'] = stats.ks_2samp(averages[args.case1][var]['annuals'], 
                                            averages[args.case2][var]['annuals'])

        details[var]['avg'] = (np.mean(averages[args.case1][var]['annuals']), 
                               np.mean(averages[args.case2][var]['annuals']))

        details[var]['max'] = (np.max(averages[args.case1][var]['annuals']), 
                               np.max(averages[args.case2][var]['annuals']))

        details[var]['min'] = (np.min(averages[args.case1][var]['annuals']), 
                               np.min(averages[args.case2][var]['annuals']))

        details[var]['std'] = (np.std(averages[args.case1][var]['annuals']), 
                               np.std(averages[args.case2][var]['annuals']))

        details[var]['h0'] = 'reject' if details[var]['ks'] < 0.05 else 'accept'

        img_file = os.path.relpath(os.path.join(args.img_dir, var + '.png'), os.getcwd())
        prob_plot(args, var, averages, 20, img_file)
        
        img_desc = 'Mean annual global average of {} for {} is {:.3e} and for {} is {:.3e}'.format(
                        var, args.case1, details[var]['avg'][0], args.case2, details[var]['avg'][1])

        img_list.append(EL.image(var, img_desc, img_file))
        
    img_gal = EL.gallery('Analysed variables', img_list)

    if _DEBUG:
        FN.write_json(details, '.', 'details.json')
        FN.write_json(averages, '.', 'avgs.json')
        FN.write_json(args, '.', 'args.json')
    
    return (details, img_gal)

if __name__ == '__main__':
    args = parse_args()
    args.img_dir = '.'
    
    print_details(main(args))
