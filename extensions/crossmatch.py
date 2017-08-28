"""The Cross Match Test:
This tests the null hypothesis that the baseline (n) and modified (m) model Short Independent
Simulation Ensembles (SISE) belong to the same population. The standardized seasonal global annual
means of all output variables are concatenated into a single multi-variable vector for each ensemble
member and pooled into a single set (size N = n + m). Each vector in the resulting set is optimally
paired with the vector closest to it, using the Mahalanobis distance. The cross match test
statistic, T, is the number of pairs with one vector from each of the control and perturbed
ensembles.

The null hypothesis is rejected if T > t, for a critical value t (the desired significance
level). When the baseline and the perturbed distributions are similar, cross matches should occur
more frequently, resulting in a larger T.

Note: because T is based on simple combinatorics, it does not depend on the assumed distribution of
the baseline or perturbed data vectors (Rosenbaum, 2005).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import numpy
import argparse

from collections import OrderedDict

from eve import utils
from livvkit.util import elements
from livvkit.util import functions 

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()


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
                        help='A JSON config file containing a `crossmatch` dictionary defining ' +
                             'the options. NOTE: command line options will override file options.')

    parser.add_argument('--R-dir',
                        default=os.path.join(os.getcwd(), 'R'),
                        help='The location of the associated R source code.')

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

    parser.add_argument('--set2-file',
                        default=os.path.join(os.getcwd(), 'input_files', 'list_fast.txt'),
                        help='File containing set of ensemble members to bused for case 2.')

    parser.add_argument('--dir2',
                        default=os.path.join(os.getcwd(), 'archive'),
                        help='Location of case 2 files.')

    parser.add_argument('--season',
                        default='SON', type=str.upper,
                        choices=['DJF', 'MAM', 'JJA', 'SON'],
                        help='The season.')

    parser.add_argument('--critical',
                        default=0.05, type=float,
                        help='The critical value (desired significance level) for rejecting the ' +
                        'null hypothesis.')

    args, _ = parser.parse_known_args(args)

    # FIXME: Test more
    # use config file arguments, but override with command line arguments
    if args.config:
        default_args = parser.parse_args([])

        for key, val, in vars(args).items():
            if val != vars(default_args)[key]:
                args.config['crossmatch'][key] = val

        config_arg_list = []
        [config_arg_list.extend(['--'+key, str(val)]) for key, val in args.config['crossmatch'].items()
         if key != 'config']
        args, _ = parser.parse_known_args(config_arg_list)

    return args


def run(name, config):
    """
    Run the crossmatch test from EVE.
    """
    config_arg_list = []
    [config_arg_list.extend(['--'+key, str(val)]) for key, val in config.items()]

    args = parse_args(config_arg_list)

    details = main(args)

    el = elements.vtable('Results', list(details.keys()), details)

    return elements.page(name, utils.format_doc(__doc__), el)


def populate_metadata():
    metadata = {'Type': 'ValSummary',
                'Title': 'Validation',
                'TableTitle': 'Multivariate',
                'Headers': ['Null hypothesis', 'Critical value', 'Test Statistic']}
    return metadata


def summarize_result(result_page):
    summary = {}
    for el in result_page['Data']['Elements']:
        if el['Title'] == 'Results':
            summary['Null hypothesis'] = el['Data']['h0']
            summary['Critical value'] = el['Data']['critical']
            summary['Test Statistic'] = el['Data']['T']
    return {'': summary}


def print_summary(summary):
    print('    Cross Match Test:')
    print('      Null hypothesis: {}'.format(summary['']['Null hypothesis'].capitalize()))
    print('      Critical value: {}'.format(summary['']['Critical value']))
    print('      Test statistic: {}\n'.format(summary['']['Test Statistic']))


def print_details(details):
    print('  Cross Match Test:\n')
    print('    ens_set1 ensemble members:\n    {}'.format(details['set1']))
    print('    ens_set2 ensemble members:\n    {}\n'.format(details['set2']))

    print('    a1: {}'.format(details['a1']))
    print('    Ea1: {}'.format(details['Ea1']))
    print('    Va1: {}'.format(details['Va1']))
    print('    dev: {}'.format(details['dev']))
    print('    pval: {}'.format(details['T']))
    print('    approxpval: {}'.format(details['approxpval']))

    if details['h0'] == 'reject':
        print('\n    {} the null hypothesis; pval less than {}\n'.format(
                        details['h0'].capitalize(), details['critical']))
        print('    Ensembles statistically different!\n')
    else:
        print('\n    {} the null hypothesis; pval greater than or equal to {}\n'.format(
                        details['h0'].capitalize(), details['critical']))
        print('    Ensembles statistically equivalent!\n')
       

def main(args):
    crossmatch_pyr = robjects.r("""
               crossmatch_r <- function(indir1, indir2, casename1, casename2, season, ens_set1, ens_set2){{
                   source("{}/crossmatch_mahalanobis_ens.R")
                   x = crossmatch_mahalanobis(indir1, indir2, casename1, casename2, season, as.vector(ens_set1), as.vector(ens_set2))
                   return(x)
               }}
               """.format(args.R_dir))

    ens_set1 = numpy.genfromtxt(args.set1_file, dtype=int)
    ens_set2 = numpy.genfromtxt(args.set2_file, dtype=int)

    x = crossmatch_pyr(args.dir1, args.dir2, args.case1, args.case2, args.season, ens_set1, ens_set2)
    x = numpy.array(x)
    
    # NOTE: values defined here: https://cran.r-project.org/web/packages/crossmatch/crossmatch.pdf
    details = OrderedDict([('T', x[4][0]),
                           ('critical', args.critical),
                           ])

    if (details['T'] < args.critical):
        details['h0'] = 'reject'
    else:
        details['h0'] = 'accept'

    details['approxpval'] = x[5][0]
    details['set1'] = ens_set1
    details['set2'] = ens_set2
    details['a1']   = int(x[0][0])
    details['Ea1']  = x[1][0]
    details['Va1']  = x[2][0]
    details['dev']  = x[3][0]
    details['set1'] = ens_set1
    details['set2'] = ens_set2

    return details


if __name__ == '__main__':
    print_details(main(parse_args()))
