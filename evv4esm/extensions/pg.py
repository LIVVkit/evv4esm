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

"""The Perturbation Growth Test:
This tests the null hypothesis that the reference (n) and modified (m) model
ensembles represent the same atmospheric state after each physics parameterization
is applied within a single time-step using the two-sample (n and m) T-test for equal
averages at a 95% confidence level. Ensembles are generated by repeating the
simulation for many initial conditions, with each initial condition subject to
multiple perturbations.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import math
import argparse
# import logging

from pprint import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import stats
from netCDF4 import Dataset

import livvkit
from livvkit.util import elements as el
from livvkit.util import functions as fn

from evv4esm.utils import bib2html

# logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config',
                        type=fn.read_json,
                        default='test/pge_pc0101123.json',
                        help='A JSON config file containing a `pg` dictionary defining ' +
                             'the options.')

    args = parser.parse_args(args)
    name = args.config.keys()[0]
    config = args.config[name]

    return name, config


def _instance2sub(instance_number, total_perturbations):
    """
    Converts an instance number (ii) to initial condition index (ci) and
    perturbation index (pi)  subscripts

    instances use 1-based indexes and vary according to this function:
        ii = ci * len(PERTURBATIONS) + pi + 1
    where both pi and ci use 0-based indexes.
    """
    perturbation_index = (instance_number - 1) % total_perturbations
    initial_condition = (instance_number - 1 - perturbation_index) // total_perturbations
    return initial_condition, perturbation_index


def _sub2instance(initial_condition, perturbation_index, total_perturbations):
    """
    Converts initial condition index (ci) and perturbation index (pi) subscripts
    to an instance number (ii)

    instances use 1-based indexes and vary according to this function:
        ii = ci * len(PERTURBATIONS) + pi + 1
    where both pi and ci use 0-based indexes.
    """
    instance = initial_condition * total_perturbations + perturbation_index + 1
    return instance


def rmse_writer(file_name, rmse, perturbation_names, perturbation_variables, init_file_template):
    """
    Opens and writes a netcdf file for PGE curves
    This function is here purely to avoid duplicate
    codes so that it is easy to maintain code longterm
    """

    with Dataset(file_name, 'w') as nc:
        ninit, nprt_m1, nvars = rmse.shape

        nc.createDimension('ninit', ninit)
        nc.createDimension('nprt', nprt_m1 + 1)
        nc.createDimension('nprt_m1', nprt_m1)
        nc.createDimension('nvars', nvars)

        nc_init_cond = nc.createVariable('init_cond_files', str, 'ninit')
        nc_perturbation = nc.createVariable('perturbation_names', str, 'nprt')
        nc_variables = nc.createVariable('perturbation_variables', str, 'nvars')
        nc_rmse = nc.createVariable('rmse', 'f8', ('ninit', 'nprt_m1', 'nvars'))

        # NOTE: Assignment to netcdf4 variable length string array can be done
        #       via numpy arrays, or in a for loop using integer indices
        nc_perturbation[:] = np.array(perturbation_names)
        nc_variables[:] = np.array(perturbation_variables)
        nc_rmse[:] = rmse[:]

        for icond in range(0, ninit):
            # NOTE: Zero vs One based indexing
            nc_init_cond[icond] = init_file_template.format('cam', 'i', icond+1)


def variables_rmse(ifile_test, ifile_cntl, var_list, var_pefix=''):
    """
    Compute RMSE difference between perturbation and control for a set of
    variables

    Args:
         ifile_test: Path to a NetCDF dataset for a perturbed simulation
         ifile_cntl: Path to a NetCDF dataset for the control simulation
         var_list (list): List of all variables to analyze
         var_pefix: Optional prefix (e.g., t_, qv_) to apply to the variable

    returns:
        rmse (pandas.DataFrame): A dataframe containing the RMSE and maximum
            difference details between the perturbed and control simulation

    """

    # ------------------ARGS-------------------------
    # ifile_test: path of test file
    # ifile_cntl: path of control file
    # var_list  : List of all variables
    # var_pefix: Prefix for var_list (e.g. t_, t_ qv_ etc.)
    # -----------------------------------------------

    with Dataset(ifile_test) as ftest, Dataset(ifile_cntl) as fcntl:
        lat = ftest.variables['lat']
        lon = ftest.variables['lon']

        rmse = pd.DataFrame(columns=('RMSE', 'max diff', 'i', 'j', 'control', 'test', 'lat', 'lon'), index=var_list)

        # reshape for RMSE
        dims = len(ftest.variables[var_pefix + var_list[0]].dimensions)
        if dims == 3:  # see if it is SE grid
            nx, ny = ftest.variables[var_pefix + var_list[0]][0, ...].shape
            nz = 1
        else:
            nx, ny, nz = ftest.variables[var_pefix + var_list[0]][0, ...].shape

        for ivar, vvar in enumerate(var_list):
            var = var_pefix + vvar
            if var in ftest.variables:
                vtest = ftest.variables[var.strip()][0, ...]  # first dimension is time (=0)
                vcntl = fcntl.variables[var.strip()][0, ...]  # first dimension is time (=0)

                vrmse = math.sqrt(((vtest - vcntl)**2).mean()) / np.mean(vcntl)

                diff = abs(vtest[...] - vcntl[...])
                ind_max = np.unravel_index(diff.argmax(), diff.shape)

                rmse.loc[vvar] = (vrmse, diff[ind_max], ind_max[0], ind_max[1],
                                  vcntl[ind_max], vtest[ind_max],
                                  lat[ind_max[1]], lon[ind_max[1]])
    return rmse


def _print_details(details):
    for set_ in details:
        print('-' * 80)
        print(set_)
        print('-' * 80)
        pprint(details[set_])


def main(args):
    
    nvar = len(args.variables)
    nprt = len(args.perturbations)

    # for test cases (new environment etc.)
    # logger.debug("PGN_INFO: Test case comparison...")

    cond_rmse = {}
    for icond in range(args.ninit):
        prt_rmse = {}
        for iprt, prt_name in enumerate(args.perturbations):
            if prt_name == 'woprt':
                continue
            iinst_ctrl = _sub2instance(icond, 0, nprt)
            ifile_ctrl = os.path.join(args.ref_dir,
                                      args.instance_file_template.format('', iinst_ctrl, '_woprt'))
            # logger.debug("PGN_INFO:CNTL_TST:" + ifile_cntl)

            iinst_test = _sub2instance(icond, iprt, nprt)
            ifile_test = os.path.join(args.test_dir,
                                      args.instance_file_template.format(
                                              args.test_case + '.', iinst_test, '_' + prt_name))
            # logger.debug("PGN_INFO:TEST_TST:" + ifile_test)

            prt_rmse[prt_name] = variables_rmse(ifile_test, ifile_ctrl, args.variables, 't_')

        cond_rmse[icond] = pd.concat(prt_rmse)

    rmse = pd.concat(cond_rmse)
    comp_rmse = np.reshape(rmse.RMSE.values, (args.ninit, nprt-1, nvar))

    rmse_writer(os.path.join(args.test_dir, 'comp_cld.nc'),
                comp_rmse, args.perturbations.keys(), args.variables, args.init_file_template)

    details = OrderedDict()
    with Dataset(os.path.join(args.ref_dir, args.pge_cld)) as ref_cld:
        ref_dims = ref_cld.variables['cld_rmse'].shape
        cmp_dims = (args.ninit, nprt - 1, nvar)
        try:
            assert(ref_dims == cmp_dims)
        except AssertionError as e:
            be = BaseException(
                    'PGE curve dimensions (ninit, nptr, nvar) should be the same:\n'
                    '    CLD:{}  COMP:{}'.format(ref_dims, cmp_dims))
            six.raise_from(be, e)

        ref_rmse = ref_cld.variables['cld_rmse'][...]
        details['ref. data'] = ref_rmse

    pge_ends_cld = ref_rmse[:, :, -1]
    pge_ends_comp = comp_rmse[:, :, -1]

    # run the t-test
    pge_ends_cld = pge_ends_cld.flatten()
    pge_ends_comp = pge_ends_comp.flatten()

    t_stat, p_val = stats.ttest_ind(pge_ends_cld, pge_ends_comp)

    if np.isnan((t_stat, p_val)).any() or np.isinf((t_stat, p_val)).any():
        details['T test (t, p)'] = (None, None)
    else:
        details['T test (t, p)'] = '({:.3f}, {:.3f})'.format(t_stat, p_val)

    # logger.warn(" T value:" + str(t_stat))
    # logger.warn(" P value:" + str(p_val))
    crit = 0.05
    if t_stat is None:
        details['h0'] = '-'
    elif p_val < crit:
        details['h0'] = 'reject'
    else:
        details['h0'] = 'accept'

    # logger.debug("PGN_INFO: POST PROCESSING PHASE ENDS")

    details['test data'] = rmse

    ref_max_y = ref_rmse.max(axis=(0, 1)).astype(np.double)
    ref_min_y = ref_rmse.min(axis=(0, 1)).astype(np.double)

    cmp_max_y = comp_rmse.max(axis=(0, 1)).astype(np.double)
    cmp_min_y = comp_rmse.min(axis=(0, 1)).astype(np.double)

    img_file = os.path.relpath(os.path.join(args.img_dir, 'plot_comp.png'), os.getcwd())
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharey='all', gridspec_kw={'width_ratios': [3, 1]})
    plt.rc('font', family='serif')

    ax1.semilogy(ref_max_y, color='C0')
    ax1.semilogy(ref_min_y, color='C0')
    ax1.fill_between(range(ref_dims[-1]), ref_min_y, ref_max_y, color='C0', alpha=0.5)

    ax1.semilogy(cmp_max_y, color='C1')
    ax1.semilogy(cmp_min_y, color='C1')
    ax1.fill_between(range(cmp_dims[-1]), cmp_min_y, cmp_max_y, color='C1', alpha=0.5)

    ax1.set_xticks(range(len(args.variables)))
    ax1.set_xticklabels(args.variables, rotation=45, ha='right')
    ax1.set_ylabel('Temperature RMSE (K)')

    patch_list = [mpatches.Patch(color='C0', alpha=0.5, label='Ref.'),
                  mpatches.Patch(color='C1', alpha=0.5, label='Test')]
    ax1.legend(handles=patch_list, loc='upper left')

    scale_std = 1/np.sqrt(len(pge_ends_comp))
    tval_crit = stats.t.ppf(1 - crit, df=len(pge_ends_comp) - 1)
    ax2.errorbar(1, pge_ends_cld.mean(), xerr=np.stack([[0.1, 0.1]]).T,
                 fmt='none', ecolor='C0')
    # Note: Because these are so close to zero, but are best plotted on a
    #        semilogy plot, the mean ± 2*σ/√N range or the mean ± Tc*σ/√N, where
    #        Tc is the critical t test value, can cross zero.
    ax2.errorbar(1, pge_ends_comp.mean(), yerr=pge_ends_comp.std() * tval_crit * scale_std,
                 fmt='oC1', elinewidth=20, ecolor='C1', alpha=0.5)
    # ax2.errorbar(0.5, pge_ends_comp.mean(), yerr=pge_ends_comp.std() * 2 * scale_std,
    #              fmt='k.', elinewidth=20, ecolor='C1', alpha=0.5)

    ax2.set_xlim([0.8, 1.2])
    ax2.set_xticks([1])
    ax2.set_xticklabels([args.variables[-1]], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(img_file, bbox_inches='tight')
    plt.close(fig)

    img_desc = 'Left: The evolution of the maximum temperature (K) RMSE over a ' \
               'single time step for the {test} simulation (orange) and the {ref} ' \
               'simulation (blue), plotted with a log scale on the y-axis. ' \
               'The x-axis details the physical parameterizations ' \
               'and/or Fortran code modules executed within this time step. ' \
               'Right: the blue line indicates the {ref} ensemble mean at the ' \
               'end of the time step and the orange circle is the {test} ensemble mean. ' \
               'The orange box highlights the threshold values corresponding to the ' \
               'critical P {crit}% in the two-sided t-test. For the test to pass, ' \
               'the orange box must overlap the blue line. Note: The orange box may appear ' \
               'exceptionally large as  these  values  are very close to zero and ' \
               'the mean ± Tc*σ/√N range may cross zero, where Tc is the  critical ' \
               't-test value, σ is the ensemble standard deviation, N is the size ' \
               'of the ensemble, and σ/√N represents the t-test scaling ' \
               'parameter.'.format(test=args.test_name, ref=args.ref_name, crit=crit * 100)
    img_link = os.path.join(os.path.basename(args.img_dir), os.path.basename(img_file))
    img_gallery = el.gallery('', [
        el.image(args.test_case, img_desc, img_link, height=600)
    ])
    return details, img_gallery


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

    tbl_el = {'Type': 'Table',
              'Title': 'Results',
              'Headers': ['Test status', 'Null hypothesis', 'T test (t, p)', 'Ensembles'],
              'Data': {'Null hypothesis': details['h0'],
                       'T test (t, p)': details['T test (t, p)'],
                       'Test status': 'pass' if details['h0'] == 'accept' else 'fail',
                       'Ensembles': 'statistically identical' if details['h0'] == 'accept' else 'statistically different'}
              }

    if print_details:
        _print_details(details)

    bib_html = bib2html(os.path.join(os.path.dirname(__file__), 'pg.bib'))
    tab_list = [el.tab('Figures', element_list=[img_gal]),
                el.tab('References', element_list=[el.html(bib_html)])]
    page = el.page(name, __doc__.replace('\n\n', '<br><br>'), element_list=[tbl_el], tab_list=tab_list)

    return page


def print_summary(summary):
    print('    Perturbation growth test: {}'.format(summary['']['Case']))
    print('      Null hypothesis: {}'.format(summary['']['Null hypothesis']))
    print('      T Test (t, p): {}'.format(summary['']['T test (t, p)']))
    print('      Ensembles: {}\n'.format(summary['']['Ensembles']))


def summarize_result(results_page):
    summary = {'Case': results_page['Title']}
    for elem in results_page['Data']['Elements']:
        if elem['Type'] == 'Table' and elem['Title'] == 'Results':
            summary['Test status'] = 'pass' if elem['Data']['Null hypothesis'] == 'accept' else 'fail'
            summary['Null hypothesis'] = elem['Data']['Null hypothesis']
            summary['T test (t, p)'] = elem['Data']['T test (t, p)']
            summary['Ensembles'] = 'statistically identical' if elem['Data']['Null hypothesis'] == 'accept' else 'statistically different'
            break
        else:
            continue
    return {'': summary}


def populate_metadata():
    """
    Generates the metadata needed for the output summary page
    """
    metadata = {'Type': 'ValSummary',
                'Title': 'Validation',
                'TableTitle': 'Perturbation growth test',
                'Headers': ['Test status', 'Null hypothesis', 'T test (t, p)', 'Ensembles']
                }
    return metadata


if __name__ == '__main__':
    test_name, test_config = parse_args()
    run(test_name, test_config, print_details=True)
