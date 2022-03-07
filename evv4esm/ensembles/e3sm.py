#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2018-2022 UT-BATTELLE, LLC
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

"""E3SM specific ensemble functions."""

import six

import os
import re
import glob

from collections import OrderedDict

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def component_file_instance(component, case_file):
    search_regex = r'{c}_[0-9]+'.format(c=component)
    result = re.search(search_regex, case_file).group(0)
    return int(result.replace('{}_'.format(component), ''))


def file_date_str(case_file, style='short'):
    if style == 'full':
        search_regex = r'h0\.[0-9]+-[0-9]+-[0-9]+-[0-9]+.nc'
    elif style == 'short':
        search_regex = r'h0\.[0-9]+-[0-9]+.nc'
    else:
        # FIXME: log warning here
        search_regex = r'h0\.[0-9]+-[0-9]+.nc'
    result = re.search(search_regex, case_file).group(0)
    return result.replace('h0.', '').replace('.nc', '')


def component_monthly_files(dir_, component, ninst):
    base = '{d}/*{c}_????.h0.????-??.nc'.format(d=dir_, c=component)
    search = os.path.normpath(base)
    result = sorted(glob.glob(search))

    instance_files = OrderedDict()
    for ii in range(1, ninst + 1):
        instance_files[ii] = sorted(filter(lambda x: component_file_instance(component, x) == ii, result),
                                    key=file_date_str)
        if len(instance_files[ii]) > 12:
            instance_files[ii] = instance_files[ii][-12:]

    return instance_files


def gather_monthly_averages(ensemble_files, variable_set=None):
    monthly_avgs = []
    for case, inst_dict in six.iteritems(ensemble_files):
        for inst, i_files in six.iteritems(inst_dict):
            # Get monthly averages from files
            for file_ in i_files:
                date_str = file_date_str(file_)

                data = None
                try:
                    data = Dataset(file_)
                    if variable_set is None:
                        variable_set = set(data.variables.keys())
                except OSError as E:
                    six.raise_from(BaseException('Could not open netCDF dataset: {}'.format(file_)), E)

                for var in data.variables.keys():
                    if var not in variable_set:
                        continue
                    if len(data.variables[var].shape) < 2 or var in ['time_bnds', 'date_written', 'time_written']:
                        continue
                    elif 'ncol' not in data.variables[var].dimensions:
                        continue
                    else:
                        m = np.mean(data.variables[var][0, ...])
                    try:
                        _name = f": {data.variables[var].getncattr('long_name')}"
                    except AttributeError:
                        _name = ""
                    try:
                        _units = f" [{data.variables[var].getncattr('units')}]"
                    except AttributeError:
                        _units = ""
                    desc = f"{_name}{_units}"
                    monthly_avgs.append((case, var, '{:04}'.format(inst), date_str, m, desc))

    monthly_avgs = pd.DataFrame(monthly_avgs, columns=('case', 'variable', 'instance', 'date', 'monthly_mean', 'desc'))
    return monthly_avgs
