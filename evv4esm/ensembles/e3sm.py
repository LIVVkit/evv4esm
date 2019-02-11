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

"""E3SM specific ensemble functions."""

import os
import re
import glob

from collections import OrderedDict


def component_file_instance(component, case_file):
    search_regex = r'{}_[0-9]+'.format(component)
    result = re.search(search_regex, case_file).group(0)
    return int(result.replace('cam_', ''))


def file_date_str(case_file):
    search_regex = r'h0\.[0-9]+-[0-9]+.nc'
    result = re.search(search_regex, case_file).group(0)
    return result.replace('h0.', '').replace('.nc', '')


def component_monthly_files(dir_, component, ninst):
    base = '{d}/*cam_????.h0.????-??.nc'.format(d=dir_)
    search = os.path.normpath(base)
    result = sorted(glob.glob(search))

    instance_files = OrderedDict()
    for ii in range(1, ninst + 1):
        instance_files[ii] = sorted(filter(lambda x: component_file_instance(component, x) == ii, result),
                                    key=file_date_str)
        if len(instance_files[ii]) > 12:
            instance_files[ii] = instance_files[ii][-12:]

    return instance_files
