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


__version_info__ = (0, 3, 2)
__version__ = '.'.join(str(vi) for vi in __version_info__)

PASS_COLOR = '#389933'
L_PASS_COLOR = '#93DA90'

FAIL_COLOR = '#BF3F46'
L_FAIL_COLOR = '#E68388'

human_color_names = {'pass': ('green', 'light green'),
                     PASS_COLOR: 'green',
                     L_PASS_COLOR: 'light green',
                     'fail': ('red', 'light red'),
                     FAIL_COLOR: 'red',
                     L_FAIL_COLOR: 'light red'}

pf_color_picker = {'Pass': PASS_COLOR, 'pass': PASS_COLOR,
                   'Accept': PASS_COLOR, 'accept': PASS_COLOR,
                   'Fail': FAIL_COLOR, 'fail': FAIL_COLOR,
                   'Reject': FAIL_COLOR, 'reject': FAIL_COLOR}

light_pf_color_picker = {'Pass': L_PASS_COLOR, 'pass': L_PASS_COLOR,
                         'Accept': L_PASS_COLOR, 'accept': L_PASS_COLOR,
                         PASS_COLOR: L_PASS_COLOR,
                         'Fail': L_FAIL_COLOR, 'fail': L_FAIL_COLOR,
                         'Reject': L_FAIL_COLOR, 'reject': L_FAIL_COLOR,
                         FAIL_COLOR: L_FAIL_COLOR}


class EVVException(Exception):
    """Base class for EVV exceptions"""
    pass
