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

import os
import sys
import time
import argparse

import evv4esm
import livvkit
from livvkit.util import options


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--extensions',
                        action='store',
                        nargs='+',
                        default=None,
                        help='Specify the location of the JSON configuration files for the extended V&V tests to run.')
  
    parser.add_argument('-o', '--out-dir',
                        default=os.path.join(os.getcwd(), "vv_" + time.strftime("%Y-%m-%d")),
                        help='Location to output the EVV webpages.')

    parser.add_argument('-s', '--serve',
                        nargs='?', type=int, const=8000,
                        help=' '.join(['Start a simple HTTP server for the output website specified',
                                       'by OUT_DIR on port SERVE.'
                                       ])
                        )

    parser.add_argument('-p', '--pool-size',
                        nargs='?',
                        type=int,
                        default=(options.mp.cpu_count() - 1 or 1),
                        help='The number of multiprocessing processes to run '
                             'analyses in. If zero, processes will run serially '
                             'outside of the multiprocessing module.')

    parser.add_argument('--version',
                        action='version',
                        version='EVV {}'.format(evv4esm.__version__),
                        help="Show EVV's version number and exit"
                        )

    args = parser.parse_args(args)

    if args.extensions:
        options.parse_args(['-V']+args.extensions + ['-o', args.out_dir])
    
    from evv4esm import resources
    args.livv_resource_dir = livvkit.resource_dir
    livvkit.resource_dir = os.sep.join(resources.__path__)
    return args


def main(cl_args=None):
    """ Direct execution. """

    if cl_args is None and len(sys.argv) > 1:
        cl_args = sys.argv[1:]
    args = parse_args(cl_args)

    print("--------------------------------------------------------------------")
    print("                   ______  __      __ __      __                    ")
    print("                  |  ____| \ \    / / \ \    / /                    ")
    print("                  | |__     \ \  / /   \ \  / /                     ")
    print("                  |  __|     \ \/ /     \ \/ /                      ")
    print("                  | |____     \  /       \  /                       ")
    print("                  |______|     \/         \/                        ")
    print("                                                                    ")
    print("    Extended Verification and Validation for Earth System Models    ")
    print("--------------------------------------------------------------------")
    print("")
    print("  Current run: " + livvkit.timestamp)
    print("  User: " + livvkit.user)
    print("  OS Type: " + livvkit.os_type)
    print("  Machine: " + livvkit.machine)
    print("  " + livvkit.comment)

    from livvkit.components import validation
    from livvkit import scheduler
    from livvkit.util import functions
    from livvkit import elements

    livvkit.pool_size = args.pool_size
    if args.extensions:
        functions.setup_output()
        summary_elements = []
        validation_config = {}
        print(" -----------------------------------------------------------------")
        print("   Beginning extensions test suite ")
        print(" -----------------------------------------------------------------")
        print("")
        for conf in livvkit.validation_model_configs:
            validation_config = functions.merge_dicts(validation_config,
                                                      functions.read_json(conf))
            summary_elements.extend(scheduler.run_quiet("validation", validation, validation_config,
                                                        group=False))
        print(" -----------------------------------------------------------------")
        print("   Extensions test suite complete ")
        print(" -----------------------------------------------------------------")
        print("")

        result = elements.Page("Summary", "", elements=summary_elements)
        with open(os.path.join(livvkit.output_dir, "index.json"), "w") as index_data:
            index_data.write(result._repr_json())
        print("-------------------------------------------------------------------")
        print(" Done!  Results can be seen in a web browser at:")
        print("   " + os.path.join(livvkit.output_dir, 'index.html'))
        print("-------------------------------------------------------------------")

    if args.serve:
        import http.server as server
        import socketserver as socket

        httpd = socket.TCPServer(('', args.serve), server.SimpleHTTPRequestHandler)

        sa = httpd.socket.getsockname()
        print('\nServing HTTP on {host} port {port} (http://{host}:{port}/)'.format(host=sa[0], port=sa[1]))
        print('\nView the generated website by navigating to:')
        print('\n    http://{host}:{port}/{path}/index.html'.format(host=sa[0], port=sa[1],
                                                                    path=os.path.relpath(args.out_dir)
                                                                    ))
        print('\nExit by pressing `ctrl+c` to send a keyboard interrupt.\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nKeyboard interrupt received, exiting.\n')
            sys.exit(0)


if __name__ == '__main__':
    main()
