import json
from collections import namedtuple
from pathlib import Path

from modelmimic import mimic as mmc

import evv4esm  # pylint: disable=import-error
from evv4esm.__main__ import main as evv  # pylint: disable=import-error

evv_lib_dir = Path(evv4esm.__file__).parent.resolve()
mimic_lib_dir = Path(mmc.__file__).parent.resolve()
cl_args = namedtuple("Args", ["cfg"])
evv_tests = {"TSC": "tsc.py", "MVK": "ks.py"}


def gen_data_run_evv(evv_test):
    _args = cl_args(cfg=Path(mimic_lib_dir, "config", f"{evv_test}.toml"))
    # Generate data for all the tests (should be a pass b4b / pass non-b4b / fail)
    out_dirs = mmc.main(_args)

    # Load the EVV4ESM configuration template
    with open(
        Path(evv_lib_dir.parent, "test", f"{evv_test}_template.json"),
        "r",
        encoding="utf-8",
    ) as _template:
        evv_cfg_template = json.loads(_template.read())

    evv_cfg = {}
    for result_check in ["pass_b4b", "pass_nb4b", "fail"]:
        _check_cfg = dict(evv_cfg_template[evv_test])
        _check_cfg["module"] = str(Path(evv_lib_dir, "extensions", evv_tests[evv_test]))
        _check_cfg["test-dir"] = str(out_dirs[result_check]["test"].resolve())
        _check_cfg["ref-dir"] = str(out_dirs[result_check]["baseline"].resolve())
        evv_cfg[f"{evv_test}_{result_check}"] = _check_cfg

    json_file = Path(f"{evv_test}_test_conf.json")
    with open(json_file, "w") as config_file:
        json.dump(evv_cfg, config_file, indent=4)

    evv_out_dir = Path(f"{evv_test}_test_output")
    evv(["-e", str(json_file), "-o", str(evv_out_dir)])

    with open(Path(evv_out_dir, "index.json")) as evv_f:
        evv_status = json.load(evv_f)

    status = {}
    for evv_ele in evv_status["Page"]["elements"]:
        if "Table" in evv_ele:
            _index = evv_ele["Table"]["index"][0]
            _status = evv_ele["Table"]["data"]["Test status"][0].lower() == "pass"
            status[_index] = _status
    for _index in status:
        if "pass" in _index:
            assert status[_index], f"{_index} SHOULD BE PASS IS FAIL"
        else:
            assert not status[_index], f"{_index} SHOULD BE FAIL IS PASS"


def test_evv_tsc():
    gen_data_run_evv("TSC")


def test_evv_mvk():
    gen_data_run_evv("MVK")
