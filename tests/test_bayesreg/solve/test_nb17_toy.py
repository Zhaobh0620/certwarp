"""C04-6 gate 4 (work package SWLI): the NB17 toy ECM with frozen support weights v (docs/bayesreg/checks/
NB17_ecm_monotone_toy_v.py; 02_理论 B4′, status unchanged): support and support_shared, Gaussian and Student-t, node
weights {1, 1/2, 1/4}, 40 rounds -- L^v never increases after the level start, dL/ds = 0 at every sigma step; the control
arm (v recomputed every round) is only recorded."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[3]/"docs"/"bayesreg"/"checks"/"NB17_ecm_monotone_toy_v.py"


def load():
    spec = importlib.util.spec_from_file_location("nb17_toy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nb17_frozen_v_is_monotone(capsys):
    toy = load()
    assert toy.main(rounds=40)
    for grouping in ("support", "support_shared"):
        for student in (False, True):
            out = toy.run(grouping, student, 40)
            series = out["series"]
            assert len(series) == (81 if not student else 41)
            assert np.all(np.diff(series) <= 1e-10*np.abs(series[:-1]))
            assert out["max_derivative"] <= 1e-12
    assert "PASS" in capsys.readouterr().out
