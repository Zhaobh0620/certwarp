"""The EB configuration record names the run's noise order (work package FUP1; review MIN1_复核_20261031 question 5 and
minor 1).  What is tested here:
- LaplaceConfig.record(logvol_points=...) / noise_order_of_points and EBConfig.record(logvol_points): the C14-7 order
  with logvol_closed27 for "gauss27+closed27"; without the argument, and with "gauss27", the record is byte for byte
  the one before FUP1 (the EB goldens golden_7579645_eb*.json call it without an argument and still pass:
  learn/test_eb_count.py, learn/test_cli_switches.py);
- run_population_eb(logvol_points=...): its result record's config (what eb_result.json carries) follows the argument,
  on the synthetic driver of _eb_digest;
- the learn CLI, dry run only: config.json eb.laplace.noise_order follows --logvol-points (default gauss27+closed27:
  the C14-7 order; gauss27: NOISE_ORDER, the whole eb record equal to EBConfig.record()).
Not tested end to end (FUP1 review minor 3): that a non-dry CLI run's eb_result.json follows --logvol-points; the CLI
passes args.logvol_points to run_population_eb in one line (bayesreg/learn/cli.py), checked by reading.  Records only;
no value changes; no image is solved.
"""
import json

import pytest

from jd_curl.validation import InputValidationError

from bayesreg.infer import laplace as lp
from bayesreg.learn import EBConfig, cli, run_population_eb
from bayesreg.learn import population as pop
from bayesreg.seg import structure_layer
from bayesreg.solve import Eta

from . import _eb_digest as E
from . import _guard_fixtures as gf
from .test_cli import argv, eta_file

CLOSED = "gauss27+closed27"


def _dumps(record):
    return json.dumps(record, sort_keys=False)


def test_laplace_config_record_by_point_set_name():
    config = lp.LaplaceConfig()
    assert _dumps(config.record(logvol_points=None)) == _dumps(config.record())
    assert _dumps(config.record(logvol_points="gauss27")) == _dumps(config.record())
    assert config.record(logvol_points=CLOSED)["noise_order"] == list(lp.NOISE_ORDER_CLOSED27)
    assert lp.noise_order_of_points(CLOSED, prior=True) == list(lp.PRIOR_NOISE_ORDER_CLOSED27)
    assert lp.noise_order_of_points("gauss27", prior=True) == list(lp.PRIOR_NOISE_ORDER)
    with pytest.raises(InputValidationError):
        lp.noise_order_of_points("gauss27+corners")
    with pytest.raises(InputValidationError):
        config.record(object(), logvol_points="gauss27")


@pytest.mark.parametrize("eb_count", ["dim", "gull_mackay"])
def test_eb_config_record(eb_count):
    config = EBConfig(n_samples=2, eb_count=eb_count)
    before = config.record()
    assert before["laplace"]["noise_order"] == list(lp.NOISE_ORDER)
    assert _dumps(config.record(logvol_points="gauss27")) == _dumps(before)
    closed = config.record(logvol_points=CLOSED)
    assert closed["laplace"]["noise_order"] == list(lp.NOISE_ORDER_CLOSED27)
    closed["laplace"]["noise_order"] = before["laplace"]["noise_order"]
    assert _dumps(closed) == _dumps(before)                     # only the noise order differs


def test_run_population_eb_record_follows_the_point_set():
    """The synthetic driver of the EB golden (_eb_digest.driver_case, first case): the result record's config without
    logvol_points and with "gauss27" is the same JSON; with "gauss27+closed27" only config.laplace.noise_order
    differs."""
    eta0 = Eta(pop.alpha_nugget(E.TAU2), 300., E.TAU2)
    target = (lambda nu: 100+.5*(nu-100))
    records = {}
    for points in (None, "gauss27", CLOSED):
        kwargs = {} if points is None else dict(logvol_points=points)
        result = run_population_eb(E._pairs(), E._solve_ready, eta0=eta0, config=EBConfig(**E.DIM), run_id="digest",
                                   statistics=E._statistics(target), **kwargs)
        records[points] = E.canonical(result.record)          # exact floats, wall-clock keys dropped
    assert _dumps(records["gauss27"]) == _dumps(records[None])
    assert records[None]["config"]["laplace"]["noise_order"] == list(lp.NOISE_ORDER)
    closed = records[CLOSED]
    assert closed["config"]["laplace"]["noise_order"] == list(lp.NOISE_ORDER_CLOSED27)
    closed["config"]["laplace"]["noise_order"] = list(lp.NOISE_ORDER)
    assert _dumps(closed) == _dumps(records[None])


def test_cli_config_json_noise_order(tmp_path, monkeypatch):
    """learn CLI dry run: config.json eb.laplace.noise_order contains logvol_closed27 under the default point set
    (gauss27+closed27) and is NOISE_ORDER under --logvol-points gauss27, where the whole eb record is
    EBConfig.record() without an argument (the record before FUP1)."""
    paths = gf.write_files(tmp_path/"split")
    monkeypatch.setattr(structure_layer, "REPO_REGISTRY", paths["registry"])
    out = tmp_path/"eb_closed27"
    assert cli.main(argv(paths, out, "--dry-run", eta0=eta_file(tmp_path))) == 0
    config = json.loads((out/"config.json").read_text())
    assert config["args"]["logvol_points"] == CLOSED
    order = config["eb"]["laplace"]["noise_order"]
    assert order == list(lp.NOISE_ORDER_CLOSED27) and any("logvol_closed27" in item for item in order)
    out = tmp_path/"eb_gauss27"
    assert cli.main(argv(paths, out, "--dry-run", "--logvol-points", "gauss27", eta0=eta_file(tmp_path))) == 0
    config = json.loads((out/"config.json").read_text())
    assert config["eb"]["laplace"]["noise_order"] == list(lp.NOISE_ORDER)
    expected = cli.eb_config(cli.parse_args(argv(paths, out, "--dry-run", "--logvol-points", "gauss27",
                                                  eta0=eta_file(tmp_path)))).record()
    assert config["eb"] == json.loads(json.dumps(expected))
