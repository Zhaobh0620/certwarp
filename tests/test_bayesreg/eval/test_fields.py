"""bayesreg.eval.fields: disp.npz / final_native.npz readers, conversions and the C16 round trip."""
import json

import numpy as np
import pytest

from bayesreg.eval import fields as F

from . import _synthetic as syn

PAIR = f"{syn.DEV[0]}__{syn.DEV[1]}"


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def test_conversions_bitwise_and_equal_to_the_l2r_script():
    script = syn.l2r_script()
    u = syn.d_to_u(syn.smooth_d())
    for spacing in ((1., 1., 1.), (1.2, .9, 1.1)):
        d = F.u_to_d(u, spacing)
        assert np.array_equal(bits(d), bits(script.disp_voxels(u, spacing)))
    d = F.u_to_d(u, (1., 1., 1.))
    assert np.array_equal(bits(F.d_to_u(d, (1., 1., 1.))), bits(u))
    record = F.roundtrip_check(u, d, (1., 1., 1.))
    assert record["passed"] and record["unit_spacing"] and record["u_equals_d_to_u_bitwise"]
    aniso = (1.2, .9, 1.1)
    record = F.roundtrip_check(u, F.u_to_d(u, aniso), aniso)
    assert record["passed"] and not record["unit_spacing"] and record["u_within_1ulp"]
    wrong = F.u_to_d(u, (1., 1., 1.))
    wrong[0, 5, 5, 5] += 1e-9
    assert not F.roundtrip_check(u, wrong, (1., 1., 1.))["passed"]


def test_bayesreg_run_h1_round_trip(tmp_path):
    u = syn.d_to_u(syn.smooth_d())
    run, grid = syn.write_bayesreg_run(tmp_path/"run", PAIR, u)
    info = F.describe_run(run)
    assert (info.kind, info.pair, info.stage, info.arm_hint, info.run_status) == (
        "bayesreg", PAIR, syn.STAGE, "tissue", "converged_B")
    assert info.field_role == F.ROLE_NATIVE and info.completion["exit_code"] == 0
    assert info.completion["native_commits"] == 3
    assert F.field_present(info)
    d, u_run, record = F.load_field(info, syn.SHAPE, (1., 1., 1.))
    assert np.array_equal(bits(u_run), bits(u)) and np.array_equal(bits(d), bits(F.u_to_d(u, (1., 1., 1.))))
    assert record["roundtrip"]["passed"] and record["roundtrip"]["u_equals_d_to_u_bitwise"]
    assert record["final_native"]["state_hash_matches"] and record["disp"]["meta"]["units"] == "voxels"
    assert record["disp"]["d_sha256"] == F.array_sha256(d)


def test_bayesreg_run_anisotropic_within_one_ulp(tmp_path):
    spacing = (1.2, .9, 1.1)
    u = syn.d_to_u(syn.smooth_d(), spacing)
    run, _ = syn.write_bayesreg_run(tmp_path/"run", PAIR, u, spacing=spacing)
    _, _, record = F.load_field(F.describe_run(run), syn.SHAPE, spacing)
    assert record["roundtrip"]["passed"] and record["roundtrip"]["u_within_1ulp"]
    with pytest.raises(F.FieldError, match="grid differs"):
        F.load_field(F.describe_run(run), syn.SHAPE, (1., 1., 1.))


def test_bayesreg_run_states(tmp_path):
    u = syn.d_to_u(syn.smooth_d())
    run, _ = syn.write_bayesreg_run(tmp_path/"failed", PAIR, u, with_disp=False)
    info = F.describe_run(run)
    assert not F.field_present(info) and info.field_role == F.ROLE_NONE and info.completion["exit_code"] == 2
    run, _ = syn.write_bayesreg_run(tmp_path/"running", PAIR, u)
    (run/"status.json").unlink()
    with pytest.raises(F.FieldError, match="not finished"):
        F.describe_run(run)
    (run/"status.json").write_text(json.dumps(dict(status="converged_B", native=True)))   # no exit_code
    with pytest.raises(F.FieldError, match="exit_code"):
        F.describe_run(run)
    run, _ = syn.write_bayesreg_run(tmp_path/"handover", PAIR, u, native_commits=0)
    info = F.describe_run(run)
    assert info.field_role == F.ROLE_HANDOVER and F.field_present(info) and info.completion["native_commits"] == 0
    run, _ = syn.write_bayesreg_run(tmp_path/"nohyper", PAIR, u, native_commits=None)
    assert F.describe_run(run).field_role == F.ROLE_UNKNOWN
    run, grid = syn.write_bayesreg_run(tmp_path/"tampered", PAIR, u)
    with np.load(run/"final_native.npz") as z:
        members = {k: np.array(z[k]) for k in z.files}
    members["state_hash"] = np.array("0"*64)
    (run/"final_native.npz").unlink()
    np.savez(run/"final_native.npz", **members)
    with pytest.raises(F.FieldError, match="state_hash"):
        F.load_field(F.describe_run(run), syn.SHAPE, (1., 1., 1.))
    with pytest.raises(F.FieldError, match="differs from the run's own pair"):
        F.describe_run(tmp_path/"failed", pair="other__pair")


def test_bayesreg_structure_registry_label(tmp_path):
    """Review PS item 2: the registration state of a structure run's segmenter registry, from status.json (run_pair
    since PS1) or provenance.json; None for other channel sources."""
    u = syn.d_to_u(syn.smooth_d())
    label = dict(status="staging_unregistered", path="/x/segmenters.json", staging_opt_in=True,
                 development_diagnostic_only=True)
    run, _ = syn.write_bayesreg_run(tmp_path/"staging", PAIR, u, channel_source="structure",
                                    status_extra=dict(structure_registry=label))
    record = F.describe_run(run).extra["structure_registry"]
    assert record == dict(status=F.STAGING_UNREGISTERED, staging_unregistered=True, source="status.json:structure_registry",
                          path="/x/segmenters.json", git_tracked=None, git_clean=None)
    # review PS1_PS2 item b: the registry file's git state, as the run recorded it
    provenance = dict(channels=dict(structure=dict(registry=dict(status="registered", path="/r/segmenters.json",
                                                                 git=dict(tracked=True, clean=False)))))
    run, _ = syn.write_bayesreg_run(tmp_path/"older", PAIR, u, channel_source="structure", provenance=provenance)
    record = F.describe_run(run).extra["structure_registry"]
    assert record["status"] == "registered" and record["staging_unregistered"] is False
    assert record["git_tracked"] is True and record["git_clean"] is False
    assert record["source"] == "provenance.json:channels.structure.registry"
    run, _ = syn.write_bayesreg_run(tmp_path/"unrecorded", PAIR, u, channel_source="structure")
    assert F.describe_run(run).extra["structure_registry"]["status"] is None
    run, _ = syn.write_bayesreg_run(tmp_path/"tissue", PAIR, u, status_extra=dict(structure_registry=None))
    assert F.describe_run(run).extra["structure_registry"] is None


def test_configc_run(tmp_path):
    u = syn.d_to_u(syn.smooth_d())
    run = syn.write_configc_run(tmp_path/"run", PAIR, u, run_id="d0v1_A0_x")
    info = F.describe_run(run)
    assert (info.kind, info.pair, info.stage, info.run_status) == ("configC", PAIR, None, "native_budget_exhausted")
    assert (info.field_role, info.metrics_role, info.run_id_hint) == (
        F.ROLE_NATIVE, "formal_native_outer_commit", "d0v1_A0_x")
    assert info.run_certificate["status"] == "certified_feasible" and info.completion["exit_code"] == 0
    d, u_run, record = F.load_field(info, syn.SHAPE, (1., 1., 1.))
    assert np.array_equal(bits(d), bits(syn.l2r_script().disp_voxels(u, (1., 1., 1.))))
    assert record["roundtrip"]["passed"] and np.array_equal(bits(u_run), bits(u))
    with pytest.raises(F.FieldError, match="differs from the pair's"):
        F.load_field(info, syn.SHAPE, (1., 1., 2.))


def test_configc_completion_and_field_roles(tmp_path):
    """Review EV items 1 and 2: an unfinished configuration C run is refused; metrics_role sets the field role."""
    u = syn.d_to_u(syn.smooth_d())
    run = syn.write_configc_run(tmp_path/"running", PAIR, u, exit_code=None)
    with pytest.raises(F.FieldError, match="not finished"):
        F.describe_run(run)
    run = syn.write_configc_run(tmp_path/"incomplete", PAIR, u, complete=False)
    with pytest.raises(F.FieldError, match="records incomplete"):
        F.describe_run(run)
    run = syn.write_configc_run(tmp_path/"nosummary", PAIR, u)
    (run/"task"/"summary.json").unlink()
    with pytest.raises(F.FieldError, match="records incomplete"):
        F.describe_run(run)
    run = syn.write_configc_run(tmp_path/"garbled", PAIR, u)
    (run/"exit_code").write_text("x\n")
    with pytest.raises(F.FieldError, match="not an integer"):
        F.describe_run(run)
    info = F.describe_run(syn.write_configc_run(tmp_path/"timeout", PAIR, u, exit_code=124))
    assert (info.field_role, info.run_status) == (F.ROLE_NONE, "exit_124") and not F.field_present(info)
    info = F.describe_run(syn.write_configc_run(
        tmp_path/"handover", PAIR, u, metrics_role="certified_native_initializer_NOT_native_outer_commit"))
    assert info.field_role == F.ROLE_HANDOVER and F.field_present(info)
    info = F.describe_run(syn.write_configc_run(tmp_path/"none", PAIR, u, metrics_role="no_native_result"))
    assert info.field_role == F.ROLE_NONE and not F.field_present(info)
    info = F.describe_run(syn.write_configc_run(tmp_path/"odd", PAIR, u, metrics_role="something_else"))
    assert info.field_role == F.ROLE_UNKNOWN and F.field_present(info)
    assert F.ROLE_HANDOVER not in F.COMPARABLE_ROLES and F.ROLE_UNKNOWN not in F.COMPARABLE_ROLES


def test_disp_directory(tmp_path):
    d64 = syn.smooth_d()
    (tmp_path/"a").mkdir()
    np.savez(tmp_path/"a"/"disp.npz", d=d64.astype(np.float32), meta=np.array(json.dumps(dict(method="x"))))
    with pytest.raises(F.FieldError, match="--pair"):
        F.describe_run(tmp_path/"a")
    info = F.describe_run(tmp_path/"a", pair=PAIR)
    assert info.field_role == F.ROLE_DISP
    d, u_run, record = F.load_field(info, syn.SHAPE, (1., 1., 1.))
    assert u_run is None and d.dtype == np.float64 and record["disp"]["stored_dtype"] == "float32"
    assert np.array_equal(d, d64.astype(np.float32).astype(np.float64)) and record["disp"]["meta"] == dict(method="x")
    bad = {"shape": np.zeros((3,)+syn.SHAPE[::-1]), "rank": np.zeros(syn.SHAPE), "int": np.zeros((3,)+syn.SHAPE, int),
           "nan": np.full((3,)+syn.SHAPE, np.nan)}
    for name, value in bad.items():
        (tmp_path/name).mkdir()
        np.savez(tmp_path/name/"disp.npz", d=value)
        with pytest.raises(F.FieldError):
            F.load_field(F.describe_run(tmp_path/name, pair=PAIR), syn.SHAPE, (1., 1., 1.))
    (tmp_path/"nokey").mkdir()
    np.savez(tmp_path/"nokey"/"disp.npz", flow=d64)
    with pytest.raises(F.FieldError, match="'d' missing"):
        F.load_field(F.describe_run(tmp_path/"nokey", pair=PAIR), syn.SHAPE, (1., 1., 1.))
    (tmp_path/"empty").mkdir()
    with pytest.raises(F.FieldError, match="not a bayesreg"):
        F.describe_run(tmp_path/"empty")
