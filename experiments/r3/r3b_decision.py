"""R3b's decision record, for R3a's composition and reading tools (task brief runs/20261111-r3, "两包的接口"; calibration_plan.md section 10 and revision 2 items 1, 2).

    from r3b_decision import load_calibration_decision, decode_kwargs
    decision = load_calibration_decision()                    # the real record under /home/rmcfz/seq_vp_tag_results/r3b_calib_v1/analysis/; not "pass": DecodeNotAdmitted
    assert decision["decode_py_sha256"] == <R3a's pinned sha256 of bayesreg/seg/decode.py>
    labels = bayesreg.seg.decode.decode(probs, **decode_kwargs(decision))

load_calibration_decision checks: the format and the verdict's name; that the verdict agrees with the checks (pass: all four hold; not_pass: one fails; undecidable:
reasons given); that the decoder is the pinned one (r3b_common.DECODE_SPEC: dice_decode_v1, expected_dice, floor 1e-3, the small six); on a real record the plan's
pinned sha256, the pinned criterion, and that it is not synthetic; and (review I1) that the file of the bayesreg.seg.decode module this process imported has the
sha256 the record binds (decode_py_sha256, the file the decode unit ran).  It returns that sha256 to the caller.  require_pass is True by default (review m3):
only a descriptive use passes False.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r3b_common as rc  # noqa: E402

DEFAULT_PATH = rc.OUT/rc.ANALYSIS_DIR/rc.DECISION_FILE


class DecodeNotAdmitted(Exception):
    """The calibration reading did not pass: no candidate with the decoding."""


def _consistent(document):
    """None when the verdict agrees with the checks and reasons, else why not."""
    verdict, checks, reasons = document.get("verdict"), document.get("checks"), document.get("reasons") or []
    if verdict == "undecidable":
        return None if reasons and checks is None else "undecidable without reasons, or with checks"
    if not isinstance(checks, dict) or set(checks) != {"small6_mean", "small6_positive", "seg35_mean", "seg4_mean"} or reasons:
        return "a verdict without its four checks, or with reasons"
    oks = [bool(c.get("ok")) for c in checks.values()]
    if verdict == "pass" and not all(oks):
        return "pass with a check that does not hold"
    if verdict == "not_pass" and all(oks):
        return "not_pass with every check holding"
    return None


def load_calibration_decision(path=DEFAULT_PATH, *, require_pass=True, expected_sha256=None, allow_synthetic=False, check_decode_py=True, decode_path=None):
    """The decision record as a dict, plus "decision_sha256" and "decode_py_sha256".  ValueError when it is not R3b's record of the pinned plan, decoder and
    criterion, or when the imported decode.py (decode_path, default the module's file) is not the file the record binds; DecodeNotAdmitted when require_pass and the
    verdict is not "pass"."""
    path = Path(path)
    sha = rc.sha256_file(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{path}: sha256 {sha} is not the expected {expected_sha256}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != rc.DECISION_FORMAT:
        raise ValueError(f"{path}: format {document.get('format')!r}, expected {rc.DECISION_FORMAT}")
    if document.get("verdict") not in rc.VERDICTS:
        raise ValueError(f"{path}: verdict {document.get('verdict')!r} not in {rc.VERDICTS}")
    problem = _consistent(document)
    if problem:
        raise ValueError(f"{path}: {problem}")
    if document.get("decoder_spec") != json.loads(json.dumps(rc.DECODE_SPEC)):
        raise ValueError(f"{path}: the decoder is not the pinned one (r3b_common.DECODE_SPEC)")
    if document.get("synthetic") and not allow_synthetic:
        raise ValueError(f"{path}: a synthetic record")
    if not document.get("synthetic"):
        if rc.PLAN_SHA256 is None or document.get("plan_sha256") != rc.PLAN_SHA256:
            raise ValueError(f"{path}: plan sha256 {document.get('plan_sha256')} is not the pinned {rc.PLAN_SHA256}")
        if document.get("criterion") != json.loads(json.dumps(rc.CRITERION)):
            raise ValueError(f"{path}: the criterion is not the pinned one (r3b_common.CRITERION)")
    bound = document.get("decode_py_sha256")
    if not bound:
        raise ValueError(f"{path}: the record binds no decode.py sha256")
    if check_decode_py:
        here = rc.sha256_file(Path(decode_path) if decode_path is not None else rc.decode_py_path())
        if here != bound:
            raise ValueError(f"{path}: bayesreg/seg/decode.py here has sha256 {here}, the record binds {bound} (the decoder the reading judged)")
    if require_pass and document["verdict"] != "pass":
        raise DecodeNotAdmitted(f"{path}: verdict {document['verdict']}: the decoding does not enter the composition")
    return dict(document, decision_sha256=sha, decode_py_sha256=bound)


def decode_kwargs(decision):
    """The keyword arguments of bayesreg.seg.decode.decode for the recorded (pinned) decoder."""
    spec = decision["decoder_spec"]
    return dict(labels=tuple(spec["labels"]), objective=spec["objective"], candidate_floor=spec["candidate_floor"])
