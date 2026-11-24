"""HS for P2/P3 (the lead's interface request, 11-06; runs/20261106-p2/plan.md §3, §3.1, experiments/p2/p2_core.load_reference, reference_pair_problems, hs_fallback_problems, decide_p2.reference_rows, run_p2.hs_problem on the branch p2-tools 2b7cceb; third_certificate_problems at 2488907).

1. The decision file: hs_v1/analysis/decision_hs.json (decide_hs.py) has `reading` ("采用新分割器" / "不采用" / "不可判"), `segmenter_id` and `per_pair` (`pair`, `line_status`, `metrics`); run_p2.hs_problem reads that one file.
2. The reference file: after HS reads 采用新分割器, `run_hs.py write-reference` writes experiments/p2/reference_<segmenter_id>.json in the schema bayesreg.p2.reference.v1 (run_p2.v1_reference's keys), from HS's own records:
   p2 the first 5 pairs, p3 the 19 pairs; each pair the repair directory, its tier, run id, attempt sha256 (repair.json), repaired field sha256 (repair_metrics.json), certified at alpha = 1 (the evaluation's certificate), the fuse,
   the T88 field directory with its file sha256 and u sha256 (run.json's checkpoint), the D factor; and, besides P2's keys, HS's line status (`hs_line_status`) and the sha256 of the three JSON records the pins come from.
   A fallback row (hx_fallback_configC, hx_fallback_identity; P2 plan revision 2) carries `method_metrics`: decision_hs.json's per_pair metrics of the pair, copied as they are, and the top-level `hs.decision_sha256` pins that file;
   its field_dir is null when the field run read no channel, its repair_dir and repair pins null when there is no repair.json.
   A row hx_certified by the third certificate (the evaluation's review certificate did not pass, C22.3-9 (d)) carries third_certificate_file and third_certificate_sha256: the fallback unit's record of the pair
   (hs_v1/fallback/<pair>/fallback.json, with third_certificate and reads_labels), P2 plan revision 3 §3.1 (revision 1d).
   channels: pred_root, models_root, image_root, registry_rel (p2_core.ChannelSet.source).
   Only JSON records and their sha256: no field, no image, no label is read.  Written once; a file that is there with other content is refused.  The lead commits it.
"""
import hashlib
import json
from pathlib import Path

from decide_ef import read_json
import hs_core as hc
import mc0_core as core
from hs_core import ALL_PAIRS, HS

SCHEMA = 'bayesreg.p2.reference.v1'             # p2_core.REFERENCE_SCHEMA
DECISION = 'analysis/decision_hs.json'


class ReferenceRefused(Exception):
    """HS does not adopt the segmenter, or the file is there with other content."""


def reference_rel(segmenter_id):
    return f'experiments/p2/reference_{segmenter_id}.json'      # p2_core.reference_rel


def _sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pins(root, pair, row):
    """One pair's pins, the keys of run_p2.v1_reference with HS's directories; row: decision_hs.json's per_pair row of the pair.
    A fallback row (P2 plan revision 2 §3.1) also carries method_metrics, decision_hs.json's per_pair metrics copied as they are; its field_dir is null when the field run read no channel (no run.json, or the
    entry_alignment_qc record, which has no channel_sources), its repair_dir and repair pins null when there is no repair.json (an entry failure, a repair that crashed)."""
    row = row or {}
    line_status = row.get('line_status')
    rcell, fcell = HS.repair_cell(pair), HS.field_cell(pair)
    rdir, fdir = HS.run_dir(root, rcell), HS.run_dir(root, fcell)
    repair, metrics, record = read_json(rdir/'repair.json'), read_json(rdir/'repair_metrics.json') or {}, read_json(fdir/'run.json')
    fallback = line_status in (hc.HX_A0, hc.HX_IDENTITY)
    channels_read = record is not None and record.get('channel_sources') is not None
    if fallback and not channels_read:
        record, fdir = None, None
    if fallback and repair is None:
        rdir, metrics = None, {}
    record, repair = record or {}, repair or {}
    checkpoint = (record.get('checkpoints') or {}).get(str(hc.NATIVE)) or {}
    certificate = (metrics.get('after') or {}).get('certificate') or {}
    out = dict(tier=rcell.tier, repair_dir=None if rdir is None else str(rdir), run_id=repair.get('run_id'), attempt_sha256=(repair.get('attempt') or {}).get('sha256'), repaired_u_sha256=metrics.get('repaired_u_sha256'), certified_at_1=certificate.get('accepted_alpha') == 1. if certificate else None,
               wall_seconds=(repair.get('config') or {}).get('wall_seconds'), field_dir=None if fdir is None else str(fdir), field_sha256=checkpoint.get('sha256'), field_u_sha256=checkpoint.get('u_sha256'), field_D_factor=(record.get('scale') or {}).get('D_factor'),
               hs_line_status=line_status, repair_json_sha256=None if rdir is None else _sha(rdir/'repair.json'), repair_metrics_sha256=None if rdir is None else _sha(rdir/'repair_metrics.json'), run_json_sha256=None if fdir is None else _sha(fdir/'run.json'))
    if fallback:
        out['method_metrics'] = row.get('metrics')
    elif line_status == hc.HX_CERTIFIED and out['certified_at_1'] is not True:
        # hx_certified by the third certificate (C22.3-9 (d)): the record of the fallback unit that holds it and reads_labels false (P2 plan revision 3 §3.1, p2_core.third_certificate_problems at p2-tools 2488907; revision 1d)
        third = Path(HS.fallback_root(root))/core.short_name(pair)/'fallback.json'
        out.update(third_certificate_file=str(third) if third.is_file() else None, third_certificate_sha256=_sha(third))
    return out


def reference_document(root, decision):
    """The reference of the segmenter HS adopted; ReferenceRefused unless decision_hs.json reads 采用新分割器."""
    if decision is None or decision.get('reading') != hc.READ_ADOPT or decision.get('candidate') not in hc.CANDIDATES:
        raise ReferenceRefused(f"HS does not read {hc.READ_ADOPT}: {None if decision is None else decision.get('reading')}; no reference is written")
    candidate = decision['candidate']
    sid = hc.CANDIDATES[candidate]['segmenter_id']
    if decision.get('segmenter_id') != sid:
        raise ReferenceRefused(f"decision_hs.json names {decision.get('segmenter_id')}, the candidate {candidate} is {sid}")
    rows = {row['pair']: row for row in decision.get('per_pair') or []}
    import nn_common as nc
    channels = dict(pred_root=str(hc.seg_dir(candidate, HS.RESULTS)/'pred'), models_root=str(HS.UF_OUT), image_root=str(nc.DATA_ROOT), registry_rel=hc.REGISTRY_REL)
    reading = f"HS {decision.get('reading')}: Delta seg35 {decision.get('compare', {}).get('seg35', {}).get('mean')}"
    return dict(schema=SCHEMA, segmenter_id=sid, channels=channels, written_from=f'JSON records of hs_v1 (experiments/hs/run_hs.py write-reference); no field read; {DECISION} sha256 {_sha(Path(root)/DECISION)}',
                hs=dict(candidate=candidate, reading=decision.get('reading'), decision_sha256=_sha(Path(root)/DECISION), counts=(decision.get('counts') or {}).get('by_status')),
                p2=dict(source=f'hs_v1 RSHx24 c600: the repairs of the frozen Hx24@600 on the channels of {sid}, the first 5 pairs (RF05, wall_seconds 1200); {reading}', pairs={core.short_name(p): pins(root, p, rows.get(core.short_name(p))) for p in core.PAIRS}),
                p3=dict(source=f'hs_v1 RSHx24: the 19 repairs of the frozen Hx24@600 on the channels of {sid} (RF05, wall_seconds 1200); {reading}', hs_reading=decision.get('reading'), pairs={core.short_name(p): pins(root, p, rows.get(core.short_name(p))) for p in ALL_PAIRS}))


def fallback_rows(document):
    """The pairs whose reference value is HS's fallback (method_metrics, hx_fallback_configC or hx_fallback_identity)."""
    return sorted({k for stage in ('p2', 'p3') for k, v in document[stage]['pairs'].items() if v.get('hs_line_status') in (hc.HX_A0, hc.HX_IDENTITY)})


def third_certificate_rows(document):
    """The pairs HS outputs as the repaired field by the third certificate (hx_certified, the evaluation's review certificate not passed: C22.3-9 (d)).  They have no method_metrics; they are not fallback rows (revision 1c, minor 2)."""
    return sorted({k for stage in ('p2', 'p3') for k, v in document[stage]['pairs'].items() if v.get('hs_line_status') == hc.HX_CERTIFIED and v.get('certified_at_1') is not True})


def write(path, document):
    """Once: a file that is there must be the same bytes."""
    path = Path(path)
    text = json.dumps(document, indent=1, ensure_ascii=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ReferenceRefused(f'{path} is there with other content; written once')
        return path
    if not path.parent.is_dir():
        raise ReferenceRefused(f'{path.parent} is not there (are P2\'s tools merged?)')
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
    return path
