"""PQH: the inputs of the hold-out PQ units and their controls Hm1-Hm4 (contract C22.H4-1, -2), and the PQ computation itself (the development package's functions, in pq_io.compute_pair's order).

Where everything is (Layout): the hold-out mode reads holdout_common's roots at call time (the tests move them); the dev mode (the smoke, C22.H8, D124 (a)) reads he_smoke_v3 and the registered development
predictions.  The method's files are found through the finalization record of step 5 (C22.H1-3; HB writes it) and, for a pair moved to D: after it, through holdout_common.move_files' record (sha256 before
and after); every file is bound by the sha256 the finalization record holds, never by a path alone (C22.H1-3 revision 7b n10: the paths in the records are mapped, not compared).

Hm1 (C22.H4-2, D113 (a)): hx_certified -- t88/run.json, field_0600.npz, repair.json, repair.npz bound to the record, v rebuilt (boundary_taper of the frozen tree, L = 4, then repair.npz's nodes and values,
           pq_io.bind_field's arithmetic) and compared bit for bit with output/final.npz's u; the state_hash chain repair.json certificate = decision = cert.json = final.npz = line.json = the record.
           A fallback pair: v = final.npz's u, its state_hash the fallback record's re-certification (line.json's fallback record and the finalization record's fallback); the identity is u = 0.
           Both: final.npz, cert.json, line.json, disp.npz, terminal.json, review/review.json of the record; cert.json passes (C22.3-6); line_status one of the three with output; the review
           consistent, or the third certificate passed on this array (HB's review.json, the last round).  The method's records follow HB (the alignment of 2026-11-09, item 1).
Hm2, Hm3 (C22.Q1-1): read_side below, pq_io.read_side's steps (the structure layer with the bound allowlist, the read-back, the method's reader raw_labels bit for bit, the labels' sha256 against t88/run.json)
           with two hold-out changes: the expected sha256 come from HI's records, not from plan.md's table; and only a fallback pair whose T88 left no labels_sha256 is checked on the first half
           (C22.H4-2 Hm3: "照实记"没有可核的 labels_sha256"及原因"); an hx_certified pair without them fails Hm3 (review I3: run_hx writes them whenever T88 is done).
Hm4:       d = disp_voxels(v, spacing) bit for bit output/disp.npz's d (float64 required), its sha256 (float64 C-order bytes, bayesreg.eval.fields.array_sha256) line.json's output.disp.d_sha256; the spacing
           load_pair's = the manifest's; argmax P~_M = warp_labels_l2r(L_M, d) (pq_io.check_m4).
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for _rel in ('holdout', 'hi'):
    if str(EXPERIMENTS/_rel) not in sys.path:
        sys.path.append(str(EXPERIMENTS/_rel))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pq_core as core  # noqa: E402
import pq_io as io_  # noqa: E402
import pq_pins as pins  # noqa: E402
import pqh_pins as hp  # noqa: E402
from pq_core import PQRefused  # noqa: E402

REPO = HERE.parents[1]


def guard_write(path):
    """I7 (the lead's ruling (8) of 11-09): in the test sandbox (holdout_common.TEST_SANDBOX_ENV set) every path PQH writes, makes, renames or locks is refused when either of its forms lies under
    one of PQH's real roots (pqh_pins.REAL_WRITE_ROOTS: he_smoke_v3, pq_replay_v1, seg/pq_holdout_members.lock, holdout_B_v1, D:'s hold-out directory, pq_v1) or holdout_common's
    (real_root_problem); before anything is written.  Returns the path unchanged; outside the sandbox it does nothing."""
    h = hc()
    if os.environ.get(h.TEST_SANDBOX_ENV):
        forms = {Path(os.path.abspath(path))}
        try:
            forms.add(Path(path).resolve())
        except (OSError, RuntimeError):
            pass
        for form in forms:
            for root in hp.REAL_WRITE_ROOTS:
                if form == root or root in form.parents:
                    raise h.RealRootInTest(f'test sandbox: {path} is under the real {root} (PQH, I7); monkeypatch the roots into tmp_path')
            h.guard_root(form)
    return path


def hc():
    import holdout_common
    return holdout_common


# ---- where everything is -----------------------------------------------------------------------------------------------------------------------
@dataclass
class Layout:
    mode: str
    code: Path                      # the code tree (the detached checkout in production, C22.H0-4)
    pairs: tuple                    # pair names <fixed>__<moving>, in the pair list's order
    stage: Path                     # <stage>/<pair>/{pair.npz, manifest.json}
    pq_small: Path                  # records: /home (C22.H5-7, C22.H6-1)
    pq_big: Path                    # V, pq.npz, baselines.npz: D: in the hold-out mode
    method_records: Path            # hold-out: holdout_B_v1/method/ (each pair's move.json); dev: None (the copy is never moved)
    images: Path
    models_root: Path
    members_lock: Path
    out_lock: Path
    finalized_path: Path = None     # hold-out: holdout_B_v1/review/finalized_outputs.json (HB's step-5 queue); dev: None
    dev_cell: Path = None           # dev: the copy of segment (ii) (its terminal.json and review/review.json stand for the finalization record; HB's dev review writes none)
    hi_lock: Path = None            # hold-out: HI's one-shot lock (its runs give the predictions); dev: None
    dev_pred_root: Path = None      # dev: the registered predictions of the 20 development subjects
    forbidden_dirs: tuple = ()
    ev_master_lock: Path = None
    batch_root: Path = None         # the PQ units' refusal records (holdout_common.write_refusal), inside pq_small
    v4: object = None               # dev: pq_pins.V4
    extra: dict = field(default_factory=dict)
    out_subdir: str = None          # SM3: dev (iv-b), (iv-c): the out unit's records under pq_small/<it> and pq_big/<it> (the members run stays at pq_small, pq_big); None: the roots themselves
    out_unit: str = None            # SM3: dev (iv-b), (iv-c): the out unit's own unit name (pqh_pins.DEV_OUT_UNITS); None: pqh_pins.UNITS / DEV_UNITS['out']

    @property
    def finalized(self):
        """The file that stands for the finalization record: HB's finalized_outputs.json, or in the dev mode the copy's review record."""
        return Path(self.finalized_path) if self.finalized_path is not None else Path(self.dev_cell)/hp.REVIEW_RECORD_REL

    @property
    def subjects(self):
        return sorted({s for p in self.pairs for s in p.split('__')})

    @property
    def pair_set(self):
        return 'holdout' if self.mode == 'holdout' else 'dev'


def holdout_layout(code=REPO, pairs=None):
    """The hold-out mode's places (holdout_common's roots read now; pairs: holdout_B_pairs.tsv in its row order, checked against its pinned sha256)."""
    h = hc()
    import hi_core as hi
    rows = h.holdout_pairs(code) if pairs is None else pairs
    names = tuple(h.pair_name(f, m) for f, m in rows) if rows and isinstance(rows[0], tuple) else tuple(rows)
    root = Path(h.HOLDOUT_ROOT)
    return Layout(mode='holdout', code=Path(code), pairs=names, stage=Path(h.STAGE_ROOT), pq_small=root/hp.PQ_ARM, pq_big=Path(h.BIG_ROOT)/hp.PQ_ARM, method_records=root/hp.METHOD_ARM,
                  finalized_path=root/hp.FINALIZED_DIR/hp.FINALIZED_NAME, images=Path(hp.DATA_IMAGES), models_root=Path(hp.MODELS_ROOT), members_lock=Path(hp.MEMBERS_LOCK),
                  out_lock=Path(hp.OUT_LOCK), hi_lock=Path(hi.HOLDOUT_LOCK),
                  forbidden_dirs=(root/hp.EV_EVAL_DIR_NAME, Path(hp.RESULTS)/'ev', Path(hp.RESULTS)/'p3_v1', Path(hp.PQ_V1), Path(hp.RESULTS)/pins.PRED_ROOT_REL, Path(hp.RESULTS)/'hs_v1',
                                  Path(hp.REPLAY_ROOT)),
                  ev_master_lock=root/hp.EV_EVAL_DIR_NAME/hp.EV_MASTER_LOCK_NAME, batch_root=root/hp.PQ_ARM/'batches')


def dev_layout(code=REPO, smoke_root=None, segment='ii'):
    """The dev mode (the smoke's segments (i)-(iii), C22.H8): the smoke root's pq/ (DEV_PQ_ROOT), segment (ii)'s copy (DEV_CELL_REL), the registered development predictions, the development stage.
    SM3: segment 'iv-b' or 'iv-c' (pqh_pins.DEV_OUT_SEGMENTS): the PQ unit on that segment's copy (pqh_pins.dev_cell_rel), its own lock pq/pq_out_dev_<segment>.lock, its records under
    pq/<segment>/ and pq/big/<segment>/ (out_subdir), its own unit (pqh_pins.DEV_OUT_UNITS); the members run, the batches and the units' records are the smoke's one pq/ (shared with (ii))."""
    if segment not in hp.DEV_OUT_SEGMENTS:
        raise ValueError(f'segment {segment!r}: one of {hp.DEV_OUT_SEGMENTS}')
    root = Path(hp.SMOKE_ROOT if smoke_root is None else smoke_root)
    pq = Path(hp.DEV_PQ_ROOT) if smoke_root is None else root/'pq'
    if segment == 'ii':
        out_lock = Path(hp.DEV_OUT_LOCK) if smoke_root is None else pq/'pq_out_dev.lock'
    else:
        out_lock = pq/f'pq_out_dev_{segment}.lock'
    return Layout(mode='dev', code=Path(code), pairs=(hp.DEV_SMOKE_PAIR,), stage=Path(hp.MAIN_REPO)/pins.STAGE_REL, pq_small=pq, pq_big=pq/'big', method_records=None,
                  dev_cell=root/hp.dev_cell_rel(segment), images=Path(hp.DATA_IMAGES), models_root=Path(hp.MODELS_ROOT),
                  members_lock=Path(hp.DEV_MEMBERS_LOCK) if smoke_root is None else pq/'pq_members_dev.lock', out_lock=out_lock, dev_pred_root=Path(hp.RESULTS)/pins.PRED_ROOT_REL,
                  forbidden_dirs=(Path(hp.PQ_V1)/'eval', Path(hp.PQ_V1)/'analysis', Path(hp.RESULTS)/'ev', Path(hp.REPLAY_ROOT)), batch_root=pq/'batches', v4=pins.V4,
                  out_subdir=None if segment == 'ii' else segment, out_unit=None if segment == 'ii' else hp.DEV_OUT_UNITS[segment])


# ---- small helpers -----------------------------------------------------------------------------------------------------------------------------
def sha_file(path):
    return io_.sha_file(path)


def float64_sha(array):
    """C22.H4-2 Hm4 (revision 7a, m13): float64, C-order bytes (bayesreg.eval.fields.array_sha256); the dtype is checked first."""
    from bayesreg.eval.fields import array_sha256
    array = np.asarray(array)
    if array.dtype != np.float64:
        raise PQRefused('hm4', f'an array of dtype {array.dtype}: the hash is defined on float64 only')
    return array_sha256(array)


def read_json_strict(path, stage):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PQRefused(stage, f'{path}: not readable JSON ({type(exc).__name__}: {exc})') from None


# ---- the finalization record (C22.H1-3) and the files it names ------------------------------------------------------------------------------------
PART_KEYS = ('output', 'state_hash', 't88_repair', 'fallback', 'superseded', 'evidence')      # hb_review.finalized_part's keys that finalize() copies into finalized_outputs.json


def normalized_entry(name, row, problems):
    """One pair of HB's finalized_outputs.json (hb_review.finalize: cell, terminal {path, sha256}, review {path, sha256} | null, line_status, part {output, state_hash, t88_repair, fallback,
    superseded, evidence} | null) as PQ reads it: {line_status, state_hash, directory (the cell when finalized), files {relative path: sha256} (the output's four files, the T88 and repair files
    that are there, terminal.json, review/review.json), fallback {kind, state_hash, certificate_passed} | None}.  Appends to `problems`."""
    if not isinstance(row, dict):
        problems.append(f'{name}: not a record')
        return None
    status = row.get('line_status')
    terminal = row.get('terminal')
    if status not in hp.LINE_STATUSES_WITH_OUTPUT+hp.LINE_STATUSES_WITHOUT_OUTPUT:
        problems.append(f'{name}: line_status {status!r}')
        return None
    if not (isinstance(terminal, dict) and isinstance(terminal.get('sha256'), str)):
        problems.append(f'{name}: no terminal.json in the record (the cell is not written)')
        return None
    entry = dict(line_status=status, state_hash=None, directory=row.get('cell'), files={hp.TERMINAL_NAME: terminal['sha256']}, fallback=None, terminal=terminal, review_record=row.get('review'))
    if status in hp.LINE_STATUSES_WITHOUT_OUTPUT:
        return entry
    part, review = row.get('part'), row.get('review')
    if not (isinstance(part, dict) and isinstance(review, dict) and isinstance(review.get('sha256'), str) and isinstance(row.get('cell'), str)):
        problems.append(f'{name}: an output status without the review record and its finalized part')
        return None
    output, t88 = part.get('output') or {}, part.get('t88_repair') or {}
    if any(not isinstance(output.get(hp.HE_FILES[k]), str) for k in hp.OUTPUT_KEYS):
        problems.append(f'{name}: the output files are not all in the record ({output})')
        return None
    if status == hp.HX_CERTIFIED and any(not isinstance(t88.get(hp.HE_FILES[k]), str) for k in hp.CERTIFIED_KEYS):
        problems.append(f'{name}: hx_certified without the four files of the reconstruction (D113 (a)): {t88}')
        return None
    fallback = part.get('fallback')
    if status in hp.FALLBACK_KINDS:
        if not (isinstance(fallback, dict) and fallback.get('kind') == hp.FALLBACK_KINDS[status] and fallback.get('state_hash') == part.get('state_hash') and fallback.get('certificate_passed') is True):
            problems.append(f'{name}: a {status} pair whose fallback in the record is {fallback} (kind, state_hash, certificate_passed)')
            return None
    files = dict(entry['files'])
    files.update({rel: sha for rel, sha in output.items()})
    files.update({rel: sha for rel, sha in t88.items() if isinstance(sha, str)})
    files[hp.REVIEW_RECORD_REL] = review['sha256']
    entry.update(state_hash=part.get('state_hash'), files=files, fallback=None if status == hp.HX_CERTIFIED else dict(fallback))
    return entry


def read_finalized(layout):
    """(document, sha256 or None): the finalization record of step 5 normalized (normalized_entry), covering every pair of the layout (C22.H3-3: each pair finalized with an output or recorded
    without one).  Hold-out: HB's holdout_B_v1/review/finalized_outputs.json (hb_review.finalize; the alignment of 2026-11-09 item 1).  Dev: HB's dev review writes no such record; the copy's own
    terminal.json and review/review.json (hb_review's finalized_part at its top) stand for the one pair's row.  PQRefused('order') when it is missing or does not hold."""
    problems = []
    if layout.mode == 'dev':
        cell = Path(layout.dev_cell)
        terminal_path, review_path = cell/hp.TERMINAL_NAME, cell/hp.REVIEW_RECORD_REL
        if not (terminal_path.is_file() and review_path.is_file()):
            raise PQRefused('order', f'{cell}: no terminal.json and review/review.json of the review unit (segment (ii) of the smoke): the PQ unit does not start')
        review = read_json_strict(review_path, 'order')
        terminal = read_json_strict(terminal_path, 'order')
        row = dict(cell=str(cell), terminal=dict(path=str(terminal_path), sha256=sha_file(terminal_path)), review=dict(path=str(review_path), sha256=sha_file(review_path)),
                   line_status=terminal.get('line_status'), part={k: review.get(k) for k in PART_KEYS})
        entry = normalized_entry(layout.pairs[0], row, problems)
        if problems:
            raise PQRefused('order', f'{cell}: {problems}')
        return dict(source='dev: the copy\'s terminal.json and review/review.json', code=None, pairs={layout.pairs[0]: entry}), None
    path = layout.finalized
    if not path.is_file():
        raise PQRefused('order', f'{path}: no finalization record of step 5 (C22.H3-3): the PQ unit does not start')
    document = read_json_strict(path, 'order')
    pairs = document.get('pairs') if isinstance(document, dict) else None
    if not isinstance(pairs, dict) or sorted(pairs) != sorted(layout.pairs):
        raise PQRefused('order', f"{path}: it does not cover exactly the {len(layout.pairs)} pairs ({len(pairs or {})} there)")
    entries = {name: normalized_entry(name, row, problems) for name, row in pairs.items()}
    if problems:
        raise PQRefused('order', f'{path}: {problems}')
    return dict(source=str(path), schema=document.get('schema'), code=document.get('code'), frozen_tree=document.get('frozen_tree'), pairs=entries), sha_file(path)


def move_rows(layout, pair):
    """The rows of holdout_common.move_files' records of this method cell, by hb_store.where's rule (HB revision 2, I3; review NI1): move.json passed -> its rows; move.json passed false and
    move_resume.json (hb_store.move_resume: the files left on /home moved on, once, through a committed resume record) passed -> the ok rows of move.json and the rows of move_resume.json; a half
    moved cell without a passed move_resume.json refuses (hm1).  Both under holdout_B_v1/method/<pair>/ (C22.H1-3 last items: after the move to D:, files are known by the sha256 before and
    after).  Returns (rows, record of the records read), or ([], None) when the cell was not moved (the dev mode never moves)."""
    if layout.method_records is None:
        return [], None
    path = Path(layout.method_records)/pair/hp.MOVE_RECORD_NAME
    if not path.is_file():
        return [], None
    record = read_json_strict(path, 'hm1')
    moves = dict(path=str(path), sha256=sha_file(path), passed=record.get('passed'), move_resume=None)
    if record.get('passed') is True:
        return list(record.get('files') or []), moves
    resume_path = path.with_name(hp.MOVE_RESUME_NAME)
    if not resume_path.is_file():
        raise PQRefused('hm1', f'{path}: the move did not pass ({record.get("problems")}, {record.get("error")}) and there is no {hp.MOVE_RESUME_NAME}: half moved (hb_store.where refuses it too)')
    resumed = read_json_strict(resume_path, 'hm1')
    moves['move_resume'] = dict(path=str(resume_path), sha256=sha_file(resume_path), passed=resumed.get('passed'))
    if resumed.get('passed') is not True:
        raise PQRefused('hm1', f'{path}: the move did not pass and {resume_path} did not pass either ({resumed.get("problems")}, {resumed.get("error")}): half moved')
    return [r for r in (record.get('files') or []) if r.get('ok') is True]+list(resumed.get('files') or []), moves


def method_paths(layout, pair, entry):
    """{relative path: Path} of every file the finalization record names for this pair: in the cell it names, or -- after hb_store's move of the whole cell -- where the move record put it (its row's
    `file` the relative path, ok, sha256 before = after = the record's; absolute paths are never compared, C22.H1-3 revision 7b n10); and the record of the moves used.  Only finds; Hm1
    hashes every file it opens."""
    directory = Path(entry['directory'])
    rows, record = move_rows(layout, pair)
    out, moved = {}, []
    for rel, want in (entry['files'] or {}).items():
        path = directory/rel
        if path.is_file():
            out[rel] = path
            continue
        hit = [r for r in rows if r.get('file') == rel and r.get('ok') is True and r.get('sha256_before') == want and r.get('sha256_after') == want]     # m5: by the relative path
        if len(hit) != 1:
            raise PQRefused('hm1', f'{pair}: {rel} is neither at {path} nor in exactly one passed row of the move records ({len(hit)} rows; move.json and, after a half move, move_resume.json)')
        out[rel] = Path(hit[0]['dst'])
        moved.append(rel)
    return out, dict(move_record=record, moved=moved)


# ---- Hm1: the certified final state --------------------------------------------------------------------------------------------------------------
def _load_u(path, key='u'):
    with np.load(path, allow_pickle=False) as z:
        return np.ascontiguousarray(np.array(z[key]))


def read_final(path):
    with np.load(path, allow_pickle=False) as z:
        u = np.ascontiguousarray(np.array(z['u']))
        stored = str(z['state_hash'].item())
        status = str(z['status'].item()) if 'status' in z.files else None
    return u, stored, status


def bind_output(layout, pair, entry, paths, grid, taper):
    """Hm1 for one pair with an output: (v float64 (nz, ny, nx, 3), record).  PQRefused('hm1') on any departure (C22.H4-2: the unit stops, pq_refused.json, BZ)."""
    files = entry['files']
    problems = []

    def need(ok, text):
        if not ok:
            problems.append(text)
    digests = {rel: sha_file(p) for rel, p in paths.items()}
    for rel, want in files.items():
        need(digests.get(rel) == want, f'{rel}: sha256 {digests.get(rel)} is not the finalization record\'s {want}')
    if problems:
        raise PQRefused('hm1', f'{pair}: {problems}')
    status = entry['line_status']
    line = read_json_strict(paths[hp.HE_FILES['line']], 'hm1')
    cert = read_json_strict(paths[hp.HE_FILES['cert']], 'hm1')
    u, stored, final_status = read_final(paths[hp.HE_FILES['final']])
    digest = float64_sha(u) if u.dtype == np.float64 else None
    need(u.dtype == np.float64 and u.shape == tuple(grid.shape)+(3,), f'final.npz u {u.dtype} {u.shape}, the grid {tuple(grid.shape)}')
    chain = dict(final_u=digest, final_stored=stored, cert=cert.get('state_hash'), line=((line.get('output') or {}).get('final') or {}).get('state_hash'), record=entry['state_hash'])
    need(len(set(chain.values())) == 1, f'the state_hash chain differs: {chain}')
    need(not io_.certificate_problems(cert), f'cert.json: {io_.certificate_problems(cert)} (C22.3-6)')
    need(status in hp.LINE_STATUSES_WITH_OUTPUT and line.get('line_status') == status and final_status == status, f"line_status {status!r}, line.json {line.get('line_status')!r}, final.npz {final_status!r}")
    out_files = (line.get('output') or {})
    for key in ('final', 'disp', 'cert'):
        need((out_files.get(key) or {}).get('sha256') == digests[hp.HE_FILES[key]], f'line.json output.{key}.sha256 is not the file\'s')
    terminal = read_json_strict(paths[hp.TERMINAL_NAME], 'hm1')                  # hb_core.write_terminal: line_status, output_dir 'output', review_record
    need(terminal.get('line_status') == status and terminal.get('output_dir') == 'output', f"terminal.json line_status {terminal.get('line_status')!r}, output_dir {terminal.get('output_dir')!r}")
    need((terminal.get('review_record') or {}).get('sha256') == digests[hp.REVIEW_RECORD_REL], "terminal.json's review_record is not this review/review.json")
    review = read_json_strict(paths[hp.REVIEW_RECORD_REL], 'hm1')               # hb_review: C22.H1-5's keys at its top (consistent, third_certificate of the last round, rounds) and finalized_part
    third = review.get('third_certificate') or {}
    need(review.get('finalized') is True and review.get('line_status') == status and review.get('state_hash') == entry['state_hash'],
         f"review.json: finalized {review.get('finalized')!r}, line_status {review.get('line_status')!r}, state_hash {review.get('state_hash')!r}")
    need(review.get('consistent') is True or (third.get('passed') is True and third.get('output_state_hash') == entry['state_hash']),
         f"the review is not consistent and no passing third certificate of this array (consistent {review.get('consistent')!r}, third {third or None})")
    record = dict(line_status=status, state_hash=entry['state_hash'], files=digests, review=dict(consistent=review.get('consistent'), third_certificate=third or None, rounds=len(review.get('rounds') or [])),
                  certificate={k: cert.get(k) for k in ('status', 'reason', 'min_jd_lower_bound', 'n_cells_checked', 'n_infeasible', 'n_uncertain', 'certificate_version')})
    if status == hp.HX_CERTIFIED:
        v, rebuilt = rebuild_v(pair, paths, digests, grid, taper, problems)
        record.update(rebuilt)
        if not problems:
            need(v.tobytes() == u.tobytes(), f'the rebuilt v differs from final.npz on {int(np.sum(v != u))} values (D113 (a))')
            need(rebuilt['state_hash'] == entry['state_hash'], f"sha256 of the rebuilt v {rebuilt['state_hash']} is not the record's {entry['state_hash']}")
        record['rebuilt_equals_final'] = not problems
    else:
        kind = hp.FALLBACK_KINDS[status]
        fallback = line.get('fallback') or {}                                     # HE writes the whole fallback record into line.json (run_hx: fallback_chain's record, with its certificate)
        fallback_cert = fallback.get('certificate') or {}
        need(fallback.get('kind') == kind and entry['fallback'].get('kind') == kind, f"fallback kind: line.json {fallback.get('kind')!r}, finalization record {entry['fallback'].get('kind')!r}, expected {kind!r}")
        hashes = dict(line_fallback=fallback.get('state_hash'), fallback_certificate=fallback_cert.get('state_hash'), record=entry['fallback'].get('state_hash'))
        need(all(h == entry['state_hash'] for h in hashes.values()), f'the fallback record\'s re-certification is not of this array: {hashes} vs {entry["state_hash"]}')
        need(not io_.certificate_problems(fallback_cert), f'the fallback certificate: {io_.certificate_problems(fallback_cert)}')
        if kind == 'identity':
            need(not np.any(u), 'an identity fallback whose u is not 0')
        v = u
        record.update(fallback=dict(kind=kind, state_hash=fallback.get('state_hash'), run_dir=fallback.get('run_dir'), certificate_passed=not io_.certificate_problems(fallback_cert)),
                      rebuilt_equals_final=None, rebuild="a fallback pair: v is final.npz bound to the fallback record's re-certification in line.json (C22.H4-1)")
    if problems:
        raise PQRefused('hm1', f'{pair}: {problems}')
    run_json = paths.get(hp.HE_FILES['run_json'])
    record['labels_sha256'] = None
    if run_json is not None:
        run = read_json_strict(run_json, 'hm1')
        record['labels_sha256'] = run.get('labels_sha256') if isinstance(run.get('labels_sha256'), dict) else None
        record['labels_sha256_absent_reason'] = None if record['labels_sha256'] else f"t88/run.json status {run.get('status')!r}, reason {str(run.get('reason'))[:300]!r}"
    else:
        record['labels_sha256_absent_reason'] = 't88/run.json is not in the finalization record (a fallback pair whose T88 left no record)'
    labels = record['labels_sha256']
    if status == hp.HX_CERTIFIED and not (isinstance(labels, dict) and all(isinstance(labels.get(k), str) and len(labels[k]) == 64 for k in ('fixed', 'moving'))):
        raise PQRefused('hm3', f"{pair}: hx_certified, but t88/run.json holds no labels_sha256 of both sides ({labels!r}): run_hx writes them whenever T88 is done (I3; C22.H4-2 Hm3's "
                               "'照实记' is for a T88 that did not read its channels, i.e. a fallback pair)")
    return np.ascontiguousarray(v, dtype=np.float64), record


ANCHOR_START_KEYS = ('anchor', 'settings_segmenter', 'registry', 'registry_rel', 'registry_sha256', 'model_segmenter_ids', 'rule')     # he_core.segmenter_start_problems's record


def anchor_problems(run):
    """m4, corrected by review NB1: run_hx writes segmenter_anchor = {start, start_problems, after_channels_problems} (run_hx.py:379), start being he_core.segmenter_start_problems's record
    (he_core.py:209): {anchor: CONTRACT_SEGMENTER, settings_segmenter, registry, registry_rel, registry_sha256 (read then), model_segmenter_ids, rule}.  Problems unless both problem lists are empty,
    start['anchor'] and start['settings_segmenter'] are this package's anchor (pqh_pins.ANCHOR) item by item, start['registry_rel'] its registry path, start['registry_sha256'] its sha256, and every
    model of start['model_segmenter_ids'] names its segmenter_id."""
    want = dict(hp.ANCHOR)
    record = run.get('segmenter_anchor')
    if not isinstance(record, dict):
        return [f'no segmenter_anchor record ({record!r})']
    problems = [f'{k} {record.get(k)!r} is not []' for k in ('start_problems', 'after_channels_problems') if record.get(k) != []]
    start = record.get('start') if isinstance(record.get('start'), dict) else {}
    for key in ('anchor', 'settings_segmenter'):
        if not isinstance(start.get(key), dict) or dict(start[key]) != want:
            problems.append(f"start['{key}'] {start.get(key)!r} is not the anchor {want}")
    if start.get('registry_rel') != want['registry'] or start.get('registry_sha256') != want['registry_sha256']:
        problems.append(f"start registry {start.get('registry_rel')!r} / {start.get('registry_sha256')!r} is not the anchor's {want['registry']} / {want['registry_sha256']}")
    ids = start.get('model_segmenter_ids')
    if not isinstance(ids, dict) or not ids or any(v != want['segmenter_id'] for v in ids.values()):
        problems.append(f"start['model_segmenter_ids'] {ids!r} do not all name {want['segmenter_id']}")
    return problems


def rebuild_v(pair, paths, digests, grid, taper, problems):
    """D113 (a), C22.Q1-2: v from t88/field_0600.npz (boundary_taper, L = 4) with repair.npz's nodes and values (pq_io.bind_field's arithmetic: v.reshape(-1, 3)[nodes] = values on the tapered copy).
    Appends to `problems`; returns (v, record)."""
    def need(ok, text):
        if not ok:
            problems.append(text)
    run = read_json_strict(paths[hp.HE_FILES['run_json']], 'hm1')
    repair = read_json_strict(paths[hp.HE_FILES['repair_json']], 'hm1')
    field_sha = digests[hp.HE_FILES['field']]
    run_field = run.get('field') or {}
    source, attempt, decision, cert = (repair.get(k) or {} for k in ('source', 'attempt', 'decision', 'certificate'))
    need(run.get('status') == 'done' and run.get('oracle') is False and run.get('pair') == pair, f"run.json status {run.get('status')!r}, oracle {run.get('oracle')!r}, pair {run.get('pair')!r}")
    for problem in anchor_problems(run):
        need(False, f'run.json segmenter_anchor: {problem}')
    need(repair.get('pair') == pair, f"repair.json pair {repair.get('pair')!r}")
    need(run_field.get('file') == 'field_0600.npz' and run_field.get('sha256') == field_sha, f'run.json field {run_field}')
    need(source.get('file') == 'field_0600.npz' and source.get('field_sha256') == field_sha, f"repair.json source {source.get('file')} {source.get('field_sha256')}")
    need(source.get('directory') is not None and Path(str(source.get('directory'))).name == 't88', f"repair.json source.directory {source.get('directory')!r} (compared by its last part, C22.H1-3 n10)")
    need(repair.get('status') == 'certified' and repair.get('passed') is True, f"repair.json status {repair.get('status')!r}, passed {repair.get('passed')!r}")
    need(attempt.get('file') == 'repair.npz' and attempt.get('sha256') == digests[hp.HE_FILES['repair_npz']], f'repair.json attempt {attempt}')
    need(not io_.certificate_problems(cert), f'repair.json certificate: {io_.certificate_problems(cert)}')
    need(cert.get('state_hash') == decision.get('state_hash'), f"repair.json certificate.state_hash {cert.get('state_hash')} is not decision.state_hash {decision.get('state_hash')}")
    if problems:
        return None, {}
    raw = _load_u(paths[hp.HE_FILES['field']])
    if raw.shape != tuple(grid.shape)+(3,) or raw.dtype != np.float64:
        problems.append(f'field_0600.npz u {raw.dtype} {raw.shape}, the grid {tuple(grid.shape)}')
        return None, {}
    need(run_field.get('u_sha256') in (None, io_.sha_array(raw)), f"run.json field.u_sha256 {run_field.get('u_sha256')} is not u's {io_.sha_array(raw)}")
    tapered, _ = taper(raw, grid, hp.TAPER_L)
    v = np.array(tapered, dtype=np.float64, order='C', copy=True)
    tapered_sha = io_.sha_array(v)
    need(tapered_sha == source.get('tapered_u_sha256'), f"the tapered field {tapered_sha} is not source.tapered_u_sha256 {source.get('tapered_u_sha256')}")
    with np.load(paths[hp.HE_FILES['repair_npz']], allow_pickle=False) as z:
        nodes, values = np.array(z['nodes']), np.array(z['values'])
    if values.shape != (nodes.size, 3) or nodes.size != attempt.get('n_free_nodes', nodes.size):
        problems.append(f'repair.npz nodes {nodes.shape}, values {values.shape}, n_free_nodes {attempt.get("n_free_nodes")}')
        return None, {}
    v.reshape(-1, 3)[nodes] = values
    state = hashlib.sha256(memoryview(v).cast('B')).hexdigest()
    need(state == cert.get('state_hash'), f"sha256 of v {state} is not repair.json certificate.state_hash {cert.get('state_hash')}")
    return v, dict(rebuilt=dict(tapered_u_sha256=tapered_sha, n_free_nodes=int(nodes.size), taper=hp.TAPER_L, already_certified=bool(nodes.size == 0)), state_hash=state,
                   source_directory=str(source.get('directory')), run_json_sha256=digests[hp.HE_FILES['run_json']])


# ---- HI's predictions (C22.H4-1 "P_F, P_M"; Hm2) -------------------------------------------------------------------------------------------------
HI_LOCK_PATTERN = r'hi_holdout_infer\.lock(\.resume_[0-9a-f]{16})?'
HI_RUN_FILES = ('hi_record.json', 'hi_stopped.json', 'hi_started.json')          # the first run's names (hi_core.run_names(0)); a resume's are hi_*_resume_<k>.json: hi_files lists both


def hi_run_outs(lock):
    import hi_core as hi
    return [Path(o) for o in hi.run_outs(lock)]


def hi_predictions(lock, subjects):
    """{subject: {pred_dir, record, record_sha256, output_sha256, run}} of HI's hold-out inference: the runs of its lock taken together (hi_core.run_consistency_problems, finished_subjects; revision 7a m12),
    each subject finished in exactly one run, its subjects/<subject>.json read (output_sha256).  PQRefused('order') when HI has not run or not finished every subject (C22.H3-3: the order, a
    refusal record); PQRefused('hm2') when its runs are inconsistent or a record lacks the outputs' sha256 (a control: pq_refused.json, review m3)."""
    import hi_core as hi
    lock = Path(lock)
    if not lock.is_file():
        raise PQRefused('order', f'{lock}: HI\'s hold-out inference has not run (C22.H3-3)')
    problems = hi.run_consistency_problems(lock)
    finished = hi.finished_subjects(lock)
    missing = sorted(set(subjects)-finished)
    if problems:
        raise PQRefused('hm2', f'HI\'s runs are not consistent: {problems}')
    if missing:
        raise PQRefused('order', f'HI\'s inference is not complete: subjects no run finished {missing[:10]} ({len(missing)}) (C22.H3-3)')
    out = {}
    for run in hi_run_outs(lock):
        for subject in hi.subject_records(run):
            if subject in subjects:
                out.setdefault(subject, []).append(run)
    twice = sorted(s for s, runs in out.items() if len(runs) != 1)
    if twice:
        raise PQRefused('hm2', f'subjects with a record in more than one HI run: {twice}')
    rows = {}
    for subject in subjects:
        run = out[subject][0]
        path = run/'subjects'/f'{subject}.json'
        record = read_json_strict(path, 'hm2')
        output = record.get('output_sha256') or {}
        if not all(isinstance(output.get(k), str) and len(output[k]) == 64 for k in ('probs36.npz', 'structure_groups.npz')):
            raise PQRefused('hm2', f'{path}: no output_sha256 of probs36.npz and structure_groups.npz')
        rows[subject] = dict(pred_dir=run/'pred'/subject, record=str(path), record_sha256=sha_file(path), output_sha256=dict(output), run=str(run))
    return rows


def hi_files(lock, subjects):
    """The files of HI's runs this unit may open (C22.H4-3 "HI 留出集推断的一次性锁与补跑记录，只为核对 hi_record.json 属于那一次运行"; the subjects' records and predictions).
    A5b item 13 (A5a implementation notes 6.2, the table's PQH row): a resume writes into the first run's root (HOLDOUT_OUT_SUGGESTED) with its own records hi_*_resume_<k>.json, which
    hi_core.run_consistency_problems opens: the run records are hi_core.run_files(root) (RUN_FILE: the first run's and every resume's that are there), not the first run's three names only.  Nothing
    under attempt<k>/ (a killed run's residue, moved out of pred/ before the resume: not a prediction) and no other file of the root."""
    import hi_core as hi
    files = []
    for run in hi_run_outs(lock):
        files += [Path(p) for p in hi.run_files(run)]
        files += [run/'subjects'/f'{s}.json' for s in subjects]
        files += [run/'pred'/s/name for s in subjects for name in ('probs36.npz', 'structure_groups.npz', 'provenance.json')]
    return files


# ---- Hm2, Hm3: one side's posterior ----------------------------------------------------------------------------------------------------------------
def structure_layer(code, pin, pred_root, allowlist, anchor=None):
    """pq_io.structure_layer with the bound allowlist (C23.4-8: StructureLayer receives the very object load_pair took) and the predictions' root of this side (HI's runs may be two)."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.structure_layer import REGISTERED, StructureLayer, StructureSource
    anchor = hp.ANCHOR if anchor is None else anchor
    registry = Path(code)/anchor['registry']
    src = StructureSource(root=Path(pred_root), registry_path=registry, allowlist_path=Path(code)/pins.ALLOWLIST_REL, models_root=Path(hp.MODELS_ROOT), image_root=Path(hp.DATA_IMAGES),
                          segmenter_id=anchor['segmenter_id'], infer='never', formal_registry=registry)
    try:
        layer = StructureLayer(pin, src, allowlist=allowlist)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise PQRefused('hm2', f'{pin.fixed_subject} <- {pin.moving_subject}: the structure layer refused ({type(exc).__name__}: {exc})') from None
    record = layer.registry_record
    git = record.get('git') or {}
    if record['status'] != REGISTERED or not (git.get('tracked') is True and git.get('clean') is True):
        raise PQRefused('hm2', f"the registry {record['path']}: status {record['status']}, git tracked {git.get('tracked')}, clean {git.get('clean')}")
    if layer.registry.sha256 != anchor['registry_sha256']:
        raise PQRefused('hm2', f"the registry sha256 {layer.registry.sha256} is not the anchor's {anchor['registry_sha256']}")
    return layer


def read_side(layer, pin, side, *, want, labels_sha256, anchor=None):
    """Hm2 and Hm3 of one side: (P float64 (36, ...), L int16, record).  pq_io.read_side (experiments/pq/pq_io.py) step for step; the two hold-out changes of the module docstring.
    want: {'probs36', 'structure_groups'} sha256 (and 'provenance' in the dev mode: plan.md's table).  labels_sha256: t88/run.json's dict, or None (the second half of Hm3 not checked, recorded)."""
    from jd_curl.validation import InputValidationError
    from bayesreg.io.pair import ProvenanceRejected
    from bayesreg.seg.channels import read_back_probabilities
    import fast_channels
    import mc0_core
    anchor = hp.ANCHOR if anchor is None else anchor
    subject = layer.subjects[side]
    try:
        groups, accepted, rec = layer.load(side)
    except (ProvenanceRejected, InputValidationError) as exc:
        raise PQRefused('hm2', f'{side} {subject}: {type(exc).__name__}: {exc}') from None
    directory = Path(rec['directory'])
    prov = io_.read_json(directory/'provenance.json')
    problems = []
    for name, key, have in (('probs36.npz', 'probs36', rec['verified_outputs'].get('probs36.npz')), ('structure_groups.npz', 'structure_groups', rec['verified_outputs'].get('structure_groups.npz')),
                            ('provenance.json', 'provenance', rec['provenance_sha256'])):
        if key in want and have != want[key]:
            problems.append(f"{name} sha256 {have} is not the expected {want[key]}")
    for key in ('segmenter_id', 'policy', 'registry_sha256'):
        if prov.get(key) != anchor[key]:
            problems.append(f'provenance {key} {prov.get(key)!r} is not the anchor\'s {anchor[key]!r}')
    if prov.get('trained_with_labels') is not True or prov.get('test_labels_seen') is not False:
        problems.append(f"trained_with_labels {prov.get('trained_with_labels')!r}, test_labels_seen {prov.get('test_labels_seen')!r}")
    if problems:
        raise PQRefused('hm2', f'{side} {subject}: {problems}')
    g = layer.grouping
    shape = tuple(pin.grid.shape)
    try:
        fast = fast_channels.raw_labels(directory/'probs36.npz', (core.K,)+shape, groups, (g.background_labels,)+tuple(g.group_labels))
    except SystemExit as exc:
        raise PQRefused('hm3', f'{side} {subject}: the method\'s reader refused: {exc}') from None
    del groups
    if fast is None:
        raise PQRefused('hm3', f'{side} {subject}: raw_labels gives None (another storage, a sign bit, NaN or Inf): the method\'s fast path did not decide')
    L_raw, _, fast_record, deviation = fast
    group_tol = mc0_core.make_inputs().GROUP_TOL
    if not deviation <= group_tol:
        raise PQRefused('hm3', f'{side} {subject}: probs36 summed by group differs from structure_groups.npz by {deviation:.3e} > {group_tol:g}')
    try:
        P, read_back = read_back_probabilities(directory/'probs36.npz', shape=(core.K,)+shape)
    except InputValidationError as exc:
        raise PQRefused('hm3', f'{side} {subject}: {exc}') from None
    P = np.asarray(P)
    L = core.argmax_smallest(P)
    if L.dtype != L_raw.dtype or L.tobytes() != np.ascontiguousarray(L_raw).tobytes():
        raise PQRefused('hm3', f'{side} {subject}: argmax of the read-back posterior is not raw_labels\' L bit for bit ({int((L != L_raw).sum())} voxels differ)')
    digest = io_.sha_array(L)
    if labels_sha256 is not None and digest != labels_sha256.get(side):
        raise PQRefused('hm3', f"{side} {subject}: sha256 of L {digest} is not the T88 run.json labels_sha256 {labels_sha256.get(side)}")
    return P, L, dict(subject=subject, directory=str(directory), provenance_sha256=rec['provenance_sha256'], probs36_sha256=rec['verified_outputs'].get('probs36.npz'),
                      structure_groups_sha256=rec['verified_outputs'].get('structure_groups.npz'), read_back=read_back, raw_labels=fast_record, group_sum_max_abs_deviation=deviation, group_tol=group_tol,
                      labels_sha256=digest, labels_checked_against_run_json=labels_sha256 is not None, models=rec.get('models'), segmenter_id=prov.get('segmenter_id'), policy=prov.get('policy'),
                      registry_sha256=prov.get('registry_sha256'), accepted=dict(check=accepted.get('check'), model_ids=accepted.get('model_ids'),
                                                                                 label_contact_subjects_sha256=accepted.get('label_contact_subjects_sha256')))


# ---- load_pair, Hm4 ---------------------------------------------------------------------------------------------------------------------------------
def load_pair_checked(layout, pair, allowlist):
    """load_pair (the only reader of pair.npz here: fixed, moving, initial_u) with the bound allowlist; the spacing and shape of the manifest (pq_io.load_pair_checked's check)."""
    from bayesreg.io.pair import _keep_whitelisted_keys, load_pair
    d = Path(layout.stage)/pair
    pin = load_pair(d/'pair.npz', allowlist=allowlist)
    manifest = json.loads((d/'manifest.json').read_text(encoding='utf-8'), object_pairs_hook=_keep_whitelisted_keys)
    spacing = [float(v) for v in pin.grid.spacing]
    if spacing != [float(v) for v in manifest['spacing_xyz']] or list(pin.grid.shape) != list(manifest['shape_zyx']):
        raise PQRefused('hm4', f"{pair}: grid spacing {spacing} / shape {list(pin.grid.shape)} differ from the manifest's {manifest['spacing_xyz']} / {manifest['shape_zyx']}")
    if f'{pin.fixed_subject}__{pin.moving_subject}' != pair:
        raise PQRefused('hm4', f'{pair}: the pair is {pin.fixed_subject} <- {pin.moving_subject}')
    return pin


def check_disp(v, pin, paths, line):
    """Hm4's first item: d = disp_voxels(v, spacing) bit for bit output/disp.npz's d (float64), its float64 sha256 line.json's output.disp.d_sha256.  Returns (d, record)."""
    d = core.disp_voxels(v, tuple(pin.grid.spacing))
    with np.load(paths[hp.HE_FILES['disp']], allow_pickle=False) as z:
        stored = np.array(z['d'])
    if stored.dtype != np.float64:
        raise PQRefused('hm4', f'disp.npz d is {stored.dtype}, float64 required (the hash rule, revision 7a m13)')
    mine = float64_sha(d)
    recorded = ((line.get('output') or {}).get('disp') or {}).get('d_sha256')
    if stored.shape != d.shape or stored.tobytes() != np.ascontiguousarray(d).tobytes() or mine != recorded:
        raise PQRefused('hm4', f'disp_voxels(v) is not disp.npz\'s d bit for bit, or its sha256 {mine} is not line.json\'s {recorded}')
    return d, dict(d_sha256=mine, rule='float64 C-order bytes (bayesreg.eval.fields.array_sha256)', spacing_xyz=[float(s) for s in pin.grid.spacing])


# ---- the members' variance (Hm5, C22.H5-5) ------------------------------------------------------------------------------------------------------
MEMBERS_RECORD, MEMBERS_RESUME_RECORD = 'members_record.json', 'members_record.resume.json'


def run_dirs(layout, unit, kind):
    """(record directory, big-file directory) of a run of a unit: the first run and its resume share them; a rerun after a failed control writes beside them, never over them (C22.H8 拒绝之后的重跑:
    "原记录保留、不覆盖")."""
    small, big = Path(layout.pq_small), Path(layout.pq_big)
    if unit == 'members':
        name = 'members_rerun' if kind == 'control_rerun' else 'members'
        return small/name, big/name
    if getattr(layout, 'out_subdir', None):                 # SM3: the dev mode's (iv-b), (iv-c) out runs, each under its own directory (the members run above stays shared)
        small, big = small/layout.out_subdir, big/layout.out_subdir
    return (small/'rerun', big/'rerun') if kind == 'control_rerun' else (small, big)


def members_dirs(layout):
    """The members run whose records count: the rerun after a failed control when it wrote its record, else the first run (with its resume)."""
    small, big = run_dirs(layout, 'members', 'control_rerun')
    if (small/MEMBERS_RECORD).is_file():
        return small, big
    return run_dirs(layout, 'members', 'first_run')


def members_view(layout, subjects):
    """The members unit's records taken together (the run that counts, members_dirs, and at most one resume of it, D123 (b)): every subject in exactly one record with Hm5 passed, the records' spy m6
    passed.  Returns ({subject: row with its V path}, record of the records); PQRefused('order') when the unit has not run or a subject has no record (C22.H3-3: the order, a refusal record),
    PQRefused('hm5') when a record's spy or a subject's Hm5 did not pass (C22.H3-3: "成员推断单元的记录完整，Hm5 过"; a control: pq_refused.json, review m3)."""
    small, big = members_dirs(layout)
    rows, records = {}, []
    for name in (MEMBERS_RECORD, MEMBERS_RESUME_RECORD):
        path = small/name
        if not path.is_file():
            continue
        document = read_json_strict(path, 'hm5')
        records.append(dict(path=str(path), sha256=sha_file(path), error=document.get('error'), m6=(document.get('spy') or {}).get('m6')))
        if (document.get('spy') or {}).get('m6') is not True:
            raise PQRefused('hm5', f'{path}: the members unit\'s spy did not pass (Hm6): both PQ units refuse until BZ decides (C22.H4-2 last item)')
        for subject, row in (document.get('subjects') or {}).items():
            if subject in rows:
                raise PQRefused('hm5', f'{subject} is in two members records')
            rows[subject] = dict(row, record=str(path), var_path=str(big/subject/io_.VAR_FILE))
    if not records:
        raise PQRefused('order', f"{small/MEMBERS_RECORD}: the members unit has not run (D91 (a): V_F, V2 are judged baselines; C22.H3-3)")
    failed = sorted(s for s in subjects if s in rows and rows[s].get('m5') is not True)
    if failed:
        raise PQRefused('hm5', f'Hm5 not passed: Hm5 false {failed}')
    missing = sorted(set(subjects)-set(rows))
    if missing:
        raise PQRefused('order', f'the members unit has not finished: subjects without a record {missing[:10]} ({len(missing)}) (C22.H3-3)')
    return rows, records


def read_variance(rows, subject, shape):
    """V of one subject with the recorded sha256 (pq_io.read_variance's checks, at the path of the run that counts)."""
    path = Path(rows[subject]['var_path'])
    want = rows[subject].get('var_sha256')
    if not path.is_file() or sha_file(path) != want:
        raise PQRefused('hm5', f'{path}: missing or not the recorded sha256 {want}')
    with np.load(path, allow_pickle=False) as z:
        V = np.array(z['V'])
    if V.dtype != np.float32 or V.shape != tuple(shape):
        raise PQRefused('hm5', f'{path}: V {V.dtype} {V.shape}, expected float32 {tuple(shape)}')
    return V


# ---- the PQ computation (pq_io.compute_pair's order and functions) -------------------------------------------------------------------------------
def pq_arrays(*, v, pin, read_fixed, read_moving, V_F, V_M, lookup):
    """The arrays and numbers of one pair, pq_io.compute_pair's arithmetic in its order (experiments/pq/pq_io.py compute_pair): read_fixed() and read_moving() give (P, L, record).
    Returns (pq {q, q4} float32, baselines {BASELINE_KEYS}, extra {numbers, tail masses, eps, d_sha256 (pq_io.sha_array), n_outside, fixed, moving, q_range})."""
    P_F, L_F, fixed = read_fixed()
    H_F = core.entropy(P_F)
    tail_F = core.tail_masses(P_F, L_F)
    P_M, L_M, moving = read_moving()
    H_M = core.entropy(P_M)
    d = core.disp_voxels(v, tuple(pin.grid.spacing))
    iota = core.sample_index(d)
    maps = core.pq_maps(P_F, P_M, iota, lookup)
    L_tilde = io_.check_m4(maps['top_tilde'], L_M, d)
    tail_M = core.tail_masses(P_M, L_tilde, iota)
    del P_F, P_M
    numbers = core.pq_numbers(maps, L_F, L_tilde, L_M, lookup)
    q = maps['q']
    omega = core.omega_pred(L_F, L_tilde)
    eps = core.eps_em(1.-q[omega])
    base = core.baseline_maps(L_F=L_F, L_tilde=L_tilde, H_F=H_F, H_M=H_M, V_F=V_F, V_M=V_M, F=pin.fixed, iota=iota, spacing_xyz=tuple(pin.grid.spacing))
    pq = dict(q=q.astype(np.float32), q4=maps['q4'].astype(np.float32))
    extra = dict(numbers=numbers, tail_mass=dict(fixed=tail_F.tolist(), moving_pulled_back=tail_M.tolist(), rule='sum_x P^c 1[L != c], c = 0..35; the moving side pulled back with L~_M'),
                 q_range=[float(q.min()), float(q.max())], eps_em=dict(eps, d89='(a) off: described only, pi not stored', omega_pred_voxels=int(omega.sum()), floor=core.EPS_FLOOR, ceiling=core.EPS_CEILING),
                 d_sha256=io_.sha_array(d), n_outside=int((iota < 0).sum()), fixed=fixed, moving=moving)
    return pq, {k: base[k] for k in core.BASELINE_KEYS}, extra


def check_hm4_top(read_moving, v, pin):
    """Pass 1's last item of Hm4 (pq_io.check_pair's): argmax P~_M = warp_labels_l2r(L_M, d), without the rest of the computation."""
    P_M, L_M, moving = read_moving()
    d = core.disp_voxels(v, tuple(pin.grid.spacing))
    iota = core.sample_index(d)
    io_.check_m4(io_.nearest_top(P_M, iota), L_M, d)
    return dict(n_outside=int((iota < 0).sum()), moving=moving['subject'])
