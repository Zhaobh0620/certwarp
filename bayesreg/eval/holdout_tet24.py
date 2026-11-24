"""bayesreg.eval.holdout_tet24 -- the 24-tetrahedron proxy of the B-line hold-out, computed after the unsealing (contract C24.12, revision 9d; BZ 2026-11-19 (乙):
docs/decision_log.md line 520 "| 2026-11-19 | **BZ：24 四面体代理选 (乙)；…" (commit 36a40941); task brief runs/20261119-t24/任务书.md; revision 9d-1 after the review
docs/bayesreg/reviews/T24工具_复核_20261119.md (0/4/11) and the lead's rulings of 2026-11-19 ~12:06 PM CST (main 6a50f0ea)).

What: the preregistration's descriptive endpoint "24 四面体代理（scripts/folding_metrics.py）" (留出集预登记_20261107.md:283, the row 折叠), which the evaluation entry did
not compute (runs/20261107-ev/实现说明.md:58; decision_log 2026-11-19 ~3:14 AM CST).  For every (arm, pair) cell that step 11 (bayesreg.eval.holdout) scored with an
output: scripts/folding_metrics.tet_ratio_summary (schemes 24, 6, 5; called as it is, never copied) of the very u the cell's BL1 certificate got in step 11 --
  d  = fields.load_field(fields.describe_run(<field dir>, pair=..., kind=...), shape_zyx, spacing_xyz)  (holdout.score_cell's call, the same arguments),
  u  = fields.d_to_u(d, spacing_xyz)                                                                   (holdout.score_d's call before certify_field_bl1),
with the field file's sha256 the manifest's before and after the load, d's sha256 the step-11 record's, and sha256(u) the step-11 BL1 certificate's state_hash (the
certificate hashes the float64 C-order bytes of the u it certifies: cert_eval.certify_jacobian_only), so u is BL1's u bit for bit or the run stops.
Described only: no truth, no image, no label, no PQ file, no hold file is read; no other metric; no record of steps 1-12 (holdout_eval.json, pairs/*.json,
decision.json) changes; no decision changes.

The input convention of tet_ratio_summary (checked, contract C24.12-4): u of shape (nz, ny, nx, 3), the last axis the components (x, y, z) in mm, spacing (hx, hy, hz):
_corner_positions slices u[z, y, x] and adds the corner offsets (ox hx, oy hy, oz hz) to the component axis.  fields.d_to_u returns exactly that: d is (3, nz, ny, nx)
in voxels with components (z, y, x); u[..., 2 - k] = d[k] h_(z,y,x)[k]; spacing_xyz is the stage manifest's (holdout.check_stage -> bayesreg.io.pair._read_manifest ->
make_grid_3d, whose spacing is xyz and shape zyx).  The tests check it on anisotropic synthetic fields against jd_curl's Q1 cell geometry (the certificate's).

Reads (C24.12-3), in this order, under the evaluation's spy (holdout.start_spy, holdout.OpenedFiles; the data root watched from the start, then the evaluation's watched
roots and its eval directory):
  - committed files of this code tree: the pinned pair list, BZ's unseal record and the allowlist (holdout.admit_holdout: SG's load_unseal, bind_unseal,
    check_subject; the sealed lists read by the guard), the pinned preregistration and arm table (holdout.check_prereg, load_arm_table), the D108 manifest
    runs/holdout_B_v1/eval_manifest.txt (holdout.read_committed, parse_manifest; its header the pinned three sha256);
  - the evaluation's lock lines (holdout.run_lines: the master lock and the reevaluation locks): the manifest, the arm table, the preregistration and P1 they name,
    and the output directories of the runs;
  - the step-11 per-pair records <out>/pairs/<pair>.json of those runs (holdout_decide.collect, the decision code's reader, which reads each whole file): they hold
    numbers computed from truth (per-label Dice, HD95, PQ); only each arm's outcome, field, certificate's four keys and the record's grid are taken (slim_records),
    the rest is dropped at once and never written or printed (the lead's ruling (1), decision_log 2026-11-19 ~12:06 PM CST);
  - then the spy allows only: those lock files and records, the pairs' stage manifest.json and the planned field files -- each field a line of the manifest at the
    sha256 the step-11 record names;
  - each pair's stage manifest.json (holdout.check_stage: hashed against the manifest first; the grid and the direction; pair.npz never opened);
  - each planned field file (hashed against the manifest before and after the load).
Nothing else under a watched root: an open of anything else there is refused at the open (holdout.SpyRefused) and stops the run, also when the cell's code swallows
the refusal (spy.refused is compared before and after every cell; after the loop any refusal, truth or outside open stops the run).

Once (C24.12-5): the lock T24_ROOT/tet24.lock (O_EXCL, written after every check, before the first field).  Each (arm, pair) cell is computed once; its result
T24_ROOT/cells/<arm>/<position>.json is written once, atomically (position = the row index 0..49 of the pinned pair list, as holdout_decide).  A cell that fails
(anything but a stop) keeps its failure record cells/<arm>/<position>.failed_<attempt>.json and the run goes on; it is computed again only under a committed
authorization splits/bayesreg_v1/holdout_B_v1_tet24_rerun_<k>.json, k = 1..999 (D109's way: written by BZ or the lead on BZ's behalf, an infrastructure reason class,
the lock's sha256, the cells by arm and position; at most once per cell) with its own lock rerun_<k>.lock.  Stops (exit 4): a field that changed (ManifestChanged),
an open the spy refuses (SpyRefused), a field or u that is not step 11's (FieldMismatch), any refusal, truth or outside open the spy saw.  After a stop the cells
left are missing; BZ decides (the lead's ruling (4)), no rerun authorization is accepted.

Code identity (C24.12-2): rule one (bayesreg.io.code_identity.unit_identity_problems: P1 from runs/holdout_B_v1/pinned_code.json; P1..HEAD commit by commit) with one
exception: exactly one commit of P1..HEAD adds MODULE_REL and changes nothing else, and the working tree's code differs from P1 by this file only (and by the allowlist
under exception B); the module's blob at HEAD equals the working tree's and the blob at the reviewed commit given by --reviewed-commit, which is an ancestor of (or)
main, changes the module against its first parent, and whose module is main's.  bayesreg and scripts/folding_metrics.py are imported from this code tree; the code
tree lies under no watched root, the eval directory, T24_ROOT or the step-11 checkout.  The machine gate (holdout.machine_gate, own unit tag-hb-tet24).

Outputs (T24_ROOT, a new directory beside the evaluation's, under no watched root): tet24.lock; run_<attempt>.json (the run record: identity, admission, manifest, the
cells' provenance -- paths that hold pair names --, the spy's record and its counts); run_<attempt>.public.json (no subject id, pair name, position or path: counts,
the stop, the exit code, the spy's counts); cells/<arm>/<position>.json (numbers and booleans only); summary_<attempt>.json and summary_<attempt>.csv (per arm and
scheme; no subject or pair id, no position).  Standard output and error carry no subject id (redact()).
Exit codes (C24.12-5): 0 every cell of the attempt complete; 1 some cells failed; 2 refused before any field; 3 subject_sealed; 4 stopped; 5 an exception outside the
cell loop (the lock may be written; what was written is kept; to the lead).
"""
import os

if __name__ == "__main__":
    # as bayesreg.eval.holdout: command-line defaults only, before numba is imported; importing this module changes no environment variable
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

from dataclasses import dataclass                                          # noqa: E402
import csv                                                                 # noqa: E402
import importlib                                                           # noqa: E402
import io                                                                  # noqa: E402
import json                                                                # noqa: E402
from pathlib import Path                                                   # noqa: E402
import re                                                                  # noqa: E402
import resource                                                            # noqa: E402
import sys                                                                 # noqa: E402
import tempfile                                                            # noqa: E402
import time                                                                # noqa: E402

import numpy as np                                                         # noqa: E402

from jd_curl.three_d.nifti import file_sha256                              # noqa: E402

from bayesreg.io import code_identity as CI                                # noqa: E402
from bayesreg.io.guard import SubjectSealed, is_hex64                      # noqa: E402
from bayesreg.io.sealed import SealedListError                             # noqa: E402

from . import holdout as H                                                 # noqa: E402
from . import holdout_decide as HD                                         # noqa: E402
from .fields import FieldError, array_sha256, d_to_u, describe_run, load_field      # noqa: E402

SCHEMA_LOCK = "bayesreg.holdout.tet24.lock.v1"
SCHEMA_RUN = "bayesreg.holdout.tet24.run.v1"
SCHEMA_PUBLIC = "bayesreg.holdout.tet24.run_public.v1"
SCHEMA_CELL = "bayesreg.holdout.tet24.cell.v1"          # named in the run record; a cell file holds numbers and booleans only
SCHEMA_SUMMARY = "bayesreg.holdout.tet24.summary.v1"
SCHEMA_DRY = "bayesreg.holdout.tet24.dry_run.v1"
SCHEMA_RERUN = "bayesreg.holdout.tet24.rerun.v1"
MODULE_REL = "bayesreg/eval/holdout_tet24.py"            # this file: the one code change after P1 the identity allows (C24.12-2)
MAIN_REF = "refs/heads/main"                             # the reviewed commit is an ancestor of this ref (C24.12-2; review minor 6; diff review minor 4: the full ref)
SCRIPTS_MODULE = "folding_metrics"                       # scripts/folding_metrics.py, imported by its flat name from this code tree (never copied)
SCHEMES = (24, 6, 5)                                     # tet_ratio_summary's default schemes (checked against its signature)
KEYS = ("min_ratio", "cells_nonpositive", "cells", "tets_nonpositive", "tets")     # tet_ratio_summary's numbers per scheme
T24_ROOT = H.HOLDOUT_RESULTS/"tet24_v1"                  # the new directory (C24.12-6); a constant, never derived from an option
STEP11_CHECKOUT = H.HOLDOUT_RESULTS/"code"               # the step-11 checkout (留出集解封与运行手册 §2): this module's code tree may not lie in it (C24.12-13)
LOCK_NAME = "tet24.lock"
CELLS_DIR = "cells"
OWN_UNITS = ("tag-hb-tet24", "tag-hb-tet24-dry")
RERUN_DIR_REL = "splits/bayesreg_v1"
RERUN_NAME = re.compile(r"holdout_B_v1_tet24_rerun_([1-9][0-9]{0,2})\.json", re.ASCII)  # k = 1..999, no leading zero (review important 4); rule one admits the name
RERUN_LOCK = "rerun_{k}.lock"
RERUN_KEYS = ("schema", "written_by", "words", "time_local", "utc", "reason", "reason_class", "lock_sha256", "cells")
RERUN_WRITERS = H.REEVAL_WRITERS                         # BZ or the lead on BZ's behalf (D109)
RERUN_REASONS = H.REEVAL_REASONS                         # the computation's own infrastructure failures only (killed, OOM, I/O)
SUBJECT_ID = re.compile(r"OASIS_OAS1_[0-9]{4}_MR[0-9]", re.ASCII)
EXIT = dict(done=0, failed=1, refused=2, sealed=3, stopped=4, error=5)


class PlanRefused(H.HoldoutRefused):
    """plan_cells' refusal (stage 'cells'): the full message names the cells (arm, position) and paths; `kinds` counts the problems by kind, for the dry run's
    report (diff review minor 3: no position, no path)."""

    def __init__(self, message, kinds):
        super().__init__("cells", message)
        self.kinds = dict(kinds)


class FieldMismatch(H.HoldoutRefused):
    """The loaded d, or u = d_to_u(d), is not what step 11 read and certified (d sha256, the BL1 certificate's state_hash): the run stops (exit 4), to the lead."""


STOPS = (H.ManifestChanged, H.SpyRefused, FieldMismatch)


def redact(text):
    """A message for standard output / error and the files without ids: every OASIS subject id replaced by <id> (the full message stays in the run record)."""
    return SUBJECT_ID.sub("<id>", str(text))


PATH_TEXT = re.compile(r"/[^\s'\",:;()\[\]{}]+")
POSITION_TEXT = re.compile(r"\bposition [0-9]+\b")


def scrub(text):
    """The dry run's report and output (C24.12-8; diff review minor 3): subject ids, absolute paths and positions replaced (<id>, <path>, position <n>)."""
    return POSITION_TEXT.sub("position <n>", PATH_TEXT.sub("<path>", redact(text)))


def public_refusal(exc):
    """A refusal for the dry run's report: the stage, the kind and (a plan) the counts by kind; the rest scrubbed."""
    if isinstance(exc, PlanRefused):
        return f"refused at cells: PlanRefused: {sum(exc.kinds.values())} cells cannot be planned; by kind {dict(sorted(exc.kinds.items()))}"
    stage = getattr(exc, "stage", None)
    return scrub(f"refused{' at '+stage if stage else ''}: {type(exc).__name__}: {exc}")


def root():
    return Path(T24_ROOT)


def write_atomic(path, text):
    """Written once and whole: a temporary file in the same directory, flushed, then os.link to the name (FileExistsError if it is there); the temporary file removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def write_json(path, document):
    return write_atomic(path, H.dumps(document)+"\n")


# ---- code identity (C24.12-2) ----------------------------------------------------------------------------------------------------------------------------
def _git_text(repo, *args):
    git, _ = CI.git_runner(repo)
    rc, out, err = git(*args)
    return rc, out.strip(), err


def reviewed_problems(repo, reviewed_commit, head_blob):
    """The reviewed commit (review minor 6; the lead's ruling (3)): given; a commit; an ancestor of MAIN_REF (or MAIN_REF itself); it changes MODULE_REL against its
    first parent (it is the commit that brought this module to main, not any later commit of main); its blob of MODULE_REL is HEAD's and MAIN_REF's."""
    if reviewed_commit is None:
        return [f"--reviewed-commit is required (the commit on {MAIN_REF} that brought {MODULE_REL})"], None, None
    problems = []
    rc_c, commit, err_c = _git_text(repo, "rev-parse", "--verify", "--quiet", f"{reviewed_commit}^{{commit}}")
    if rc_c or not commit:
        return [f"--reviewed-commit {reviewed_commit!r} is not a commit of this repository ({err_c.strip()})"], None, None
    rc_a, _, err_a = _git_text(repo, "merge-base", "--is-ancestor", commit, MAIN_REF)
    if rc_a:
        problems.append(f"--reviewed-commit {commit} is not an ancestor of {MAIN_REF} ({err_a.strip() or 'not an ancestor'})")
    rc_r, blob, err_r = _git_text(repo, "rev-parse", f"{commit}:{MODULE_REL}")
    if rc_r or blob != head_blob:
        problems.append(f"{MODULE_REL} at the reviewed commit {commit} ({blob if not rc_r else err_r.strip()}) is not HEAD's ({head_blob})")
    rc_p, parent_blob, _ = _git_text(repo, "rev-parse", f"{commit}^1:{MODULE_REL}")
    if not rc_r and not rc_p and parent_blob == blob:
        problems.append(f"the reviewed commit {commit} does not change {MODULE_REL} against its first parent: not the commit that brought it to {MAIN_REF}")
    rc_m, main_blob, err_m = _git_text(repo, "rev-parse", f"{MAIN_REF}:{MODULE_REL}")
    if rc_m or main_blob != blob:
        problems.append(f"{MODULE_REL} on {MAIN_REF} ({main_blob if not rc_m else err_m.strip()}) is not the reviewed one ({blob}): {MAIN_REF} changed it since")
    return problems, commit, blob


def identity(repo=None, *, reviewed_commit=None, run=None):
    """(record, problems): rule one (code_identity.unit_identity_problems) with the one exception of C24.12-2.  The problems rule one reports for this module
    -- its adding commit ("A <MODULE_REL>: the code changed after the pinned commit (dir)") and the working tree's code differing from P1 by it -- are excused
    only when exactly one commit of P1..HEAD touches MODULE_REL, adds it, and changes nothing else, and when the code differs from P1 by MODULE_REL alone (and the
    allowlist under exception B).  Every other problem stays.  The blob of MODULE_REL at HEAD must equal the working tree's and the reviewed commit's
    (reviewed_problems)."""
    repo = Path(H.REPO if repo is None else repo)
    record, problems = CI.unit_identity_problems(repo=repo, run=run)
    rows = record.get("commits") or []
    own, excused, prefixes = [], [], []
    touching = [(row, change) for row in rows for change in row.get("changes", []) if change.get("path") == MODULE_REL]
    expected = f"A {MODULE_REL}: the code changed after the pinned commit (dir)"
    module_commit = None
    if len(touching) != 1:
        own.append(f"{MODULE_REL}: exactly one commit of P1..HEAD must add it and nothing touch it after (found {len(touching)} changes of it)")
    else:
        row, change = touching[0]
        if change.get("status") != "A" or len(row["changes"]) != 1 or change.get("problem") != expected:
            own.append(f"{row['commit'][:12]}: the commit that touches {MODULE_REL} must only add it ({[(c.get('status'), c.get('path')) for c in row['changes']][:5]})")
        else:
            excused.append(f"{row['commit'][:12]}: {expected}")
            module_commit = row["commit"]
    differ = list(record.get("code_dirs_differ") or [])
    rest = [p for p in differ if p != MODULE_REL]
    allowed = any(c.get("path") == CI.ALLOWLIST_REL and c.get("problem") is None for row in rows for c in row.get("changes", []))
    if MODULE_REL in differ and (not rest or (allowed and rest == [CI.ALLOWLIST_REL])):
        prefixes.append(f"the working tree's code directories differ from the pinned commit {record.get('pinned')}: {differ[:10]} (git diff: exit 0 ")
    elif MODULE_REL not in differ:
        own.append(f"{MODULE_REL} is not in the working tree's difference to P1 (code directories {differ[:10]})")
    remaining = list(problems)
    for text in excused:
        if text in remaining:
            remaining.remove(text)
    for prefix in prefixes:
        hit = next((p for p in remaining if p.startswith(prefix)), None)
        if hit is not None:
            remaining.remove(hit)
    rc_h, head_blob, _ = _git_text(repo, "rev-parse", f"HEAD:{MODULE_REL}")
    rc_w, work_blob, _ = _git_text(repo, "hash-object", "--", MODULE_REL)
    if rc_h or rc_w or not head_blob or head_blob != work_blob:
        own.append(f"{MODULE_REL}: its blob at HEAD ({head_blob if not rc_h else 'none'}) is not the working tree's ({work_blob if not rc_w else 'none'})")
    reviewed, reviewed_commit_full, reviewed_blob = reviewed_problems(repo, reviewed_commit, head_blob if not rc_h else None)
    own += reviewed
    record = dict(record, rule="rule one (C24.3-4) with the exception of C24.12-2: one commit of P1..HEAD only adds bayesreg/eval/holdout_tet24.py; the code is P1's but this file "
                               "(and the allowlist under exception B); the module is the reviewed commit's on main",
                  rule_one_problems=problems, excused=excused+prefixes,
                  module=dict(path=MODULE_REL, commit=module_commit, blob=head_blob, reviewed_commit=reviewed_commit_full or reviewed_commit, reviewed_blob=reviewed_blob,
                              main_ref=MAIN_REF))
    out = remaining+own
    record.update(passed=not out, problems=out)
    return record, out


def folding_module(repo=None):
    """scripts/folding_metrics.py of this code tree by its flat name (experiments/ convention), its origin checked; tet_ratio_summary's default schemes are SCHEMES."""
    import inspect
    folder = Path(H.REPO if repo is None else repo)/"scripts"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    module = importlib.import_module(SCRIPTS_MODULE)
    H.check_module_origin(module, folder)
    default = inspect.signature(module.tet_ratio_summary).parameters["schemes"].default
    if tuple(default) != SCHEMES:
        raise H.HoldoutRefused("code", f"folding_metrics.tet_ratio_summary's default schemes {default} are not {SCHEMES}")
    return module


def origins(repo=None):
    """bayesreg and folding_metrics imported from this code tree (the global rule; holdout.imported_from's first half, without the PQ helpers this entry does not use)."""
    import bayesreg
    repo = Path(H.REPO if repo is None else repo)
    if not H._inside(Path(bayesreg.__file__).parent, repo/"bayesreg"):
        raise H.HoldoutRefused("code", f"bayesreg was imported from {bayesreg.__file__}, not from {repo}")
    module = folding_module(repo)
    return dict(bayesreg=str(Path(bayesreg.__file__).resolve()), folding_metrics=str(Path(module.__file__).resolve()))


# ---- places (C24.12-6, -8, -13) --------------------------------------------------------------------------------------------------------------------------
def forbidden_places(table=None):
    """The roots nothing of this tool may lie under: the evaluation's watched roots (with a table), its eval directory, T24_ROOT, the step-11 checkout."""
    roots = [Path(H.EVAL_DIR), root(), Path(STEP11_CHECKOUT)]
    return roots+(H.watched_roots(table) if table is not None else [Path(H.DATA_ROOT)])


def check_root(table):
    """T24_ROOT: not under the code tree, the data root, a seq_vp_3d tree, any root the spy watches or the eval directory (the run could not write there), not an arm's
    or a reserved name."""
    out = root()
    watched = H.watched_roots(table)+[Path(H.EVAL_DIR)]
    names = {a["name"] for a in table["arms"]} | set(H.RESERVED_NAMES)
    if H._inside(out, H.REPO) or H._under_read_only(out) or H.forbidden_path(out) or out.name in names or any(H._inside(out, r) for r in watched):
        raise H.HoldoutRefused("output", f"{out}: not under the code tree, the data root, a seq_vp_3d tree or a watched root; not an arm's or a reserved name")
    return out


def check_code_place(repo, table):
    """The code tree (C24.12-13; the lead's ruling (3)): not under a watched root, the eval directory, T24_ROOT or the step-11 checkout."""
    hits = [str(r) for r in forbidden_places(table) if H._inside(repo, r)]
    if hits:
        raise H.HoldoutRefused("code", f"the code tree {repo} lies under {hits}: a new checkout elsewhere (C24.12-13)")
    return str(Path(repo).resolve())


def check_dry_out(out, table=None):
    """The dry run's --out-dir (review minor 7): new; not under the code tree, the data root, a seq_vp_3d tree, a watched root, the eval directory, T24_ROOT or the
    step-11 checkout.  HoldoutRefused('output') otherwise (dry_run returns 2; diff review minor 2)."""
    out = Path(out)
    if out.exists() or H._inside(out, H.REPO) or H._under_read_only(out) or H.forbidden_path(out) or any(H._inside(out, r) for r in forbidden_places(table)):
        raise H.HoldoutRefused("output", "--out-dir: a new directory outside the code tree, the data root, the seq_vp_3d trees, the watched roots, the eval directory, "
                                         "T24_ROOT and the step-11 checkout")
    return out


# ---- the plan of the cells (C24.12-3) ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Cell24:
    arm: str
    family: str
    kind: str
    index: int                  # the row index 0..49 of the pinned pair list (the position)
    pair: str
    outcome: str                # "output" | "no_output" | "not_evaluated" (no step-11 entry)
    path: str | None = None
    file_sha256: str | None = None
    d_sha256: str | None = None
    state_hash: str | None = None
    bl1_passed: bool | None = None


CERTIFICATE_KEYS = ("mode", "configuration", "state_hash", "passed")      # of the step-11 certificate (diff review minor 1: not its cells, counted on the truth brain mask)


def slim_records(records):
    """Of holdout_decide.collect's merged records only what C24.12-3 (g) takes: the grid and, per arm, outcome, field and of the certificate CERTIFICATE_KEYS (the
    lead's ruling (1): the numbers computed from truth -- the certificate's cell counts on the truth brain mask among them -- are not kept, written or printed)."""
    def entry(e):
        cert = e.get("certificate")
        return dict(outcome=e.get("outcome"), field=e.get("field"),
                    certificate=None if not isinstance(cert, dict) else {k: cert.get(k) for k in CERTIFICATE_KEYS})
    return {name: dict(grid=r.get("grid"), arms={a: entry(e) for a, e in (r.get("arms") or {}).items() if isinstance(e, dict)})
            for name, r in records.items() if isinstance(r, dict)}


def field_place_ok(arm, row, path):
    """The step-11 record's field path is where the arm table puts the arm's field (holdout.classify_cell's rule): <field.dir>/disp.npz, or a disp.npz under the
    field root (the terminal record named it); never under the data root or a seq_vp_3d tree."""
    if H.forbidden_path(path):
        return False
    field = arm["field"]
    if "dir" in field:
        return path == str(Path(H.fill(field["dir"], arm, row))/"disp.npz")
    froot = H.template_root(field["root"].replace("{arm}", arm["name"]))
    return Path(path).name == "disp.npz" and H._inside(path, froot) and Path(path).resolve() != froot.resolve()


def plan_cells(table, rows, records, listed):
    """Every (arm, pair) cell in pair order, arm-table order within a pair: from the step-11 records (slim_records).  A cell with output must name a disp.npz at the
    arm's field place, listed in the manifest at the sha256 the record names, its d sha256, and a BL1 Jacobian-only certificate with a state_hash.
    HoldoutRefused('cells') listing the cells (by arm and position) that do not."""
    cells, problems, kinds, seen = [], [], {}, {}

    def problem(kind, text):
        problems.append(text)
        kinds[kind] = kinds.get(kind, 0)+1
    for row in rows:
        record = records.get(row[3])
        for arm in table["arms"]:
            base = dict(arm=arm["name"], family=arm["family"], kind=arm["kind"], index=row[0], pair=row[3])
            entry = None if record is None else (record.get("arms") or {}).get(arm["name"])
            if entry is None:
                cells.append(Cell24(**base, outcome="not_evaluated"))
                continue
            if entry.get("outcome") != "output":
                cells.append(Cell24(**base, outcome="no_output"))
                continue
            where = f"{arm['name']} position {row[0]}"
            field = entry.get("field") or {}
            disp = field.get("disp") or {}
            cert = entry.get("certificate") or {}
            path = disp.get("path")
            if field.get("kind") != arm["kind"] or field.get("source") != "disp.npz" or not isinstance(path, str) or not path.startswith("/"):
                problem("not_a_disp_npz", f"{where}: the step-11 field is not a disp.npz of kind {arm['kind']}")
            elif not field_place_ok(arm, row, path):
                problem("field_place", f"{where}: {path} is not the arm's field place (arm table) or lies under the data root or a seq_vp_3d tree")
            elif listed.get(path) is None or listed[path] != disp.get("file_sha256"):
                problem("not_listed_at_sha256", f"{where}: {path} is not a line of the manifest at the sha256 step 11 read ({disp.get('file_sha256')})")
            elif not (is_hex64(disp.get("d_sha256")) and is_hex64(cert.get("state_hash"))):
                problem("no_d_sha256_or_state_hash", f"{where}: the step-11 record has no d sha256 or BL1 state_hash")
            elif (cert.get("mode"), cert.get("configuration")) != H.ARM_CERTIFICATE:
                problem("not_bl1", f"{where}: the step-11 certificate is not {H.ARM_CERTIFICATE}")
            elif path in seen:
                problem("shared_file", f"{where}: {path} is also the field of {seen[path]}")
            else:
                seen[path] = where
                cells.append(Cell24(**base, outcome="output", path=path, file_sha256=disp["file_sha256"], d_sha256=disp["d_sha256"], state_hash=cert["state_hash"],
                                    bl1_passed=cert.get("passed") is True))
    if problems:
        raise PlanRefused(f"{len(problems)} cells cannot be planned: {problems[:10]}{' ...' if len(problems) > 10 else ''}", kinds)
    return cells


def record_files(outs):
    """The per-pair record files holdout_decide.collect reads (its glob), for the spy's allowance."""
    return [p for out in outs for p in (sorted((Path(out)/"pairs").glob("*.json")) if (Path(out)/"pairs").is_dir() else [])]


# ---- stage 1: the admission (no field read) ------------------------------------------------------------------------------------------------------------
def admit(*, repo=None, reviewed_commit=None, run=None, cgroup=H.PROC_CGROUP, own_units=OWN_UNITS):
    """The admission of C24.12-3 in its order, under the spy started here first (holdout.start_spy: the data root watched from the start, the sealed lists read by the
    guard recorded): the code identity, then _admit.  Returns the context; on a refusal the spy is stopped.  Nothing of a field is read; no lock is taken."""
    repo = Path(H.REPO if repo is None else repo)
    H.stop_spy()
    spy = H.start_spy([H.DATA_ROOT])
    try:
        code, problems = identity(repo, reviewed_commit=reviewed_commit)
        if problems:
            raise H.HoldoutRefused("code_identity", f"rule one with the exception of C24.12-2 does not hold: {problems[:5]}; nothing is read")
        return _admit(repo, spy, code, run=run, cgroup=cgroup, own_units=own_units)
    except BaseException:
        H.stop_spy()
        raise


def _admit(repo, spy, code, *, run, cgroup, own_units, results=True):
    """After the identity: the module origins, SG (holdout.admit_holdout), the pins, the spy's roots, the code tree's place, T24_ROOT, the machine gate, the manifest
    -- all in the code tree --; then (results) the lock lines, the step-11 records, the plan, the spy's allowance, the stage manifests.  results False (the dry run
    after a refused identity, C24.12-8): returns after the manifest, before anything of the results root is opened."""
    imported = origins(repo)
    unseal, bound, pairs, admitted = H.admit_holdout(repo=repo)                   # SG: load_unseal, bind_unseal, check_subject (C24.12-3 (c))
    prereg_raw, prereg = H.check_prereg(repo)
    table, table_record = H.load_arm_table(prereg_raw, repo)
    spy.watch(H.watched_roots(table)+[Path(H.EVAL_DIR)])
    place = check_code_place(repo, table)
    out = check_root(table)
    machine = H.machine_gate(run=run, cgroup=cgroup, own_units=own_units)
    raw, digest, git = H.read_committed(H.MANIFEST_REL, "manifest", repo)
    header, listed = H.parse_manifest(raw)
    want = dict(prereg_sha256=prereg["sha256"], arm_table_sha256=table_record["sha256"], unseal_sha256=unseal.sha256)
    if {k: header.get(k) for k in H.MANIFEST_HEADER} != want:
        raise H.HoldoutRefused("manifest", f"{H.MANIFEST_REL}: header {header} is not {want}")
    context = dict(repo=repo, spy=spy, code=code, code_place=place, imported=imported, unseal=unseal, pairs=pairs, admitted=admitted, prereg=prereg, table=table,
                   table_record=table_record, machine=machine, manifest=dict(path=H.MANIFEST_REL, sha256=digest, git=git, n_inputs=len(listed), header=header),
                   listed=listed, lines=[], outs=[], record_files=[], records={}, cells=[], grids={}, allowed=[], out=out, results=False)
    if not results:
        return context
    lines = H.run_lines()                                                         # the master lock and the reevaluation locks (C24.12-3 (f))
    if not lines:
        raise H.HoldoutRefused("eval", f"{H.MASTER_LOCK}: no master lock; nothing was evaluated")
    for path, line in lines:
        if (line.get("manifest_sha256"), line.get("arm_table_sha256"), line.get("prereg_sha256")) != (digest, table_record["sha256"], prereg["sha256"]):
            raise H.HoldoutRefused("eval", f"{path}: its manifest, arm table or preregistration is not the committed one ({digest}, {table_record['sha256']}, {prereg['sha256']})")
    if lines[0][1].get("pinned_commit") != code.get("pinned"):
        raise H.HoldoutRefused("eval", f"the master lock's pinned commit {lines[0][1].get('pinned_commit')} is not P1 of runs/holdout_B_v1/pinned_code.json ({code.get('pinned')})")
    outs = [line["out_dir"] for _, line in lines if line.get("out_dir")]
    files = record_files(outs)
    try:
        merged = HD.collect(outs, pairs)                                          # holdout_decide's reader of the step-11 records (C24.12-3 (g))
    except HD.DecisionRefused as exc:
        raise H.HoldoutRefused("eval", f"the step-11 records: {exc}") from None
    records = slim_records(merged)                                                # only outcome, field, certificate and grid kept (the lead's ruling (1))
    del merged
    cells = plan_cells(table, pairs, records, listed)
    stage = Path(table["stage_root"])
    allowed = sorted({*(c.path for c in cells if c.outcome == "output"), *(str(stage/row[3]/"manifest.json") for row in pairs), *(str(p) for p, _ in lines),
                      *(str(p) for p in files)})
    spy.allow(allowed)                                                            # from here an open under a watched root outside these is refused at the open
    grids = H.check_stage(table, pairs, listed)                                   # the stage manifests, hashed first; the grid and the direction (C24.12-3 (j))
    for row in pairs:
        theirs = (records.get(row[3]) or {}).get("grid")
        if theirs is not None and theirs != grids[row[3]]:
            raise H.HoldoutRefused("pairs", f"position {row[0]}: the stage grid {grids[row[3]]} is not the step-11 record's {theirs}")
    context.update(lines=lines, outs=outs, record_files=[str(p) for p in files], records={n: dict(grid=r.get("grid")) for n, r in records.items()}, cells=cells,
                   grids=grids, allowed=allowed, results=True)
    return context


def spy_counts(spy, context):
    """What the spy saw, by class (C24.12-7): field, stage_manifest, eval_lock, eval_record; truth (an open under the data root or of a label file) and outside (any
    other open under a watched root) must both be 0, and so must the refusals.  Sealed-list reads (the guard's) are counted apart."""
    import os.path as op
    data = {op.realpath(str(H.DATA_ROOT)), op.abspath(str(H.DATA_ROOT))}
    groups = (("field", {op.realpath(c.path) for c in context["cells"] if c.outcome == "output"}),
              ("stage_manifest", {op.realpath(str(Path(context["table"]["stage_root"])/row[3]/"manifest.json")) for row in context["pairs"]} if context["results"] else set()),
              ("eval_lock", {op.realpath(str(p)) for p, _ in context["lines"]}), ("eval_record", {op.realpath(p) for p in context["record_files"]}))

    def kind(path):
        if any(path == d or path.startswith(d+os.sep) for d in data) or "aligned_seg" in path:
            return "truth"
        return next((name for name, group in groups if path in group), "outside")
    counts = {k: 0 for k in ("field", "stage_manifest", "eval_lock", "eval_record", "truth", "outside")}
    for event in spy.events:
        if event["kind"] != "sealed_list":
            counts[kind(event["path"])] += 1
    counts.update(truth_kind_events=sum(e["kind"] == "truth" for e in spy.events), refused=len(spy.refused),
                  sealed_list_reads=sum(e["kind"] == "sealed_list" for e in spy.events),
                  distinct_fields=len({e["path"] for e in spy.events if kind(e["path"]) == "field"}))
    return counts


def spy_problem(counts):
    """The stop of C24.12-5 after the loop: any refusal, truth or outside open."""
    bad = {k: counts[k] for k in ("refused", "truth", "outside", "truth_kind_events") if counts.get(k)}
    return f"the spy's counts {bad} (refused, truth and outside must all be 0)" if bad else None


# ---- the cell (C24.12-4) ---------------------------------------------------------------------------------------------------------------------------------
def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20         # Linux: KiB


def compute_cell(cell, metadata, listed, fm):
    """One cell: the field hashed against the manifest, fields.describe_run and load_field with the stage grid's shape and spacing (holdout.score_cell's call), hashed
    again; d's sha256 the step-11 record's; u = fields.d_to_u(d, spacing) (holdout.score_d's call) and sha256(u) the BL1 certificate's state_hash; then
    folding_metrics.tet_ratio_summary(u, spacing) as it is.  Returns the cell document (numbers and booleans only).  FieldMismatch / ManifestChanged stop the run."""
    path = Path(cell.path)
    shape, spacing = tuple(metadata["shape_zyx"]), tuple(metadata["spacing_xyz"])
    wall, cpu = time.perf_counter(), time.process_time()
    H.verify_files([path], listed)
    info = describe_run(path.parent, pair=cell.pair, kind=cell.kind)
    d, _, field = load_field(info, shape, spacing)
    H.verify_files([path], listed)
    disp = field.get("disp") or {}
    if disp.get("file_sha256") != listed[str(path)] or disp.get("d_sha256") != cell.d_sha256:
        raise FieldMismatch("field", f"{cell.arm} position {cell.index}: d is not the d step 11 read (file {disp.get('file_sha256')}, d {disp.get('d_sha256')} vs {cell.d_sha256})")
    u = d_to_u(d, spacing)
    del d
    if array_sha256(u) != cell.state_hash:
        raise FieldMismatch("field", f"{cell.arm} position {cell.index}: sha256(u) is not the BL1 certificate's state_hash {cell.state_hash}")
    load_seconds = time.perf_counter()-wall
    tick = time.perf_counter()
    summary = fm.tet_ratio_summary(u, spacing)
    tet_seconds = time.perf_counter()-tick
    if sorted(summary) != sorted(f"tet{s}" for s in SCHEMES) or any(sorted(v) != sorted(KEYS) for v in summary.values()):
        raise RuntimeError(f"tet_ratio_summary returned {sorted(summary)}: not the schemes {SCHEMES} with {KEYS}")
    return dict(position=int(cell.index), tet_ratio_summary=summary, bl1_passed=bool(cell.bl1_passed),
                checks=dict(field_sha256_before=True, field_sha256_after=True, d_equals_step11=True, u_equals_bl1_u=True),
                seconds=dict(load=load_seconds, tet=tet_seconds, wall=time.perf_counter()-wall, cpu=time.process_time()-cpu), peak_rss_gib=peak_rss_gib())


def cell_path(out, arm, index, suffix=""):
    return Path(out)/CELLS_DIR/arm/f"{index:02d}{suffix}.json"


def cell_done(out, cell):
    return cell_path(out, cell.arm, cell.index).is_file()


def stop_records(out=None):
    out = root() if out is None else Path(out)
    return sorted(out.glob("stopped_*.json")) if out.is_dir() else []


# ---- the rerun authorization (C24.12-5; D109's way) ------------------------------------------------------------------------------------------------------
def load_rerun(path, cells, *, repo=None):
    """A committed authorization splits/bayesreg_v1/holdout_B_v1_tet24_rerun_<k>.json (k = 1..999), unchanged against HEAD: exactly RERUN_KEYS, schema SCHEMA_RERUN,
    written by BZ or the lead on BZ's behalf with words, time, a reason of RERUN_REASONS, the first run's lock sha256, and cells [[arm, position], ...]: output cells
    of the plan, distinct, without a complete result, named by no other authorization or rerun lock (at most once per cell).  Refused whenever a run stopped (the
    cells left are missing; BZ decides, the lead's ruling (4)).  Returns (k, [Cell24], record); HoldoutRefused('rerun')."""
    repo = Path(H.REPO if repo is None else repo)
    target = Path(path).resolve()
    match = RERUN_NAME.fullmatch(target.name)
    if target.parent != (repo/RERUN_DIR_REL).resolve() or not match:
        raise H.HoldoutRefused("rerun", f"{path}: a rerun authorization is {RERUN_DIR_REL}/holdout_B_v1_tet24_rerun_<k>.json, k = 1..999 without a leading zero")
    stops = stop_records()
    if stops:
        raise H.HoldoutRefused("rerun", f"a run stopped ({[p.name for p in stops]}): the cells left are missing and BZ decides (C24.12-5); no rerun authorization")
    rel = target.relative_to(repo.resolve()).as_posix()
    raw, digest, git = H.read_committed(rel, "rerun", repo)
    document = H.strict_json(raw, rel)
    if not isinstance(document, dict) or sorted(document) != sorted(RERUN_KEYS) or document["schema"] != SCHEMA_RERUN:
        raise H.HoldoutRefused("rerun", f"{rel}: exactly the keys {list(RERUN_KEYS)} and schema {SCHEMA_RERUN}")
    problems = []
    if document["written_by"] not in RERUN_WRITERS:
        problems.append(f"written_by {document['written_by']!r}: one of {RERUN_WRITERS}")
    for key in ("words", "time_local", "reason"):
        if not isinstance(document[key], str) or not document[key].strip():
            problems.append(f"{key}: non-empty")
    if document["reason_class"] not in RERUN_REASONS:
        problems.append(f"reason_class {document['reason_class']!r}: one of {RERUN_REASONS} (the computation's own infrastructure failures only)")
    problems += [p for p in (H.utc_problem(document["utc"], "utc"),) if p]
    lock = root()/LOCK_NAME
    if not lock.is_file():
        raise H.HoldoutRefused("rerun", f"{lock}: there is no first run; a rerun follows it")
    if document["lock_sha256"] != file_sha256(lock):
        problems.append(f"lock_sha256 is not the sha256 of {lock}")
    by_key = {(c.arm, c.index): c for c in cells if c.outcome == "output"}
    asked = document["cells"]
    if not (isinstance(asked, list) and asked and all(isinstance(a, list) and len(a) == 2 and isinstance(a[0], str) and isinstance(a[1], int) and not isinstance(a[1], bool)
                                                      for a in asked) and len({tuple(a) for a in asked}) == len(asked)):
        problems.append("cells: a non-empty list of distinct [arm, position]")
        asked = []
    earlier = earlier_rerun_cells(int(match.group(1)), repo)
    chosen = []
    for arm, index in asked:
        cell = by_key.get((arm, index))
        if cell is None:
            problems.append(f"[{arm}, {index}] is not an output cell of the plan")
        elif cell_done(root(), cell):
            problems.append(f"[{arm}, {index}] has a complete result")
        elif (arm, index) in earlier:
            problems.append(f"[{arm}, {index}] is named by {earlier[(arm, index)]} (at most once per cell, an earlier refused authorization included)")
        else:
            chosen.append(cell)
    if problems:
        raise H.HoldoutRefused("rerun", f"{rel}: {problems}")
    return int(match.group(1)), chosen, dict(path=rel, sha256=digest, git=git, reason_class=document["reason_class"])


def earlier_rerun_cells(k, repo):
    """{(arm, position): where} of every cell named by another committed rerun authorization (any number but k; each committed and unchanged) or by a rerun lock's
    line (holdout.earlier_reeval_pairs' rule: a cell is computed again at most once, an authorization that was refused counts too)."""
    out = {}
    folder = Path(repo)/RERUN_DIR_REL
    for path in sorted(folder.glob("holdout_B_v1_tet24_rerun_*.json")) if folder.is_dir() else []:
        match = RERUN_NAME.fullmatch(path.name)
        if not match or int(match.group(1)) == k:
            continue
        raw, _, _ = H.read_committed(path.relative_to(repo).as_posix(), "rerun", repo)
        document = H.strict_json(raw, path.name)
        for item in (document.get("cells") if isinstance(document, dict) else None) or []:
            if isinstance(item, list) and len(item) == 2:
                out.setdefault((item[0], item[1]), path.name)
    for path in sorted(root().glob("rerun_*.lock")) if root().is_dir() else []:
        for line in H.lock_lines(path):
            for item in line.get("cells") or []:
                out.setdefault((item[0], item[1]), path.name)
    return out


# ---- the run ------------------------------------------------------------------------------------------------------------------------------------------
def lock_line(context, attempt, rerun=None, cells=None):
    code = context["code"]
    return dict(schema=SCHEMA_LOCK, attempt=attempt, manifest_sha256=context["manifest"]["sha256"], arm_table_sha256=context["table_record"]["sha256"],
                prereg_sha256=context["prereg"]["sha256"], unseal_sha256=context["unseal"].sha256, master_lock_sha256=file_sha256(H.MASTER_LOCK),
                pinned_commit=code.get("pinned"), head=code.get("head"), module_blob=code["module"]["blob"], reviewed_commit=code["module"]["reviewed_commit"],
                out_dir=str(context["out"]), rerun=rerun, cells=cells, utc=H.utc(), pid=os.getpid())


def provenance(cells):
    return [dict(arm=c.arm, position=c.index, outcome=c.outcome, path=c.path, file_sha256=c.file_sha256, d_sha256=c.d_sha256, bl1_state_hash=c.state_hash,
                 bl1_passed=c.bl1_passed) for c in cells]


def compute(context, *, rerun_path=None, echo=print):
    """Stage 2: folding_metrics checked, then the lock (first run) or a committed rerun authorization and its lock, the start record, then every planned cell in order.
    Returns the exit code: 0 every cell of this attempt complete, 1 some failed (their failure records kept; a rerun needs an authorization), 4 stopped (a stop of
    a cell, a refusal of the spy, also one the cell's code swallowed, or after the loop any refusal, truth or outside open: no failure record, the cells left are
    missing).  A refusal of folding_metrics, the lock or the authorization raises HoldoutRefused before any field (the spy stopped)."""
    out, cells, listed, spy = context["out"], context["cells"], context["listed"], context["spy"]
    try:
        fm = folding_module(context["repo"])                                      # before the lock (review minor 3)
        if rerun_path is None:
            attempt, todo, rerun = 0, [c for c in cells if c.outcome == "output"], None
            lock = H.take_lock(out/LOCK_NAME, lock_line(context, 0))
        else:
            attempt, todo, rerun = load_rerun(rerun_path, cells, repo=context["repo"])
            lock = H.take_lock(out/RERUN_LOCK.format(k=attempt), lock_line(context, attempt, rerun, [[c.arm, c.index] for c in todo]))
        started = dict(schema=SCHEMA_RUN, attempt=attempt, started_utc=H.utc(), lock=lock, cell_schema=SCHEMA_CELL, n_todo=len(todo), cells=provenance(cells))
        write_json(out/f"run_{attempt}.started.json", started)
    except BaseException:
        if spy is not None and spy is H._SPY:
            H.stop_spy()
        raise
    done, failed, stopped, stop_at, stop_kind = [], [], None, None, None
    try:
        for n, cell in enumerate(todo):
            before = len(spy.refused) if spy is not None else 0
            error = None
            try:
                document = compute_cell(cell, context["grids"][cell.pair], listed, fm)
            except STOPS as exc:
                stopped, stop_kind = f"{type(exc).__name__}: {exc}", type(exc).__name__
            except Exception as exc:                                          # noqa: BLE001 - kept in its failure record unless the spy refused meanwhile
                import traceback
                error = (exc, traceback.format_exc())
            if stopped is None and spy is not None and len(spy.refused) > before:      # a refusal the cell's code swallowed (review important 2)
                stopped, stop_kind = f"SpyRefused (during the cell, swallowed by its code): {spy.refused[before:]}", "SpyRefused_swallowed"
            if stopped is not None:
                stop_at = [cell.arm, cell.index]
                echo(redact(f"stopped at {cell.arm} position {cell.index}: {stopped}"))
                break
            if error is not None:
                exc, trace = error
                write_json(cell_path(out, cell.arm, cell.index, f".failed_{attempt}"), dict(error=f"{type(exc).__name__}: {exc}", traceback=trace, utc=H.utc()))
                failed.append([cell.arm, cell.index])
                echo(redact(f"failed: {cell.arm} position {cell.index}: {type(exc).__name__}: {exc}"))
                continue
            write_json(cell_path(out, cell.arm, cell.index), document)
            done.append([cell.arm, cell.index])
            if (n+1) % 16 == 0 or n+1 == len(todo):
                echo(f"{n+1}/{len(todo)} cells; last {cell.arm} {document['seconds']['wall']:.1f} s; peak RSS {document['peak_rss_gib']:.2f} GiB")
    finally:
        if spy is not None and spy is H._SPY:
            H.stop_spy()
    counts = None if spy is None else spy_counts(spy, context)
    if stopped is None and counts is not None and spy_problem(counts):           # after the loop: any refusal, truth or outside open is a stop
        stopped, stop_kind = spy_problem(counts), "spy_counts"
    finished = {tuple(x) for x in done+failed}
    missing = [[c.arm, c.index] for c in todo if (c.arm, c.index) not in finished] if stopped else []
    if stopped is not None:
        write_json(out/f"stopped_{attempt}.json", dict(at=stop_at, kind=stop_kind, reason=stopped, missing=missing, utc=H.utc(),
                                                       rule="the cells left are missing; BZ decides (C24.12-5); no rerun authorization is accepted"))
    code = EXIT["stopped"] if stopped else (EXIT["failed"] if failed else EXIT["done"])
    summary = summarize(context["table"], cells, out, stopped=bool(stop_records(out)))
    write_json(out/f"summary_{attempt}.json", summary)
    write_csv(out/f"summary_{attempt}.csv", summary)
    record = dict(started, ended_utc=H.utc(), done=done, failed=failed, stopped=stopped, stop_kind=stop_kind, stop_at=stop_at, missing_after_stop=missing, exit_code=code,
                  code=context["code"],
                  code_place=context["code_place"], imported=context["imported"], admission=context["admitted"], prereg=context["prereg"], arm_table=context["table_record"],
                  machine=context["machine"], manifest=context["manifest"], eval_locks=[str(p) for p, _ in context["lines"]], eval_outs=context["outs"],
                  spy=None if spy is None else spy.record(), spy_counts=counts, summary=f"summary_{attempt}.json")
    write_json(out/f"run_{attempt}.json", record)
    write_json(out/f"run_{attempt}.public.json", public_record(record, todo))
    echo(f"attempt {attempt}: done {len(done)}, failed {len(failed)}, stopped {stopped is not None}, missing after a stop {len(missing)}; spy counts {counts}; exit {code}")
    return code


def public_record(record, todo):
    """run_<k>.public.json (review minor 2): what the runner and the reviewers read -- no subject id, pair name, position or path."""
    per_arm = {}
    for c in todo:
        row = per_arm.setdefault(c.arm, dict(todo=0, done=0, failed=0, missing_after_stop=0))
        row["todo"] += 1
    for key in ("done", "failed", "missing_after_stop"):
        for arm, _ in record[key]:
            per_arm[arm][key] += 1
    code = record["code"]
    return dict(schema=SCHEMA_PUBLIC, attempt=record["attempt"], started_utc=record["started_utc"], ended_utc=record["ended_utc"], exit_code=record["exit_code"],
                n_todo=record["n_todo"], done=len(record["done"]), failed=len(record["failed"]), missing_after_stop=len(record["missing_after_stop"]),
                stopped=record["stopped"] is not None, stop_kind=record["stop_kind"],
                per_arm=per_arm, spy_counts=record["spy_counts"], summary=record["summary"],
                code=dict(pinned=code.get("pinned"), head=code.get("head"), module_blob=(code.get("module") or {}).get("blob"),
                          reviewed_commit=(code.get("module") or {}).get("reviewed_commit")),
                manifest_sha256=record["manifest"]["sha256"])


# ---- the summary (C24.12-6) --------------------------------------------------------------------------------------------------------------------------
def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)), max=float(a.max())) if a.size else dict(mean=None, median=None, max=None)


def summarize(table, cells, out, *, stopped=False):
    """Per arm (arm-table order), over its cells with a complete result: for each scheme the fraction of cells with a ratio <= 0 and of tetrahedra <= 0 per pair
    (mean, median, max), min_ratio (min, median), the pairs with any cell <= 0; beside them (not compared) the pairs whose step-11 BL1 certificate did not pass.  No
    subject id, pair id or position."""
    arms = []
    for arm in table["arms"]:
        mine = [c for c in cells if c.arm == arm["name"]]
        output = [c for c in mine if c.outcome == "output"]
        docs = [json.loads(cell_path(out, c.arm, c.index).read_text(encoding="utf-8")) for c in output if cell_done(out, c)]
        schemes = {}
        for s in SCHEMES:
            rows = [d["tet_ratio_summary"][f"tet{s}"] for d in docs]
            ratios = [r["min_ratio"] for r in rows]
            schemes[f"tet{s}"] = dict(cells_nonpositive_fraction=_stats([r["cells_nonpositive"]/r["cells"] for r in rows]),
                                      tets_nonpositive_fraction=_stats([r["tets_nonpositive"]/r["tets"] for r in rows]),
                                      min_ratio=dict(min=float(min(ratios)) if ratios else None, median=float(np.median(ratios)) if ratios else None),
                                      pairs_with_nonpositive_cells=sum(r["cells_nonpositive"] > 0 for r in rows))
        arms.append(dict(arm=arm["name"], family=arm["family"], pairs=len(mine), output=len(output), no_output=sum(c.outcome == "no_output" for c in mine),
                         not_evaluated=sum(c.outcome == "not_evaluated" for c in mine), computed=len(docs), missing=len(output)-len(docs), schemes=schemes,
                         side_by_side=dict(bl1_not_passed_pairs=sum(c.bl1_passed is not True for c in output),
                                           tet24_pairs_with_nonpositive_cells=schemes["tet24"]["pairs_with_nonpositive_cells"],
                                           note="并列，不比较（C24.12-6）：BL1 证书判不通过的对数（第 11 步记录，全部有输出的格）与 tet24 有 ≤ 0 单元的对数（已算完的格）")))
    return dict(schema=SCHEMA_SUMMARY, role="解封后补算的预登记描述终点（24 四面体代理；BZ 2026-11-19 选 (乙)）：只描述，不进任何判定", schemes=list(SCHEMES),
                over="each arm's cells with output whose result is complete; no-output cells and cells not evaluated in step 11 are counted apart", stopped=stopped,
                missing_rule="after a stop (exit 4) the cells left are missing; BZ decides (C24.12-5)" if stopped else None, arms=arms, utc=H.utc())


CSV_COLUMNS = ("arm", "family", "scheme", "pairs", "output", "no_output", "computed", "missing", "cells_nonpositive_fraction_mean", "cells_nonpositive_fraction_median",
               "cells_nonpositive_fraction_max", "tets_nonpositive_fraction_mean", "tets_nonpositive_fraction_median", "tets_nonpositive_fraction_max", "min_ratio_min",
               "min_ratio_median", "pairs_with_nonpositive_cells", "bl1_not_passed_pairs")


def csv_rows(summary):
    out = []
    for arm in summary["arms"]:
        for s in SCHEMES:
            sc = arm["schemes"][f"tet{s}"]
            out.append(dict(arm=arm["arm"], family=arm["family"], scheme=f"tet{s}", pairs=arm["pairs"], output=arm["output"], no_output=arm["no_output"],
                            computed=arm["computed"], missing=arm["missing"],
                            **{f"cells_nonpositive_fraction_{k}": sc["cells_nonpositive_fraction"][k] for k in ("mean", "median", "max")},
                            **{f"tets_nonpositive_fraction_{k}": sc["tets_nonpositive_fraction"][k] for k in ("mean", "median", "max")},
                            min_ratio_min=sc["min_ratio"]["min"], min_ratio_median=sc["min_ratio"]["median"], pairs_with_nonpositive_cells=sc["pairs_with_nonpositive_cells"],
                            bl1_not_passed_pairs=arm["side_by_side"]["bl1_not_passed_pairs"]))
    return out


def write_csv(path, summary):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in csv_rows(summary):
        writer.writerow({k: (repr(v) if isinstance(v, float) else ("" if v is None else v)) for k, v in row.items()})
    return write_atomic(path, stream.getvalue())


# ---- the dry run --------------------------------------------------------------------------------------------------------------------------------------
def dry_run(out_dir, *, reviewed_commit=None, hash_fields=False, run=None, cgroup=H.PROC_CGROUP, echo=print):
    """The admission without the lock and without loading any field (C24.12-8).  The identity first: refused, it is recorded and only the code-tree part of the
    admission and the machine gate follow -- nothing under the results root or D: is opened (review important 1); passed, the whole admission follows (the lock
    lines, the step-11 records, the plan, the stage manifests, hashed), and per field its existence (a stat) and the manifest's sha256 equal to step 11's; with
    hash_fields also each field's sha256 now (its bytes read under the spy, nothing decoded).  Writes <out>/dry_run.json (counts and scrubbed conclusions only: no
    subject id, position or path; a refused plan by kind and count).  Returns 0 when nothing refused, 2 otherwise (a refused --out-dir too: nothing written)."""
    repo = Path(H.REPO)
    try:
        prereg_raw, _ = H.check_prereg(repo)
        table_for_out, _ = H.load_arm_table(prereg_raw, repo)                     # code-tree files only: the out-dir check needs the watched roots
    except H.HoldoutRefused:
        table_for_out = None
    try:
        out = check_dry_out(out_dir, table_for_out)
    except H.HoldoutRefused as exc:                                               # nothing is written (diff review minor 2: exit 2, not SystemExit's 1)
        echo(public_refusal(exc))
        return EXIT["refused"]
    report = dict(schema=SCHEMA_DRY, utc=H.utc(), reviewed_commit=reviewed_commit, field_loaded=False, fields_hashed=bool(hash_fields), conditions={})
    context = None
    H.stop_spy()
    spy = H.start_spy([H.DATA_ROOT])
    try:
        code, problems = identity(repo, reviewed_commit=reviewed_commit)
        report["conditions"]["code identity (C24.12-2)"] = "passed" if not problems else scrub(f"refused: {problems[:5]}")
        context = _admit(repo, spy, code, run=run, cgroup=cgroup, own_units=OWN_UNITS, results=not problems)
        if not problems:
            report["conditions"]["admission (C24.12-3)"] = "passed"
            per_arm = {}
            for c in context["cells"]:                                            # under the spy: a stat each, and with hash_fields the bytes of each listed field
                row = per_arm.setdefault(c.arm, dict(output=0, no_output=0, not_evaluated=0, exists=0, listed_sha_equals_step11=0, sha256_now_equals_listed=None,
                                                     bl1_not_passed=0))
                row[c.outcome] += 1
                if c.outcome == "output":
                    row["exists"] += Path(c.path).is_file()
                    row["listed_sha_equals_step11"] += context["listed"].get(c.path) == c.file_sha256
                    row["bl1_not_passed"] += c.bl1_passed is not True
                    if hash_fields:
                        row["sha256_now_equals_listed"] = (row["sha256_now_equals_listed"] or 0)+(file_sha256(c.path) == context["listed"][c.path])
            report.update(arms=per_arm, n_pairs=len(context["pairs"]), n_output=sum(c.outcome == "output" for c in context["cells"]), eval_runs=len(context["lines"]),
                          manifest_sha256=context["manifest"]["sha256"], out_root=str(context["out"]), out_root_exists=context["out"].exists())
        else:
            report["conditions"]["admission (C24.12-3)"] = ("code-tree part passed (SG, pins, code place, T24_ROOT, machine gate, manifest header); stopped before the "
                                                            "evaluation's lock lines and records: the code identity is refused (C24.12-8)")
    except (H.HoldoutRefused, SubjectSealed, SealedListError, OSError, ValueError) as exc:
        report["conditions"]["admission (C24.12-3)"] = public_refusal(exc)
    finally:
        H.stop_spy()
    if context is not None:
        report["spy_counts"] = spy_counts(spy, context)
    else:
        report["spy_events_outside_sealed_lists"] = sum(e["kind"] != "sealed_list" for e in spy.events)
    report["first_refusal"] = next((k for k, v in report["conditions"].items() if v != "passed"), None)
    out.mkdir(parents=True)
    write_json(out/"dry_run.json", report)
    for label, text in report["conditions"].items():
        echo(scrub(f"{label}: {text}"))
    for arm, row in (report.get("arms") or {}).items():
        echo(f"{arm}: {row}")
    if "spy_counts" in report:
        echo(f"spy: {report['spy_counts']}")
    echo(f"first refusal: {report['first_refusal']}; no field was loaded")
    return EXIT["done"] if report["first_refusal"] is None else EXIT["refused"]


# ---- the command line ------------------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    """Exit codes (C24.12-5): 0 done, 1 cells failed, 2 refused before any field, 3 subject_sealed, 4 stopped, 5 an exception outside the cell loop."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m bayesreg.eval.holdout_tet24", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dry-run", help="the admission, the plan, sha256 previews; no lock, no field loaded")
    d.add_argument("--out-dir", required=True, type=Path)
    d.add_argument("--reviewed-commit", required=True, help=f"the commit on {MAIN_REF} that brought this module (its blob must equal HEAD's)")
    d.add_argument("--hash-fields", action="store_true", help="also hash every planned field (bytes read, nothing decoded)")
    c = sub.add_parser("compute", help="the lock (or a rerun authorization), then every planned cell once")
    c.add_argument("--reviewed-commit", required=True)
    c.add_argument("--rerun", type=Path, help="a committed splits/bayesreg_v1/holdout_B_v1_tet24_rerun_<k>.json")
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        try:
            return dry_run(args.out_dir, reviewed_commit=args.reviewed_commit, hash_fields=args.hash_fields)
        except Exception as exc:                                                  # noqa: BLE001 - review minor 5: its own exit code
            print(redact(f"error: {type(exc).__name__}: {exc}"), file=sys.stderr)
            return EXIT["error"]
    try:
        context = admit(reviewed_commit=args.reviewed_commit)
    except SubjectSealed as exc:
        print(redact(f"subject_sealed: {exc}; no field was read"), file=sys.stderr)
        return EXIT["sealed"]
    except H.HoldoutRefused as exc:
        print(redact(f"refused at {exc.stage}: {exc}; no field was read"), file=sys.stderr)
        return EXIT["refused"]
    except (SealedListError, FieldError, OSError, ValueError) as exc:
        print(redact(f"refused: {type(exc).__name__}: {exc}; no field was read"), file=sys.stderr)
        return EXIT["refused"]
    except Exception as exc:                                                      # noqa: BLE001
        print(redact(f"error before any field: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return EXIT["error"]
    try:
        return compute(context, rerun_path=args.rerun)
    except H.HoldoutRefused as exc:                                               # folding_metrics, the lock or the rerun authorization: refused before any field
        print(redact(f"refused at {exc.stage}: {exc}; no field was read"), file=sys.stderr)
        return EXIT["refused"]
    except Exception as exc:                                                      # noqa: BLE001 - outside the cell loop (review minor 5)
        print(redact(f"error outside the cell loop: {type(exc).__name__}: {exc}; what was written is kept"), file=sys.stderr)
        return EXIT["error"]


__all__ = ["Cell24", "EXIT", "FieldMismatch", "PlanRefused", "admit", "compute", "compute_cell", "dry_run", "field_place_ok", "folding_module", "identity", "load_rerun", "main",
           "plan_cells", "public_record", "public_refusal", "redact", "reviewed_problems", "scrub", "slim_records", "spy_counts", "summarize", "write_atomic", "write_csv"]


if __name__ == "__main__":
    sys.exit(main())
