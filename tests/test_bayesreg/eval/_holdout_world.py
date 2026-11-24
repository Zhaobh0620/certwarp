"""A synthetic B-line hold-out for the tests of bayesreg.eval.holdout (tmp_path only: a temporary git repository, synthetic subjects OASIS_OAS1_09xx_MR1,
synthetic NIfTI labels, synthetic cells and PQ outputs).  No real subject file is opened.

SG (contract C23, merged in ecf4621) is used as it is: the temporary repository holds synthetic sealed lists (holdout_B_v1, H1, H4, H3), the pinned
allowlist, BZ's synthetic unseal record at the pinned path and its line in docs/decision_log.md, and bayesreg.io.sealed.REPO / SEALED_LISTS point into it
(monkeypatch only, as tests/test_bayesreg/sg/_sg_world.py does).  The real load_unseal, bind_unseal, check_subject and guard_sealed_lists run on it.

The arm table is the v3 form of the interface alignment (runs/20261109-align/接口对齐.md; runs/20261107-ev/实现说明.md §11):
  method   HB's layout: its cell cells/method/<pair>/ (terminal.json, output/{final,disp}.npz, cert.json, line.json, review/review.json finalized true), its
           queue records records/method/<pair>/ (step4.json: what a no_resume names), authorizations of tool hb;
  main     XB's layout: records records/<arm>/<pair>/ (result.json, overridden.json, resume/...), the field big/<arm>/<pair>/disp.npz (a resume:
           big/<arm>/<pair>.resume/) named with its sha256 by the terminal record, authorizations of tool xb;
  PQ       small files pq_small/<pair>/pq.json, big files pq_big/<pair>/{pq,baselines}.npz, the list pq_small/pq_outputs_manifest.txt, its release
           (runs/holdout_B_v1/pq_outputs_manifest.sha256 and pq_release.md) in one commit;
  records  holdout_common's resume / no_resume records (splits/bayesreg_v1/holdout_B_v1_<tool>_<kind>_<k>.json), holdout_common.AUTH_ROOT pointed here.
"""
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.nifti import file_sha256, save_scalar

import bayesreg.io.sealed as S
from bayesreg.eval import holdout as H
from bayesreg.eval.fields import array_sha256, read_disp
from bayesreg.io import unseal as U
from bayesreg.io.pair import subject_list_sha256
from bayesreg.seg.unet import synthetic

from . import _synthetic as syn


def hb_core():
    """experiments/hb/hb_core.py (HB's selection of BZ's decision, hb_core.bz_decision_auth: the finalized record's bz_decision; A5b review I2)."""
    import sys
    folder = str(Path(H.REPO)/"experiments"/"hb")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import hb_core as module
    return module

SUBJECTS = ("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "OASIS_OAS1_0903_MR1", "OASIS_OAS1_0904_MR1")
ROWS = ((SUBJECTS[0], SUBJECTS[1]), (SUBJECTS[2], SUBJECTS[3]))
H1_ID = "OASIS_OAS1_0905_MR1"            # a synthetic id on the synthetic H1 list
H4_ID = "OASIS_OAS1_0906_MR1"
DEV_ID = "OASIS_OAS1_0907_MR1"           # a synthetic development subject (on no sealed list)
MAIN = ("configc", "ants", "fga", "fsyn", "cadam")
WORDS = "（合成测试）解封记录你代我写；BZ 过目清单（合成）"
PQ_LIST = "pq_outputs_manifest.txt"
BASELINE_KEYS = ("h", "H_F", "H2", "V_F", "V2", "G_plus", "G_minus", "delta")     # experiments/pq/pq_core.BASELINE_KEYS (tested against it)
REPLAY_PASS = "逐位相同 19/19"
GIT_ENV = dict(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")


def git(repo, *args):
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)


def commit(repo, message="world"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)


def head(repo):
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def pair_name(fixed, moving):
    return H.PAIR_NAME.format(fixed=fixed, moving=moving)


# ---- the world ---------------------------------------------------------------------------------------------------------------------------------------------
def seg_pair(seed, lookup):
    seg35 = syn.labels(seed)
    return dict(seg35=seg35, seg4=lookup[seg35.astype(int)].astype(np.float64))


def method_classes():
    return dict(output=["hx_certified", "hx_fallback_configC", "hx_fallback_identity"], method_failure=[], unit_limit=["unit_limit"], wrapper_exception=[],
                implementation_error=["hx_implementation_error"], infrastructure=["infra_failure"], not_written=["hx_provenance_rejected"])


def arm_classes():
    return dict(output=["ok"], method_failure=["method_failed"], unit_limit=["unit_limit"], wrapper_exception=["wrapper_error"], implementation_error=[],
                infrastructure=["infra_failure"], not_written=["admission_refused", "code_not_clean", "list_error", "machine_refused", "needs_lead_rejudgment"])


def arm_table(root):
    """The method arm (HB's cell and records) and five main arms (XB's records and fields); every arm the BL1 Jacobian-only certificate (preregistration §3,
    D112 (a)); the method's review record and decision certificate; PQ's two roots."""
    method = dict(name="method", family="method", kind="disp", record_dir=str(root/"cells"/"method"/"{pair}"), field=dict(dir=str(root/"cells"/"method"/"{pair}"/"output")),
                  cert_mode="jacobian_only", jacobian_config="BL1", terminal=dict(files=["terminal.json"], status_key="line_status", classes=method_classes()),
                  complete_files=["terminal.json", "output/final.npz", "output/disp.npz", "output/cert.json", "output/line.json", "review/review.json"],
                  review_record=dict(file="review/review.json", status_key="finalized", final=[True]),
                  decision_certificate=dict(final="output/final.npz", cert="output/cert.json"), export_check=None,
                  stop_record="runs/holdout_B_v1/arm_stop_{arm}.json", authorization=dict(tool="hb", arm="method", records=str(root/"records"/"method"/"{pair}")))
    arms = [method]
    for name in MAIN:
        arms.append(dict(name=name, family="main", kind="disp", record_dir=str(root/"records"/"{arm}"/"{pair}"),
                         field=dict(file_key="disp.file", sha256_key="disp.sha256", root=str(root/"big"/"{arm}")), cert_mode="jacobian_only",
                         jacobian_config="BL1", terminal=dict(files=["resume/overridden.json", "resume/result.json", "overridden.json", "result.json"],
                                                              status_key="status", classes=arm_classes()),
                         complete_files=[], review_record=None, decision_certificate=None, export_check=dict(key="export_check.suspicious"),
                         stop_record="runs/holdout_B_v1/arm_stop_{arm}.json", authorization=dict(tool="xb", arm=name, records=str(root/"records"/"{arm}"/"{pair}"))))
    return dict(schema=H.SCHEMA_ARMS, stage_root=str(root/"stage"), main_family=list(MAIN), arms=arms,
                pq=dict(small_root=str(root/"pq_small"), big_root=str(root/"pq_big")), segmentation=None)


class World:
    """Everything of one synthetic hold-out; `patch(monkeypatch)` points the module's pins (and bayesreg.io.sealed's, holdout_common's) at it.
    subjects and pair_rows: the synthetic hold-out list and the pair list (SUBJECTS, ROWS); U5 (tests/test_bayesreg/pqh/test_pqh_to_entry.py) gives its own for the PQ world's pair."""
    subjects = SUBJECTS
    pair_rows = ROWS

    def __init__(self, tmp_path):
        self.tmp = Path(tmp_path)
        self.repo = self.tmp/"repo"
        self.data = self.tmp/"data"
        self.results = self.tmp/"results"
        self.lookup = H.seg4_map()
        self.grid = make_grid_3d(nx=syn.SHAPE[2], ny=syn.SHAPE[1], nz=syn.SHAPE[0], hx=1., hy=1., hz=1., unit="mm")
        self.labels = {s: seg_pair(10+k, self.lookup) for k, s in enumerate(self.subjects)}
        self.rows = tuple((i, f, m, pair_name(f, m)) for i, (f, m) in enumerate(self.pair_rows))
        self.reads = []
        self.bz_ii = None
        self.review_root = self.results/"holdout_B_v1"/"review"         # hb_core.review_root(): finalized_outputs.json
        self.synced = False
        self._build()

    # files ------------------------------------------------------------------------------------------------------------------------------------------
    def write_lists(self, *, h4=(H4_ID,)):
        repo = self.repo
        (repo/"lists").mkdir(exist_ok=True)
        (repo/"lists"/"b.txt").write_text("".join(f"{s}\n" for s in self.subjects), encoding="utf-8")
        (repo/"lists"/"h1.txt").write_text(f"{H1_ID}\n", encoding="utf-8")
        (repo/"lists"/"h4.txt").write_text("# H4 (synthetic)\n"+"".join(f"{s}\n" for s in h4), encoding="utf-8")
        (repo/"lists"/"h3.tsv").write_text("fixed\tmoving\nIBSR_01\tIBSR_02\n", encoding="utf-8")
        self.specs = (dict(name="holdout_B_v1", path="lists/b.txt", sha256=file_sha256(repo/"lists"/"b.txt"), columns=None, count=len(self.subjects)),
                      dict(name="H1", path="lists/h1.txt", sha256=file_sha256(repo/"lists"/"h1.txt"), columns=None, count=1),
                      dict(name="H4", path="lists/h4.txt", sha256=file_sha256(repo/"lists"/"h4.txt"), columns=None, count=len(h4)),
                      dict(name="H3", path="lists/h3.tsv", sha256=file_sha256(repo/"lists"/"h3.tsv"), columns=("fixed", "moving"), count=2))

    def write_record(self, **changes):
        """BZ's synthetic unseal record (written on BZ's behalf) and its line in docs/decision_log.md."""
        holdout = self.specs[0]
        document = dict(schema=U.UNSEAL_SCHEMA, written_by="lead on BZ's behalf", bz_words=WORDS, time_local="2026-11-07 6:00 AM CST", utc="2026-11-07T12:00:00Z",
                        sealed_list="holdout_B_v1", subjects_file=holdout["path"], subjects_file_sha256=holdout["sha256"],
                        subjects_sha256=subject_list_sha256(sorted(self.subjects)), n_subjects=len(self.subjects), allowlist_source="holdout_B_v1_synth",
                        allowlist_sha256=file_sha256(self.repo/H.ALLOWLIST_REL), segmenter_id="unet35_v4_ens", registry_sha256="a"*64,
                        purpose="synthetic test of the hold-out evaluation entry (C23.12)")
        document.update(changes)
        (self.repo/H.UNSEAL_REL).write_text(json.dumps(document, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
        sha = file_sha256(self.repo/H.UNSEAL_REL)
        log = self.repo/U.DECISION_LOG_REL
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"| 2026-11-07 | the unseal record {sha} | {document['bz_words']} |\n")
        return document

    def write_table(self, table=None):
        self.table = arm_table(self.tmp) if table is None else table
        (self.repo/H.ARM_TABLE_REL).write_text(json.dumps(self.table, indent=1), encoding="utf-8")
        self.table_sha = file_sha256(self.repo/H.ARM_TABLE_REL)
        (self.repo/H.PREREG_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/H.PREREG_REL).write_text(f"# synthetic preregistration\n\narm table sha256 {self.table_sha}\n", encoding="utf-8")
        self.prereg_sha = file_sha256(self.repo/H.PREREG_REL)

    def write_code_tree(self):
        """The code tree of the hold-out, not yet committed: a new git repository, the pair list, the sealed lists, the allowlist, BZ's unseal record and the arm table with the
        preregistration naming its sha256."""
        repo = self.repo
        (repo/"splits"/"bayesreg_v1").mkdir(parents=True)
        (repo/"configs"/"bayesreg").mkdir(parents=True)
        (repo/"configs"/"frozen").mkdir(parents=True)
        git(repo.parent, "init", "-q", str(repo))
        self.write_pairs(self.pair_rows)
        self.write_lists()
        allowlist = synthetic.allowlist_document([("holdout_B_v1_synth", list(self.subjects), True), ("dev", [DEV_ID], False)])
        (repo/H.ALLOWLIST_REL).write_text(json.dumps(allowlist), encoding="utf-8")
        self.write_record()
        self.write_table()

    def _build(self):
        repo = self.repo
        self.write_code_tree()
        for subject, arrays in self.labels.items():
            for kind in ("seg35", "seg4"):
                save_scalar(self.data/subject/H.LABEL_FILES[kind], arrays[kind], self.grid)
        for row in self.rows:
            self.write_stage(row)
            for arm in self.table["arms"]:
                self.write_cell(arm, row, status="hx_certified" if arm["family"] == "method" else "ok")
            self.write_pq(row)
        (self.tmp/"moves").mkdir()
        (self.tmp/"moves"/"move_0.json").write_text("{}", encoding="utf-8")
        self.write_input_lock()
        commit(repo)
        self.sync_method()
        self.release_pq()
        self.write_manifest()

    def write_pairs(self, rows):
        (self.repo/H.PAIRS_REL).write_text("fixed\tmoving\n"+"".join(f"{f}\t{m}\n" for f, m in rows), encoding="utf-8")

    def write_stage(self, row, *, reversed_=False):
        folder = self.tmp/"stage"/row[3]
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(row[0])
        np.savez_compressed(folder/"pair.npz", fixed=rng.random(syn.SHAPE), moving=rng.random(syn.SHAPE), initial_u=np.zeros(syn.SHAPE+(3,)))   # no seg key
        fixed, moving = (row[2], row[1]) if reversed_ else (row[1], row[2])
        manifest = dict(shape_zyx=list(syn.SHAPE), spacing_xyz=[1., 1., 1.], origin_xyz=[0., 0., 0.], unit="mm", pair_sha256=file_sha256(folder/"pair.npz"),
                        fixed_subject=fixed, moving_subject=moving, sources=dict(fixed=dict(sha256="a"*64), moving=dict(sha256="b"*64)))
        (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def field(self, arm_name, row):
        k = (MAIN.index(arm_name)+1) if arm_name in MAIN else 0
        return syn.smooth_d(amplitude=(1.2-.1*k, -.9+.05*row[0], .7), freq=1)

    def method_u(self, row):
        return syn.d_to_u(self.field("method", row))

    def cell(self, name, row):
        return self.tmp/"cells"/"method"/row[3] if name == "method" else self.tmp/"records"/name/row[3]

    def record_dir(self, name, row):
        return self.tmp/"records"/name/row[3]

    def write_cell(self, arm, row, *, status, extra=None, field=True, suspicious=False, resume=False, overridden=False):
        """method: HB's cell (terminal.json; output/ and review/review.json with an output); a main arm: XB's result.json (resume/: a resumed attempt;
        overridden.json: the unit's reading), the field on big/ named in it."""
        from bayesreg.solve.export import write_disp
        name = arm["name"]
        if arm["family"] == "method":
            base = self.cell(name, row)
            base.mkdir(parents=True, exist_ok=True)
            if field:
                out = base/"output"
                out.mkdir(parents=True, exist_ok=True)
                u = self.method_u(row)
                digest = array_sha256(u)
                write_disp(out/"disp.npz", u, self.grid, state_hash=digest)
                np.savez(out/"final.npz", u=u, state_hash=np.array(digest))
                (out/"cert.json").write_text(json.dumps(dict(status="certified_feasible", state_hash=digest)), encoding="utf-8")
                (out/"line.json").write_text(json.dumps(dict(line_status=status)), encoding="utf-8")
                self.write_review(row, finalized=True)
            (base/"terminal.json").write_text(json.dumps(dict(line_status=status, **(extra or {}))), encoding="utf-8")
            if status == "hx_implementation_error" and not field:          # HE's line.json of the cell names the case of C22.5-6 (hb_core.implementation_kind reads it)
                (base/"line.json").write_text(json.dumps(dict(line_status=status, implementation_error=dict(kind="a"))), encoding="utf-8")
            records = self.record_dir(name, row)
            records.mkdir(parents=True, exist_ok=True)
            (records/"step4.json").write_text(json.dumps(dict(status=status)), encoding="utf-8")
            if getattr(self, "synced", False):
                self.sync_method()
            return base/"terminal.json"
        records = self.record_dir(name, row)/("resume" if resume else "")
        records.mkdir(parents=True, exist_ok=True)
        disp = None
        if field:
            store = self.tmp/"big"/name/(row[3]+(".resume" if resume else ""))
            store.mkdir(parents=True, exist_ok=True)
            np.savez(store/"disp.npz", d=self.field(name, row).astype(np.float32))
            disp = dict(file=str(store/"disp.npz"), sha256=file_sha256(store/"disp.npz"))
        target = records/("overridden.json" if overridden else "result.json")
        target.write_text(json.dumps(dict(status=status, export_check=dict(suspicious=suspicious), disp=disp, **(extra or {}))), encoding="utf-8")
        return target

    # the method arm's step-5 and move records (A5b: the entry knows the method cell by them) ---------------------------------------------------------------
    def work_cell(self, row):
        """Where HB's finalized record says the cell was (holdout_B_v1/work/method/<pair>/ on /home) before the move to 'D:' (cells/method/<pair>/)."""
        return self.tmp/"work"/"method"/row[3]

    def bz_decision(self, row):
        """hb_review.finalize's bz_decision by HB's own selection (A5b review I2: hb_core.bz_decision_auth, the function bz_declined and review_decision call): for a method cell whose
        terminal line_status is hx_implementation_error or hx_provenance_rejected, each (arm, record) that a committed hb no_resume record of the pair names, in the records' name order, is
        offered to HB (arm review: step 5, the case of C22.5-6 from the record's implementation_error.kind, a withdrawal's hb_review.WITHDRAW_KIND (b); arm method: step 4,
        hb_core.implementation_kind of the cell, from its line.json); the first that HB takes, else None."""
        hb = hb_core()
        term = self.cell("method", row)/"terminal.json"
        status = json.loads(term.read_text(encoding="utf-8")).get("line_status") if term.is_file() else None
        if status not in ("hx_implementation_error", "hx_provenance_rejected"):
            return None
        offered = []
        for path in sorted((self.repo/"splits"/"bayesreg_v1").glob("holdout_B_v1_hb_no_resume_*.json"), key=lambda q: int(q.stem.rsplit("_", 1)[1])):
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("pair") == row[3] and isinstance(doc.get("record"), str) and (doc.get("arm"), doc["record"]) not in offered:
                offered.append((doc.get("arm"), doc["record"]))
        for arm, record in offered:
            if arm == "review":
                kept = json.loads(Path(record).read_text(encoding="utf-8")) if Path(record).is_file() else {}
                kind, step = ((kept.get("implementation_error") or {}).get("kind") if isinstance(kept, dict) else None), 5
                if kind is None and Path(record).name.startswith(("review_not_finalized", "review_unit_stopped")):
                    kind = "b"                                               # a withdrawal: hb_review.WITHDRAW_KIND (withdrawal_authorization)
            else:
                kind, step = hb.implementation_kind(self.cell("method", row)), 4
            auth = hb.bz_decision_auth(arm, row[3], record, line_status=status, kind=kind)
            if auth is not None:
                return dict(step=step, path=str(auth.path), sha256=auth.sha256)
        return None

    def move_rows(self, row):
        cell = self.cell("method", row)
        return [dict(file=p.relative_to(cell).as_posix(), src=str(self.work_cell(row)/p.relative_to(cell)), dst=str(p), bytes=p.stat().st_size, sha256_before=file_sha256(p),
                     sha256_after=file_sha256(p), ok=True, problem=None) for p in sorted(cell.rglob("*")) if p.is_file()]

    def sync_method(self, *, moves=True):
        """HB's finalized record (hb_review.finalize's fields: cell, terminal, review, line_status, part.output / part.t88_repair, bz_decision) and each pair's move.json
        (holdout_common.move_files' format: every file of the cell one ok row, src under work/, dst the cell, sha256 before and after) from the method cells as they are now.
        Synthetic: rewritten whenever the world changes a method cell (the real records are written once)."""
        pairs = {}
        for row in self.rows:
            cell = self.cell("method", row)
            rows = self.move_rows(row)
            sha = {r["file"]: r["sha256_after"] for r in rows}
            if moves:
                rec = self.record_dir("method", row)
                rec.mkdir(parents=True, exist_ok=True)
                (rec/H.MOVE_NAME).write_text(json.dumps(dict(schema=H.MOVE_SCHEMA, src_dir=str(self.work_cell(row)), dst_dir=str(cell), utc=H.utc(), files=rows, passed=True, problems=[],
                                                             error=None, rule="synthetic")), encoding="utf-8")
            term = cell/"terminal.json"
            reviewed = "review/review.json" in sha
            pairs[row[3]] = dict(cell=str(self.work_cell(row)), terminal=dict(path=str(self.work_cell(row)/"terminal.json"), sha256=sha["terminal.json"]) if "terminal.json" in sha else None,
                                 review=dict(path=str(self.work_cell(row)/"review"/"review.json"), sha256=sha["review/review.json"]) if reviewed else None,
                                 line_status=json.loads(term.read_text(encoding="utf-8")).get("line_status") if term.is_file() else None,
                                 part=dict(output={f: sha.get(f) for f in ("output/final.npz", "output/disp.npz", "output/cert.json", "output/line.json")}, state_hash=None,
                                           t88_repair={f: sha.get(f) for f in ("t88/run.json", "t88/field_0600.npz", "repair/repair.json", "repair/repair.npz")},
                                           fallback=None, superseded=[], evidence={}) if reviewed else None,
                                 bz_decision=self.bz_decision(row))
        self.review_root.mkdir(parents=True, exist_ok=True)
        (self.review_root/H.FINALIZED_NAME).write_text(json.dumps(dict(schema=H.FINALIZED_SCHEMA, batch="synthetic", pairs=pairs, utc=H.utc(), rule="synthetic")), encoding="utf-8")
        self.synced = True

    def write_prerepair(self, row, *, passed=True, checks=None, disp=True, inputs=None):
        """The repair ablation's inputs as experiments/hb/hb_prerepair.py leaves them: t88/ and repair/ files in the method cell (synthetic; the finalized record names them), the preparation
        record PREREPAIR_ROOT/<pair>/prerepair.json and PREREPAIR_BIG/<pair>/prerepair_disp.npz (a synthetic d: the method's field scaled)."""
        cell = self.cell("method", row)
        (cell/"t88").mkdir(parents=True, exist_ok=True)
        (cell/"repair").mkdir(parents=True, exist_ok=True)
        np.savez(cell/"t88"/"field_0600.npz", u=self.method_u(row)*1.1)
        (cell/"t88"/"run.json").write_text(json.dumps(dict(status="done", field=dict(file="field_0600.npz"))), encoding="utf-8")
        (cell/"repair"/"repair.json").write_text(json.dumps(dict(status="certified", source=dict(field_sha256="0"*64))), encoding="utf-8")
        self.sync_method()
        record_dir, big = self.roots()["PREREPAIR_ROOT"]/row[3], self.roots()["PREREPAIR_BIG"]/row[3]
        record_dir.mkdir(parents=True, exist_ok=True)
        big.mkdir(parents=True, exist_ok=True)
        output = None
        if disp:
            np.savez(big/H.PREREPAIR_DISP, d=self.field("method", row)*1.1)
            output = dict(path=str(big/H.PREREPAIR_DISP), sha256=file_sha256(big/H.PREREPAIR_DISP), d_sha256=None)
        names = ("t88/field_0600.npz", "t88/run.json", "repair/repair.json")
        document = dict(schema=H.PREREPAIR_SCHEMA, pair=row[3], passed=passed, checks=dict(checks or {k: True for k in H.PREREPAIR_CHECKS}), problems=[] if passed else ["synthetic"],
                        inputs=dict(inputs or {n: dict(path=str(cell/n), sha256=file_sha256(cell/n)) for n in names}), output=output, rule="synthetic")
        (record_dir/H.PREREPAIR_RECORD).write_text(json.dumps(document), encoding="utf-8")
        return document

    def write_review(self, row, *, finalized=True):
        review = self.cell("method", row)/"review"
        review.mkdir(parents=True, exist_ok=True)
        (review/"review.json").write_text(json.dumps(dict(schema="synthetic.review", pair=row[3], finalized=finalized, redo=None)), encoding="utf-8")
        if getattr(self, "synced", False):
            self.sync_method()

    def method_d_sha256(self, row):
        return read_disp(self.cell("method", row)/"output"/"disp.npz")[1]["d_sha256"]

    def pq_dirs(self, row):
        return self.tmp/"pq_small"/row[3], self.tmp/"pq_big"/row[3]

    def write_pq(self, row, *, d_sha256=None, files=("pq.json", "pq.npz", "baselines.npz")):
        """C22.H6-1: pq.json (d's sha256 at sampling.d_sha256), pq.npz (q, q4), baselines.npz (h uint8, the rest float32), on the pair's grid."""
        small, big = self.pq_dirs(row)
        small.mkdir(parents=True, exist_ok=True)
        big.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(100+row[0])
        if "pq.json" in files:
            inputs = {side: {key: input_sha(subject, name) for name, key in H.PQ_INPUT_KEYS.items()} for side, subject in (("fixed", row[1]), ("moving", row[2]))}
            (small/"pq.json").write_text(json.dumps(dict(schema="synthetic.pq", sampling=dict(d_sha256=d_sha256 or self.method_d_sha256(row)), inputs=inputs,
                                                         method_files={"t88/field_0600.npz": dict(sha256=input_sha(row[3], "field_0600.npz"))})), encoding="utf-8")
        if "pq.npz" in files:
            np.savez(big/"pq.npz", q=rng.random(syn.SHAPE), q4=rng.random(syn.SHAPE))
        if "baselines.npz" in files:
            base = {k: rng.random(syn.SHAPE).astype(np.float32) for k in BASELINE_KEYS}
            base["h"] = (rng.random(syn.SHAPE) < .2).astype(np.uint8)
            np.savez(big/"baselines.npz", **base)

    def pq_files(self):
        out = []
        for row in self.rows:
            small, big = self.pq_dirs(row)
            out += [p for root in (small, big) if root.is_dir() for p in sorted(root.iterdir()) if p.is_file()]
        return out

    def pq_list(self):
        return self.tmp/"pq_small"/PQ_LIST

    def release_pq(self, *, wrong=None, note=None, split_commits=False):
        """PQ's list (pq_small/pq_outputs_manifest.txt, sha256sum lines, sorted) and its release in the code tree: pq_outputs_manifest.sha256 ('<sha256>  <list
        path>') and pq_release.md ('放行：<sha256>') in one commit (alignment item 5)."""
        rows = sorted((str(p), "0"*64 if str(p) == str(wrong) else file_sha256(p)) for p in self.pq_files())
        listed = self.pq_list()
        listed.parent.mkdir(parents=True, exist_ok=True)
        listed.write_text("".join(f"{sha}  {path}\n" for path, sha in rows), encoding="utf-8")
        sha = file_sha256(listed)
        (self.repo/H.PQ_RELEASE_SHA_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.repo/H.PQ_RELEASE_SHA_REL).write_text(f"{sha}  {listed}\n", encoding="utf-8")
        if split_commits:
            commit(self.repo, "PQ list's sha256")
        (self.repo/H.PQ_RELEASE_NOTE_REL).write_text((f"放行：{sha}" if note is None else note)+"\n", encoding="utf-8")
        commit(self.repo, "PQ release")
        return sha

    def arm(self, name):
        return next(a for a in self.table["arms"] if a["name"] == name)

    def write_input_lock(self, **wrong):
        """The step-8 input lock in HB's format (hb_store.input_lock): rows of the subjects' probs36, structure_groups, provenance and of the pairs'
        field_0600.npz by sha256 (synthetic values); `wrong`: {subject or pair: file name} whose row holds another sha256."""
        rows = []
        for subject in self.subjects:
            for name in H.PQ_INPUT_KEYS:
                sha = "0"*64 if wrong.get(subject) == name else input_sha(subject, name)
                rows.append(dict(kind="prediction", subject=subject, path=f"/x/pred/{subject}/{name}", sha256=sha))
        for row in self.rows:
            rows.append(dict(kind="method_field", pair=row[3], path=f"/x/method/{row[3]}/t88/field_0600.npz",
                             sha256="0"*64 if wrong.get(row[3]) else input_sha(row[3], "field_0600.npz"), line_status="hx_certified"))
        path = self.repo/H.INPUT_LOCK_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(schema=H.INPUT_LOCK_SCHEMA, rows=rows)), encoding="utf-8")
        return path

    def roots(self):
        """The world's roots of the module (REVIEW_ROOT, PREREPAIR_ROOT, PREREPAIR_BIG), set for the world's own calls before patch() (the build) and kept as patch() sets them."""
        return dict(REVIEW_ROOT=self.review_root, PREREPAIR_ROOT=self.results/"holdout_B_v1"/"prerepair", PREREPAIR_BIG=self.tmp/"d_prerepair")

    def needed(self):
        saved = {k: getattr(H, k) for k in self.roots()}
        try:
            for k, v in self.roots().items():
                setattr(H, k, v)
            return self._needed()
        finally:
            for k, v in saved.items():
                setattr(H, k, v)

    def _needed(self):
        table = H.validate_arm_table(json.loads((self.repo/H.ARM_TABLE_REL).read_text()))
        cache = {}
        cells = H.check_cells(table, self.rows, repo=self.repo, cache=cache)
        ablation = H.check_ablation(table, self.rows, cells, cache)
        if self.bz_ii is not None:
            return sorted(H.needed_inputs(table, self.rows, cells, ablation=ablation))
        entries = {}
        if self.pq_list().is_file():
            for line in self.pq_list().read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sha, path = line.split("  ", 1)
                    entries[path] = sha
        states = H.pq_states(table, self.rows, cells, dict(method_side="released", entries=entries))
        return sorted(H.needed_inputs(table, self.rows, cells, states, self.pq_list(), ablation))

    def write_manifest(self, *, drop=None, extra_input=None, records=True, commit_it=True, header=None):
        """The D108 manifest (input lines only) and the record-only file, in one commit (revision 1c item 1)."""
        needed = self.needed()
        lines = [f"# {H.SCHEMA_MANIFEST}", f"# prereg_sha256 {self.prereg_sha}", f"# arm_table_sha256 {self.table_sha}",
                 f"# unseal_sha256 {file_sha256(self.repo/H.UNSEAL_REL)}"]
        if self.bz_ii is not None:
            lines.append(f"# {H.PQ_HEADER_BZ_II} {self.bz_ii}")
        lines += list(header or [])
        lines += [f"{file_sha256(path)}  {path}" for path in needed if path != drop]
        if extra_input:
            lines.append(f"{file_sha256(extra_input)}  {extra_input}")
        (self.repo/H.MANIFEST_REL).parent.mkdir(parents=True, exist_ok=True)
        path, text = self.repo/H.MANIFEST_REL, "\n".join(lines)+"\n"
        same = path.is_file() and path.read_text(encoding="utf-8") == text
        path.write_text(text, encoding="utf-8")
        if records and not same:                                       # an unchanged manifest keeps its record-only file (they were committed together)
            moved = self.tmp/"moves"/"move_0.json"
            gone = self.tmp/"cells"/"method"/self.rows[0][3]/"t88"/"field_0600.npz"
            self.versions = getattr(self, "versions", 0)+1              # the two files change together in every commit of the world
            (self.repo/H.RECORDS_REL).write_text(f"# version {self.versions}\n{file_sha256(moved)}  {moved}\n{'e'*64}  {gone}\n"
                                                 f"#begin input_lock {'0'*40} runs/x {'1'*64}\n{'7'*64}  /x/probs36.npz\n#end input_lock\n", encoding="utf-8")
        if commit_it:
            commit(self.repo, "manifest")

    def write_authorization(self, arm, row, kind, *, as_arm=None, **changes):
        """holdout_common's committed record of one cell (resume / no_resume; splits/bayesreg_v1/holdout_B_v1_<tool>_<kind>_<k>.json), written on BZ's behalf."""
        spec = arm["authorization"]
        folder = self.repo/"splits"/"bayesreg_v1"
        k = 1+len(list(folder.glob(f"holdout_B_v1_{spec['tool']}_{kind}_*.json")))
        document = dict(schema=f"bayesreg.{spec['tool']}.{kind}.v1", written_by="lead on BZ's behalf", words="（合成测试）BZ：照办", time_local="11-09 10:00 PM CST",
                        utc=H.utc(), reason="synthetic", tool=spec["tool"], arm=as_arm or spec["arm"], pair=row[3], pair_set="holdout")
        if kind == "resume":
            document.update(lock_sha256="1"*64, previous_record=None, previous_record_sha256=None)
        elif kind == "no_resume":
            document.update(record=None, record_sha256=None, note="未批补跑")
        document.update(changes)
        path = folder/f"holdout_B_v1_{spec['tool']}_{kind}_{k}.json"
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        commit(self.repo, f"authorization {path.name}")
        if getattr(self, "synced", False):
            self.sync_method()
        return path

    # pins --------------------------------------------------------------------------------------------------------------------------------------------
    def decision(self):
        import nibabel
        out = self.results/"ev"/"truth_gate_v1"
        out.mkdir(parents=True, exist_ok=True)
        counts = dict(reader=dict(subjects=20, identical=20), metrics=dict(pairs=19, identical=19))
        commit_ = head(self.repo)
        result = dict(schema=H.SCHEMA_GATE, reading=H.GATE_READING_PASS, plan=dict(sha256="f"*64), counts=counts, code=dict(head=commit_, clean=True))
        path = out/"truth_gate.json"
        path.write_text(json.dumps(result), encoding="utf-8")                  # the gate as if it ran on HEAD now (its result, its lock's first line)
        lock = self.results/"ev"/"truth_gate.lock"
        lock.write_text(json.dumps(dict(tool="experiments/ev/truth_gate.py", code_head=commit_))+"\n", encoding="utf-8")
        return dict(result_json=str(path), result_sha256=file_sha256(path), reading=H.GATE_READING_PASS, code_commit=commit_,
                    environment=dict(numpy=np.__version__, nibabel=nibabel.__version__), counts=counts, plan_sha256="f"*64, decided_by="test", review="test")

    def pq_replay(self):
        """A stand-in of pqh_pins.PQ_REPLAY_DECISION: what replay_decision_problem would check (its result at the gate's one path, pqh_replay.result_path()), and the code commit."""
        return dict(result_json=str(self.results/"pq_replay_v1"/"result"/"replay_result.json"), result_sha256="e"*64, reading=REPLAY_PASS, code_commit=head(self.repo),
                    plan_sha256="d"*64, decided_by="test", review="test")

    def pin_lists(self, monkeypatch):
        monkeypatch.setattr(S, "REPO", self.repo)
        monkeypatch.setattr(S, "SEALED_LISTS", tuple(self.specs))
        S._reset_cache()

    def patch(self, monkeypatch, *, spy=True, rule_one="stub"):
        self.pin_lists(monkeypatch)
        monkeypatch.setattr(H, "PAIRS_SHA256", file_sha256(self.repo/H.PAIRS_REL))
        monkeypatch.setattr(H, "N_PAIRS", len(self.rows))
        monkeypatch.setattr(H, "PREREG_SHA256", self.prereg_sha)
        monkeypatch.setattr(H, "ARM_TABLE_SHA256", self.table_sha)
        monkeypatch.setattr(H, "DATA_ROOT", self.data)
        monkeypatch.setattr(H, "RESULTS", self.results)
        monkeypatch.setattr(H, "EVAL_DIR", self.results/"holdout_B_v1"/"eval")
        monkeypatch.setattr(H, "MASTER_LOCK", self.results/"holdout_B_v1"/"eval"/"holdout_eval.lock")
        monkeypatch.setattr(H, "PAIR_LOCK_DIR", self.results/"holdout_B_v1"/"eval"/"pair_locks")
        monkeypatch.setattr(H, "TRUTH_GATE_OUT", self.results/"ev"/"truth_gate_v1")
        monkeypatch.setattr(H, "TRUTH_GATE_LOCK", self.results/"ev"/"truth_gate.lock")
        for key, value in self.roots().items():
            monkeypatch.setattr(H, key, value)
        if rule_one == "stub":                                         # rule one needs runs/holdout_B_v1/pinned_code.json and P1..HEAD of records only: tested apart (pinned())
            self.rule_one_calls = []
            monkeypatch.setattr(H, "unit_identity_problems", lambda commit=None, *, repo=None, run=None: (self.rule_one_calls.append(str(repo)) or
                                                                                                      dict(stub="rule one stubbed by the world", pinned=None), []))
        monkeypatch.setattr(H, "TRUTH_GATE_PLAN_SHA256", "f"*64)
        monkeypatch.setattr(H, "replay_decision_problem",                    # pqh_replay's check, stood in: the reading only (the check itself: tests/test_bayesreg/pqh/test_pqh_replay.py)
                            lambda decision=None, *, repo=None: None if isinstance(decision, dict) and decision.get("reading") == REPLAY_PASS
                            else "PQ_REPLAY_DECISION is None or not 逐位相同 (synthetic)")
        monkeypatch.setattr(H, "imported_from", lambda root: True)      # the world's repository is not where bayesreg is imported from (tested apart)
        monkeypatch.setattr(H._holdout_common(), "AUTH_ROOT", self.repo)
        if spy:
            original = H._load_label_volume

            def spy_reader(path):
                self.reads.append(str(path))
                return original(path)
            monkeypatch.setattr(H, "_load_label_volume", spy_reader)

    def open(self, out=None, *, fresh=True, **kwargs):
        """open_holdout as a fresh process would run it: the guard's cache of the sealed lists empty first (C23.6-5) unless fresh is False."""
        if "decision" not in kwargs:                                    # (not setdefault: the gate's result would be rewritten for a given decision)
            kwargs["decision"] = self.decision()
        kwargs.setdefault("pq_replay", self.pq_replay())
        kwargs.setdefault("run", no_units)
        if fresh:
            S._reset_cache()
        return H.open_holdout(Path(H.EVAL_DIR)/"out" if out is None else out, repo=self.repo, **kwargs)


def input_sha(subject, name):
    """A synthetic sha256 of a subject's HI output (or a pair's field_0600.npz): the same in pq.json and the input lock unless a test changes one."""
    import hashlib
    return hashlib.sha256(f"{subject}/{name}".encode()).hexdigest()


def no_units(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, "", "")


def units(*names):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "".join(f"{n} loaded active running x\n" for n in names), "")
    return run


__all__ = ["World", "units", "no_units", "commit", "git", "head", "SUBJECTS", "ROWS", "MAIN", "H1_ID", "H4_ID", "DEV_ID", "WORDS", "BASELINE_KEYS", "REPLAY_PASS"]
