"""The read rules of R3b's units (runs/20261111-r3/calibration_plan.md section 4), for PQ's spy (experiments/pq/pq_firewall.Spy, used unchanged).

The spy calls rules.in_scope(raw, resolved), rules.judge(raw, resolved) and rules.describe().  Watched scope as C22.Q1-3: the results root, the data roots (the
repository's data/ and what it links to, /mnt/d/data), the repository's configs/, splits/, runs/, docs/; code files (.py) outside it.  A watched file may be
opened iff it is allowed and not forbidden (forbidden wins), except that the eval unit's planned truth files are allowed by name and only they.
Allowed: the files listed by the unit (explicit), the unit's own output directories under OUT, configs/bayesreg/, the sealed lists the guard reads.
Forbidden: label files under the data roots (aligned_seg35, aligned_seg4, any .nii.gz with 'seg' in its name) unless listed as the unit's truth; metrics_*.json,
repair_metrics.json, evaluation.json; OUT/eval/ and OUT/analysis/ for the units before eval; any analysis/ directory, decision*, summary*, per_pair*, long.csv
(the unit's own outputs excepted); the read locks of seg/ (*_dev_read.lock) and the directories they protect; R3b's own read lock (the eval unit creates it); result*.md and reviews/ under
docs/ and runs/.
"""
import os
from pathlib import Path

import r3b_common as rc

PROTECTED_DIRS = ("eval_uf", "eval_u1", "eval_seg", "control_check", "eval_nn", "tta_v4", "eval_r2")
EVALUATION_NAMES = ("repair_metrics.json", "evaluation.json")


def _forms(path):
    raw = Path(os.path.abspath(os.fsdecode(path)))
    return raw, raw.resolve()


def _both(path):
    return set(_forms(path))


def _under(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


class CalibRules:
    """unit: 'infer', 'decode' or 'eval'.  files: explicit allowed files; own_dirs: the unit's output directories (allowed, even if named like an analysis
    product); truth: the planned truth files (eval only)."""

    def __init__(self, *, unit, repo, results, data, images, out, files=(), own_dirs=(), truth=(), lock=rc.READ_LOCK):
        r = lambda p: Path(p).resolve()     # noqa: E731
        self.unit = unit
        self.repo, self.out = r(repo), r(out)
        self.scope = tuple(sorted(set().union(*(_both(p) for p in (results, data, images, Path(repo)/"configs", Path(repo)/"splits", Path(repo)/"runs",
                                                                    Path(repo)/"docs", out))), key=str))
        self.data_roots = tuple(sorted(_both(data) | _both(images), key=str))
        self.results_forms = tuple(sorted(_both(results), key=str))
        self.repo_forms = tuple(sorted(_both(repo), key=str))
        self.out_forms = tuple(sorted(_both(out), key=str))
        from bayesreg.io import sealed as sealed_lists
        self.sealed = frozenset(r(Path(sealed_lists.REPO)/spec["path"]) for spec in sealed_lists.SEALED_LISTS)
        self.files = frozenset({r(p) for p in files} | self.sealed)
        self.truth = frozenset(r(p) for p in truth)
        self.own_dirs = tuple(r(d) for d in own_dirs)
        self.dirs = (self.repo/"configs"/"bayesreg",)+self.own_dirs
        self.plan = r(Path(repo)/rc.PLAN_REL)
        self.lock = frozenset(_both(lock))                     # forbidden to infer and decode; the eval unit creates it (an allowed file)
        if unit == "eval":
            self.files = self.files | {r(lock)}

    def in_scope(self, path, resolved=None):
        if path is None:
            return False
        resolved = path.resolve() if resolved is None else resolved
        if path.suffix == ".py" and resolved.suffix == ".py":
            return False
        return _under(path, self.scope) or _under(resolved, self.scope)

    def _own(self, resolved):
        return any(resolved.is_relative_to(d) for d in self.own_dirs)

    def forbidden(self, path, resolved):
        name, parts = path.name, set(path.parts)
        if resolved in self.truth:
            return None
        if _under(path, self.data_roots) and (name in ("aligned_seg35.nii.gz", "aligned_seg4.nii.gz") or (name.endswith(".nii.gz") and "seg" in name)):
            return f"a label file ({name})"
        if (name.startswith("metrics_") and name.endswith(".json")) or name in EVALUATION_NAMES:
            return f"an evaluation record ({name})"
        if path in self.lock and self.unit != "eval":
            return "R3b's read lock"
        if self._own(resolved):
            return None
        if self.unit != "eval" and _under(path, [p/d for p in self.out_forms for d in (rc.EVAL_DIR, rc.ANALYSIS_DIR)]):
            return "R3b's evaluation or analysis directory"
        if "analysis" in parts or name.startswith(("decision", "summary", "per_pair")) or name == "long.csv":
            return f"an analysis product ({name})"
        if _under(path, [p/"seg" for p in self.results_forms]):
            if name.endswith("_dev_read.lock"):
                return f"a development read lock ({name})"
            if parts & set(PROTECTED_DIRS):
                return "an output a development read lock protects"
        if _under(path, [p/d for p in self.repo_forms for d in ("docs", "runs")]) and resolved != self.plan:
            if name.startswith("result") or "reviews" in parts:
                return f"a result document or a review ({name})"
        return None

    def allowed(self, resolved):
        return resolved in self.files or resolved in self.truth or any(resolved.is_relative_to(d) for d in self.dirs)

    def judge(self, path, resolved=None):
        resolved = path.resolve() if resolved is None else resolved
        if not self.in_scope(path, resolved):
            return None
        for form in (path, resolved):
            reason = self.forbidden(form, resolved)
            if reason is not None:
                return f"forbidden: {reason}"
        if not self.allowed(resolved):
            return "outside the allowed items of calibration_plan.md section 4"
        return None

    def describe(self):
        return dict(unit=self.unit, scope=[str(s) for s in self.scope], allowed_dirs=[str(d) for d in self.dirs], n_allowed_files=len(self.files),
                    n_truth_files=len(self.truth), rule="a watched file may be opened iff allowed and not forbidden; forbidden wins; the eval unit's planned "
                    "truth files are allowed by name (calibration_plan.md section 4)")
