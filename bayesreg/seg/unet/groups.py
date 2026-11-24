"""Structure groups (configs/bayesreg/structure_groups_v1.json; docs/bayesreg/04_分割器.md §2).

The network predicts the 36 seg35 classes (class c = label c; 0 = background "Unknown"). A grouping
partitions labels 0..35 into background_labels (channel 0) and K_S named groups (channels 1..K_S);
the group probability is the sum of its class probabilities, so the K_S + 1 group channels are again a
simplex wherever the class probabilities are. Pure numpy.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from jd_curl.validation import InputValidationError

from ...io.guard import is_safe_id, strict_json_object

GROUPS_SCHEMA = "bayesreg.structure_groups.v1"
DEFAULT_GROUPS_PATH = Path(__file__).resolve().parents[3]/"configs"/"bayesreg"/"structure_groups_v1.json"
BACKGROUND_NAME = "background"
# Statuses a grouping may have to be used for inference (review 2026-10-30, item 10). Anything else -- in
# particular "reference_unverified" (tissue4_ref before check-tissue4 has confirmed it) -- is refused
# unless the caller passes allow_unverified=True, which is recorded in the provenance.
INFERENCE_STATUSES = ("default", "ablation", "verified")


class GroupingUnverified(Exception):
    """The grouping's status does not admit inference; status `grouping_unverified`."""
    status = "grouping_unverified"


def require_inference_grouping(grouping, *, allow_unverified=False):
    """Refuse a grouping whose status is not in INFERENCE_STATUSES (unless explicitly allowed)."""
    if grouping.status not in INFERENCE_STATUSES and not allow_unverified:
        raise GroupingUnverified(f"grouping {grouping.name!r} has status {grouping.status!r}; inference uses only "
                                 f"{INFERENCE_STATUSES} unless allow_unverified is set (and recorded)")
    return dict(grouping_status=grouping.status, grouping_status_admitted=grouping.status in INFERENCE_STATUSES,
                allow_unverified_grouping=bool(allow_unverified))


@dataclass(frozen=True)
class Grouping:
    name: str
    group_names: tuple            # K_S names (class_names of the P_S provenance), channel k+1
    group_labels: tuple           # tuple of label tuples, aligned with group_names
    background_labels: tuple
    n_classes: int
    status: str
    source_path: str
    source_sha256: str

    @property
    def k(self):
        return len(self.group_names)

    def class_to_channel(self):
        """int array (n_classes,): label c -> group channel (0 = background, k+1 = group k)."""
        table = np.full(self.n_classes, -1, dtype=np.int64)
        table[list(self.background_labels)] = 0
        for k, labels in enumerate(self.group_labels):
            table[list(labels)] = k+1
        return table

    def record(self):
        return dict(grouping=self.name, grouping_status=self.status, groups_path=self.source_path,
                    groups_sha256=self.source_sha256, class_names=list(self.group_names),
                    background_class=BACKGROUND_NAME, background_labels=list(self.background_labels),
                    group_labels={n: list(l) for n, l in zip(self.group_names, self.group_labels)})


def _check_grouping(name, spec, n_classes):
    where = f"groupings.{name}"
    if not isinstance(spec, dict) or not {"status", "background_labels", "groups"} <= set(spec):
        raise InputValidationError(f"{where}: status, background_labels and groups required")
    groups = spec["groups"]
    if not isinstance(groups, list) or not groups:
        raise InputValidationError(f"{where}.groups: non-empty list required")
    names, labels_seen = [], []
    background = spec["background_labels"]
    for item in [background]+[g.get("labels") if isinstance(g, dict) else None for g in groups]:
        if not isinstance(item, list) or not item or not all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n_classes for v in item):
            raise InputValidationError(f"{where}: every label list must hold integers in [0, {n_classes})")
        labels_seen += item
    for group in groups:
        if set(group) != {"name", "labels"} or not is_safe_id(group["name"]) or group["name"].lower() == BACKGROUND_NAME:
            raise InputValidationError(f"{where}: each group is {{name: identifier (not 'background'), labels: [...]}}")
        names.append(group["name"])
    if len({n.lower() for n in names}) != len(names):
        raise InputValidationError(f"{where}: duplicate group names (case-insensitive)")
    if sorted(labels_seen) != list(range(n_classes)):
        raise InputValidationError(f"{where}: background and groups must partition labels 0..{n_classes-1}")
    return tuple(names), tuple(tuple(g["labels"]) for g in groups), tuple(background)


def load_grouping(path=DEFAULT_GROUPS_PATH, name=None):
    """Read and check the grouping table; `name` None -> the file's default_grouping."""
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_json_object)
    if not isinstance(document, dict) or document.get("schema") != GROUPS_SCHEMA:
        raise InputValidationError(f"{path}: schema must be {GROUPS_SCHEMA!r}")
    n_classes = document.get("n_network_classes")
    if not isinstance(n_classes, int) or n_classes < 2:
        raise InputValidationError("n_network_classes: integer >= 2 required")
    groupings = document.get("groupings")
    if not isinstance(groupings, dict) or not groupings:
        raise InputValidationError("groupings: non-empty object required")
    for key, spec in groupings.items():           # every grouping in the file must be well-formed
        _check_grouping(key, spec, n_classes)
    name = document.get("default_grouping") if name is None else name
    if name not in groupings:
        raise InputValidationError(f"grouping {name!r} not in {sorted(groupings)}")
    names, labels, background = _check_grouping(name, groupings[name], n_classes)
    return Grouping(name=name, group_names=names, group_labels=labels, background_labels=background,
                    n_classes=n_classes, status=groupings[name]["status"], source_path=str(path),
                    source_sha256=hashlib.sha256(raw).hexdigest())


def group_probabilities(probs, grouping, *, dtype=np.float64):
    """(n_classes, ...) class probabilities -> (K_S + 1, ...) group probabilities, background at channel 0."""
    probs = np.asarray(probs)
    if probs.shape[0] != grouping.n_classes:
        raise InputValidationError(f"probs: expected {grouping.n_classes} classes on axis 0, got {probs.shape[0]}")
    out = np.empty((grouping.k+1,)+probs.shape[1:], dtype=dtype)
    out[0] = probs[list(grouping.background_labels)].sum(axis=0, dtype=dtype)
    for k, labels in enumerate(grouping.group_labels):
        out[k+1] = probs[list(labels)].sum(axis=0, dtype=dtype)
    return out


def group_labels(labels, grouping):
    """Hard label map (values 0..n_classes-1) -> group channel index map (0 = background)."""
    return grouping.class_to_channel()[np.asarray(labels, dtype=np.int64)]


__all__ = ["BACKGROUND_NAME", "DEFAULT_GROUPS_PATH", "GROUPS_SCHEMA", "Grouping", "GroupingUnverified",
           "INFERENCE_STATUSES", "group_labels", "group_probabilities", "load_grouping", "require_inference_grouping"]
