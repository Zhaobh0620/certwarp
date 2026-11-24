"""P0 (runs/20261114-p0/任务书_P0.md section 2; the table in runs/20261114-p0/P0_pins.md): the positive check of a pin -- the pinned sha256 equals the sha256 of its
source file both as committed at HEAD (`git show HEAD:<rel>`, the blob's bytes) and as the file is in the work tree.  Each module's tests call it next to the
test that sets the pin back to None and checks the old refusal.  BRIEF holds the three values the brief wrote out (the other plans: "the file's sha256")."""
import hashlib
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
BRIEF = {'docs/bayesreg/notes/留出集预登记_20261107.md': 'df7b497887dc7b8290493c040635edf7334e9ae3be584e31a89177080836f493',
         'configs/frozen/holdout_B_v1_arm_table.json': '69cb1ddf3e2006116c3d2237f3a4912e68a7238f2ce813f7d67723e5eeb8d661',
         'runs/20261107-ev/truth_gate_fields.sha256': '93cb3f20be07fc4a267ac311b9a9bad82266d577f49336016a8e77210f581c58'}


def committed_sha256(rel, repo=REPO):
    done = subprocess.run(['git', '-C', str(repo), 'show', f'HEAD:{rel}'], capture_output=True)
    assert done.returncode == 0, f"{rel} is not committed at HEAD of {repo}: {done.stderr.decode('utf-8', 'replace')[:300]}"
    return hashlib.sha256(done.stdout).hexdigest()


def tree_sha256(rel, repo=REPO):
    return hashlib.sha256((Path(repo)/rel).read_bytes()).hexdigest()


def assert_pinned(pin, rel, repo=REPO):
    """The pin is a 64-hex sha256, equal to the committed file's and the work tree file's sha256 (and to the brief's value where the brief wrote one)."""
    assert isinstance(pin, str) and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), (rel, pin)
    assert pin == committed_sha256(rel, repo) == tree_sha256(rel, repo), (rel, pin)
    assert pin == BRIEF.get(rel, pin), (rel, pin, BRIEF.get(rel))
