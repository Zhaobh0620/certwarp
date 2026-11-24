"""Shared by the SG tests (contract C23.9): a temporary git repository that bayesreg's pinned values point into (bayesreg.io.sealed.REPO and SEALED_LISTS,
changed with monkeypatch only; C23.9 preamble), holding synthetic sealed lists named holdout_B_v1, H1, H4 and H3, the pinned allowlist
configs/bayesreg/dev_subjects.json, BZ's synthetic unseal record at the pinned path and its line in docs/decision_log.md.

Synthetic ids only (OASIS_OAS1_09xx_MR1: no OASIS subject has a number above 0457; IBSR_xx for H3); every file in tmp_path; no subject's image or label is
opened by the guard tests.  The real ids (only ids) are read at run time from the real pinned lists where a test needs them (C23.9-1, -9)."""
import os
from pathlib import Path
import subprocess
import types

import bayesreg.io.sealed as S
from bayesreg.io import unseal as U
from bayesreg.io.guard import load_dev_allowlist
from bayesreg.io.pair import subject_list_sha256
from bayesreg.seg.unet import synthetic
from jd_curl.three_d.nifti import file_sha256

ROOT = Path(__file__).resolve().parents[3]
REAL = dict(REPO=S.REPO, SEALED_LISTS=S.SEALED_LISTS)
oasis = synthetic.oasis
HOLDOUT = [oasis(960), oasis(961), oasis(962)]          # the synthetic holdout_B_v1
H1 = [oasis(970), oasis(971)]
H4 = [oasis(972)]
H3 = ['IBSR_01', 'IBSR_02']
DEV = [oasis(980), oasis(981)]
T0 = [oasis(982)]
TRAIN = [oasis(983), oasis(984)]
WORDS = '（合成测试）解封合成留出集'
GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='sg-test', GIT_AUTHOR_EMAIL='sg-test@example.invalid',
               GIT_COMMITTER_NAME='sg-test', GIT_COMMITTER_EMAIL='sg-test@example.invalid')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True, env=GIT_ENV)


def commit_all(repo, message='test'):
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--allow-empty', '-m', message)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def write_json(path, document):
    import json
    return write_text(path, json.dumps(document, indent=1, ensure_ascii=False)+'\n')


def pin(monkeypatch, repo, specs):
    monkeypatch.setattr(S, 'REPO', Path(repo))
    monkeypatch.setattr(S, 'SEALED_LISTS', tuple(specs))
    S._reset_cache()


def sources_default(extra=()):
    return [('unsealed_B', HOLDOUT, True), ('dev', DEV, False), ('t0', T0, True), ('train', TRAIN, False)]+list(extra)


def record_document(allowlist, specs, **changes):
    holdout = {s['name']: s for s in specs}['holdout_B_v1']
    document = dict(schema=U.UNSEAL_SCHEMA, written_by='BZ', bz_words=WORDS, time_local='2026-11-07 6:00 AM CST', utc='2026-11-07T12:00:00Z', sealed_list='holdout_B_v1',
                    subjects_file=holdout['path'], subjects_file_sha256=holdout['sha256'], subjects_sha256=subject_list_sha256(sorted(HOLDOUT)), n_subjects=len(HOLDOUT),
                    allowlist_source='unsealed_B', allowlist_sha256=allowlist.sha256, segmenter_id='unet35_v4_ens', registry_sha256='a'*64,
                    purpose='synthetic test of the sealed guard (C23)')
    document.update(changes)
    return document


def log_line(repo, record_path, words):
    import hashlib
    sha = hashlib.sha256(Path(record_path).read_bytes()).hexdigest()
    log = Path(repo)/U.DECISION_LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'| 2026-11-07 | the unseal record {sha} | {words} |\n')
    return sha


def make_world(tmp_path, monkeypatch, *, sources=None, record=True, **changes):
    """The synthetic tree, pinned.  `changes`: fields of the record to change; record False: no record (the lists and the allowlist only)."""
    repo = Path(tmp_path)/'repo'
    repo.mkdir(parents=True)
    git(repo, 'init', '-q')
    write_text(repo/'lists/b.txt', ''.join(f'{s}\n' for s in HOLDOUT))
    write_text(repo/'lists/h1.txt', ''.join(f'{s}\n' for s in H1))
    write_text(repo/'lists/h4.txt', '# H4 (synthetic)\n'+''.join(f'{s}\n' for s in H4))
    write_text(repo/'lists/h3.tsv', 'fixed\tmoving\n'+f'{H3[0]}\t{H3[1]}\n')
    specs = (dict(name='holdout_B_v1', path='lists/b.txt', sha256=file_sha256(repo/'lists/b.txt'), columns=None, count=len(HOLDOUT)),
             dict(name='H1', path='lists/h1.txt', sha256=file_sha256(repo/'lists/h1.txt'), columns=None, count=len(H1)),
             dict(name='H4', path='lists/h4.txt', sha256=file_sha256(repo/'lists/h4.txt'), columns=None, count=len(H4)),
             dict(name='H3', path='lists/h3.tsv', sha256=file_sha256(repo/'lists/h3.tsv'), columns=('fixed', 'moving'), count=len(H3)))
    allowlist = load_dev_allowlist(write_json(repo/U.ALLOWLIST_REL, synthetic.allowlist_document(sources_default() if sources is None else sources)))
    w = types.SimpleNamespace(repo=repo, specs=specs, allowlist=allowlist, path=repo/U.UNSEAL_REL, document=None, tmp=Path(tmp_path))
    if record:
        w.document = record_document(allowlist, specs, **changes)
        write_json(w.path, w.document)
        log_line(repo, w.path, w.document['bz_words'] if isinstance(w.document.get('bz_words'), str) else WORDS)
    commit_all(repo, 'the synthetic tree')
    pin(monkeypatch, repo, specs)
    return w


def rewrite(w, *, log=True, commit=True, **changes):
    """The record with `changes`, logged (its sha256 and bz_words on one line) and committed unless told otherwise."""
    w.document = dict(w.document, **changes)
    write_json(w.path, w.document)
    if log:
        log_line(w.repo, w.path, w.document['bz_words'] if isinstance(w.document.get('bz_words'), str) else WORDS)
    if commit:
        commit_all(w.repo, 'the record changed')
    return w


def bound(w):
    """(the unseal object of the real loader, the allowlist bound to it)."""
    unseal = U.load_unseal(w.path)
    return unseal, U.bind_unseal(w.allowlist, unseal)
