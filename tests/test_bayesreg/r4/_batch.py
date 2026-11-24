"""修订 6a M2：队列与单步只从这一角色的 batch 文件记的运行检出起，batch 文件不在也拒。测试里的队列与单步用这里写的 batch：运行检出就是这份代码（rc.ROOT），
续跑仓库记成测试根本身（这些测试都显式传 repo 与 require_committed=False，不走 git）。"""
from pathlib import Path


def batch_here(root, roles=('bench', 'dev')):
    import r4_common as rc
    for role in roles:
        path = Path(root)/('batch.json' if role == 'bench' else 'dev_batch.json')
        if not path.exists():
            rc.write_json(path, dict(schema='bayesreg.r4.test_batch.v1', role=role, checkout=str(rc.ROOT.resolve()), resume_repo=str(Path(root).resolve()),
                                     note='a test batch for launch_problem (修订 6a M2); no code_commit, no unit reads it'))
    return root
