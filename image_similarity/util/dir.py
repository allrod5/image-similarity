import os
from pathlib import Path

import git


def get_project_root() -> Path:
    search_path = __file__
    while True:
        search_path = os.path.dirname(search_path)
        if not _is_project_root(search_path):
            continue
        return Path(search_path)


def _is_project_root(path: str) -> bool:
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False
