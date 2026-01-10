import os
import subprocess

def get_git_metadata():
    commit = (
        os.getenv("CI_COMMIT_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        tag = None

    repo_url = (
        os.getenv("CI_REPOSITORY_URL")
        or subprocess.check_output(
            ["git", "remote", "get-url", "origin"]
        ).decode().strip()
    )

    if repo_url.startswith("git@"):
        repo_url = repo_url.replace("git@", "https://").replace(":", "/")
    repo_url = repo_url.removesuffix(".git")

    print(repo_url, commit, tag)
    return repo_url, commit, tag

repo_url, commit, tag = get_git_metadata()

commit_url = f"{repo_url}/commit/{commit}"
tag_url = (f"{repo_url}/releases/tag/{tag}" if tag else None)