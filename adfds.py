import os
import subprocess


commit = (
    os.getenv("CI_COMMIT_SHA")
    or subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
)

    # repo url
repo_url = (
    os.getenv("CI_REPOSITORY_URL")
     or subprocess.check_output(
        ["git", "remote", "get-url", "origin"]
    ).decode().strip()
)

    # нормализуем SSH → HTTPS
if repo_url.startswith("git@"):
    repo_url = repo_url.replace("git@", "https://").replace(":", "/")
repo_url = repo_url.removesuffix(".git")

commit_url = f"{repo_url}/-/commit/{commit}"

print(commit_url)