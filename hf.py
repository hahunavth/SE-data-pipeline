import time
from huggingface_hub import upload_folder, repo_exists, create_repo, create_branch

from mp import hf_api_lock


def hf_retry_decorator(max_retries=3, time_between_retries=0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    time.sleep(time_between_retries)
                    print(f"Error {func.__name__}: {e}")
                    print(f"Retrying {func.__name__}")
            if max_retries > 0:
                raise Exception(
                    f"Failed to execute {func.__name__} after {max_retries} retries"
                )

        return wrapper

    return decorator


@hf_retry_decorator()
def create_repo_if_not_exists(repo_id, repo_type="dataset", **kwargs):
    # if not repo_exists(repo_id, repo_type=repo_type):
    return create_repo(
        repo_id, repo_type=repo_type, private=True, exist_ok=True, **kwargs
    )


@hf_retry_decorator(max_retries=10, time_between_retries=60)
def upload_folder_retry(
    repo_id, repo_type, folder_path, path_in_repo, revision="main", commit_message="", **kwargs
):
    with hf_api_lock:
        return upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            revision=revision,
            commit_message=commit_message,
            **kwargs,
        )
