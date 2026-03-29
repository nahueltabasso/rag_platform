import os

def register_error_url(dir_name: str, url: str, reason: str) -> None:
    try:
        err_file_path = os.environ.get('BASE_ERROR_DIR') + dir_name # type: ignore
        with open(err_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{url} - Reason: {reason}\n")
    except FileNotFoundError as e:
        raise e