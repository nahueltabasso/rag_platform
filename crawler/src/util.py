from crawler.src.base import CommonSpider
import os

def register_error_url(spider: CommonSpider, url: str, reason: str) -> None:
    try:
        err_file_path = os.environ.get('BASE_ERROR_DIR') + "/" + spider.config.error_output_filename # type: ignore
        spider.logger.info(f"Error file: {err_file_path}")
        spider.logger.info(f"Registering error URL: {url} - Reason: {reason} in file: {err_file_path}")
        with open(err_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{url} - Reason: {reason}\n")
    except FileNotFoundError as e:
        raise e