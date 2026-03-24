from typing import Any

class ErrorLoggingMiddleware:
    """Downloader middleware that logs non-200 responses and updates Scrapy stats.

    It increments per-status and per-range counters so you can monitor 4xx/5xx rates.
    """

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def process_response(self, request: Any, response: Any, spider: Any):
        try:
            status = int(getattr(response, 'status', 0))
        except Exception:
            status = 0

        # Increment per-status counter
        try:
            spider.crawler.stats.inc_value(f"response_status/{status}")
        except Exception:
            pass

        # Increment range counters for quick aggregation
        if 400 <= status < 500:
            try:
                spider.crawler.stats.inc_value("response_status_count/4xx")
            except Exception:
                pass
            spider.logger.warning(f"Response {status} for {request.url}")
        elif 500 <= status < 600:
            try:
                spider.crawler.stats.inc_value("response_status_count/5xx")
            except Exception:
                pass
            spider.logger.error(f"Server error {status} for {request.url}")

        return response
