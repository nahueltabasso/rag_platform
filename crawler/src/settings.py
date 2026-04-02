import os

VERSION = '1.O.O'
BOT_NAME = 'scrapy_crawlers'

SPIDER_MODULES = ["crawler.src.spiders"]

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 4 # how many requests will be processed in parallel
DOWNLOAD_DELAY = 15 # delay between requests to the same website
CONCURRENT_REQUESTS_PER_DOMAIN = 4 # how many requests will be processed in parallel to the same domain
CONCURRENT_REQUESTS_PER_IP = 4 # how many requests will be processed in parallel to the same IP
CONCURRENT_ITEMS = 20 # how many items will be processed in parallel

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

FEED_EXPORT_ENCODING = 'utf-8' # encoding for exported data

# CHROMADB SETTINGS
CHROMA_DB_DIR = os.environ.get('CHROMA_DB_DIR', '')
EMBEDDING_MODEL_NAME = 'text-embedding-3-large'

