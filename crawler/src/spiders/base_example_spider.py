from scrapy import Selector
from scrapy.http.response import Response
from scrapy.http.request import Request
from scrapy.utils.project import get_project_settings

from crawler.src.base import CommonSpider, CustomSettings
from crawler.src.item import UrlDTO
from crawler.src.loader import UrlsLoader

CUSTOM_SETTINGS = CustomSettings(
    project_settings=get_project_settings().copy_to_dict(),
    spider_name='base_example_spider',
    collection_name='site_base_example',
    base_url='https://quotes.toscrape.com/',
    api_chunks=None,
    api_embeddings=None,
    chunk_size=1200,
    chunk_overlap=200,
    error_output_filename='base_example.txt',
    item_pipelines={
        'crawler.src.pipeline.ValidateFieldsPipeline': 100,
    }
)

class BaseExampleSpider(CommonSpider):
    
    name = CUSTOM_SETTINGS.spider_name
    custom_settings = CUSTOM_SETTINGS.to_dict()
    
    def __init__(self, **kwargs):
        super().__init__(config=CUSTOM_SETTINGS)
        
    def _get_url(self) -> str:
        url = self.config.base_url
        if not url:
            raise ValueError("URL not found in config.")
        return url
    
    async def start(self):
        url = self._get_url()
        yield Request(url=url,
                      method='GET',
                      callback=self.parse,
                      dont_filter=True)
        
    def parse(self, response: Response):
        self.logger.info(f"Enter to parse with URL: {response.url}")
        error = self.error_response_status(response=response)
        if error:
            self.logger.error(f"Error response status: {response.status} for URL: {response.url}")
            return
        
        text = self.get_text_from_response(response=response)
        if text is None:
            self.logger.error(f"Failed to extract text from response for URL: {response.url}")
            return

        urls_loader = UrlsLoader(item=UrlDTO(), selector=Selector(text=text))
        urls_loader.add_xpath('urls', 
                              '//div[@class="row"]//div[@class="col-md-8"]//a[@class="tag"]/@href')
        
        urlDto: UrlDTO = urls_loader.load_item()
        if urlDto:
            for url in urlDto.get('urls', []):
                self.logger.info(f"Extracted URL: {url}")