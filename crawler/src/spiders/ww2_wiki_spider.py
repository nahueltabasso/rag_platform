from scrapy.http.response import Response
from scrapy.http.request import Request
from scrapy.utils.project import get_project_settings
from scrapy.selector import Selector
from urllib.parse import urlparse
from datetime import datetime
from crawler.src.base import CommonSpider, CustomSettings
from crawler.src.loader import DataLoader
from crawler.src.item import ItemDTO

CUSTOM_SETTINGS = CustomSettings(
    project_settings=get_project_settings().copy_to_dict(),
    spider_name='wwii_wikipedia_spider',
    collection_name='site_wwII_wikipedia',
    base_url='https://es.wikipedia.org/wiki/Segunda_Guerra_Mundial',
    api_chunks=None,
    api_embeddings=None,
    chunk_size=4000,
    chunk_overlap=800,
    error_output_filename='ww2_wikipedia.txt',
    item_pipelines={
        'crawler.src.pipeline.ValidateFieldsPipeline': 100,
        'crawler.src.pipeline.ChunksPipeline': 200,
        # 'crawler.src.pipeline.EmbeddingsPipeline': 300,
        'crawler.src.pipeline.SaveToChromaDBPipeline': 300,
    }
)

class WW2WikipediaSpider(CommonSpider):
    
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
        
        self.logger.info(f"Successful response status: {response.status} for URL: {response.url}")
        text = self.get_text_from_response(response)
        if text is None:
            self.logger.error(f"Failed to extract text from response for URL: {response.url}")
            return
        
        loader = DataLoader(item=ItemDTO(),
                            selector=Selector(text=text))

        loader.add_value('url', response.url)
        loader.add_value('site', urlparse(self.config.base_url).netloc)
        loader.add_value('date', datetime.now().isoformat())
        loader.add_xpath('title', '//h1[@id="firstHeading"]//span/text()')
        loader.add_xpath('content', 
                         '//div[contains(@class, "mw-parser-output")]/p//text()[not(ancestor::sup or ancestor::b)] | '
                         '//div[contains(@class, "mw-parser-output")]/p//b/text()[not(ancestor::sup)]')
        
        itemDto: ItemDTO = loader.load_item()
        if itemDto:
            self.logger.info(f"Loaded item: {itemDto}")
            yield itemDto
