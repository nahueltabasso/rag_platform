from typing import List

from scrapy import Selector
from scrapy.http.response import Response
from scrapy.http.request import Request
from scrapy.utils.project import get_project_settings
from urllib.parse import urlparse
from datetime import datetime
from crawler.src.base import CommonSpider, CustomSettings
from crawler.src.item import ItemDTO, UrlDTO
from crawler.src.loader import DataLoader, UrlsLoader

CUSTOM_SETTINGS = CustomSettings(
    project_settings=get_project_settings().copy_to_dict(),
    spider_name='mercadolibre_faq_spider',
    collection_name='site_mercadolibre_faq',
    base_url='https://www.mercadolibre.com.ar/ayuda',
    api_chunks=None,
    api_embeddings=None,
    chunk_size=1200,
    chunk_overlap=200,
    error_output_filename='mercadolibre_faq.txt',
    item_pipelines={
        'crawler.src.pipeline.ValidateFieldsPipeline': 100,
        'crawler.src.pipeline.ChunksPipeline': 200,
        'crawler.src.pipeline.SaveToChromaDBPipeline': 300,
    }
)

class MercadoLibreFAQSpider(CommonSpider):
    
    name = CUSTOM_SETTINGS.spider_name
    custom_settings = CUSTOM_SETTINGS.to_dict()
    
    def __init__(self, **kwargs):
        super().__init__(config=CUSTOM_SETTINGS)
        self.sections_faq = [
            {'section': 'Compras', 'path': '/comprando_637'},
            {'section': 'Ventas', 'path': '/Vendiendo_643'},
            {'section': 'Cuenta', 'path': '/39873'}
        ]

    def _get_url(self) -> str:
        url = self.config.base_url
        if not url:
            raise ValueError("URL not found in config.")
        return url
    
    async def start(self):
        url = self._get_url()
        for section in self.sections_faq:
            full_url = url + section['path']
            yield Request(url=full_url,
                      method='GET',
                      callback=self.parse,
                      dont_filter=True,
                      cb_kwargs={'section': section['section']})
            
    def parse(self, response: Response, section: str):
        self.logger.info(f"Enter to parse with URL: {response.url} - Section: {section}")
        error = self.error_response_status(response=response)
        if error:
            self.logger.error(f"Error response status: {response.status} for URL: {response.url}")
            return
        
        self.logger.info(f"Successful response status: {response.status} for URL: {response.url}")
        
        text = self.get_text_from_response(response)
        if text is None:
            self.logger.error(f"Failed to extract text from response for URL: {response.url}")
            return
        
        urls: List[str] = self._get_urls_from_response(content=text) # type: ignore
        if urls:
            yield from self._yield_requests(urls=urls, section=section)
        
    def parse_item(self, response: Response, section: str):
        self.logger.info(f"Enter to parse level two with URL: {response.url} - Section: {section}")
        error = self.error_response_status(response=response)
        if error:
            self.logger.error(f"Error response status: {response.status} for URL: {response.url}")
            return
        
        self.logger.info(f"Successful response status: {response.status} for URL: {response.url}")
        
        text = self.get_text_from_response(response)
        if text is None:
            self.logger.error(f"Failed to extract text from response for URL: {response.url}")
            return
        
        # Check if the URL is a FAQ or a section with more FAQs
        urls: List[str] = self._get_urls_from_response(content=text) # type: ignore
        if urls:
            yield from self._yield_requests(urls=urls, section=section)
        else:
            loader = DataLoader(item=ItemDTO(),
                        selector=Selector(text=text))

            loader.add_value('url', response.url)
            loader.add_value('site', urlparse(self.config.base_url).netloc)
            loader.add_value('date', datetime.now().isoformat())
            loader.add_value('title', section)
            loader.add_xpath('title',
                             '//div[@class="cx-peach-content__title-container"]//h1[@class="cx-peach-content__title"]/text()')
            loader.add_xpath('content',
                             '//div[@class="cx-peach-content__data"]//text()[normalize-space()]')   
            
            itemDto: ItemDTO = loader.load_item()
            if itemDto:
                self.logger.info(f"Loaded item: {itemDto}")
                yield itemDto
        
    def _get_urls_from_response(self, content: str) -> List[str] | None:
        urls_loader = UrlsLoader(item=UrlDTO(), selector=Selector(text=content))
        urls_loader.add_xpath('urls', '//div[@class="cx-contents-list"]//li//a/@href')
        urlsDTO: UrlDTO = urls_loader.load_item()
        
        if urlsDTO:
            urls = urlsDTO.get('urls', [])
            self.logger.info(f"Extracted {len(urls)} URLs from the response")
            return urls
        return None
    
    def _yield_requests(self, urls: List[str], section: str):
        for url in urls:
            if url.endswith('/Vendiendo_643') and section == 'Cuenta':
                print("Entra aca porque es la seccion cuentas, y no tiene que ir a la seccion ventas")
                continue
            self.logger.info(f"Yielding URL: {url}")
            yield Request(url=url,
                        method='GET',
                        callback=self.parse_item,
                        dont_filter=True,
                        cb_kwargs={'section': section})
