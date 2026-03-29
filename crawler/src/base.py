from __future__ import annotations
from scrapy import Spider
from scrapy.http.response import Response
from twisted.internet.defer import Deferred
from typing import Optional, cast

class CommonSpider(Spider):
    
    name = "base_rag_spider"
    
    def __init__(self, config: CustomSettings): # type: ignore
        if not config:
            raise ValueError("Config cannot be empty.")
        
        if not isinstance(config, CustomSettings):
            raise ValueError("Config must be an instance of CustomSettings.")

        self.config = config        
        super().__init__(name=self.config.spider_name)
        
    def parse(self, response: Response):
        pass
    
    @staticmethod
    def close(spider: Spider, reason: str):
        closed = getattr(spider, 'closed', None)
        class_ = getattr(spider, '__class__', None)
        name = ''
        if class_:
            name = getattr(class_, 'name', '')
        spider.logger.info(f"Spider [{name}] closed - Reason: {reason}")
        
        if callable(closed):
            return cast("Deferred[None] | None", closed(reason))
        return None
    
    def error_response_status(self, response) -> bool:
        resp = False
        if response:
            if isinstance(response, Response):
                self.logger.info(f"Checking response status for URL: {response.url}")
                self.logger.info(f"Response status: {response.status}")
                if response.status != 200:
                    resp = True
        return resp
    
    def get_text_from_response(self, response: Response) -> Optional[str]:
        if response and isinstance(response, Response):
            return response.text
        return None
    
class CustomSettings():
    
    def __init__(self, 
                 project_settings: dict,
                 spider_name: str,
                 item_pipelines: dict | None,
                 collection_name: str,
                 base_url: str,
                 api_chunks: str | None,
                 api_embeddings: str | None,
                 error_output_filename: str = '') -> None:
        """Constructor

        Args:
            project_settings (dict): Project Settings.
            spider_name (str): Spider Name.
            item_pipelines (dict | None): Item Pipelines.
            collection_name (str): Collection Name.
            base_url (str): Base URL.
            api_chunks (str | None): API Chunks.
            api_embeddings (str | None): API Embeddings.
        """
        
        for (key, value) in project_settings.items():
            setattr(self, key, value)
            
        self.spider_name = spider_name
        self.item_pipelines = item_pipelines
        self.collection_name = collection_name
        self.base_url = base_url
        self.api_chunks = api_chunks
        self.api_embeddings = api_embeddings
        self.error_output_filename = error_output_filename
    
    def to_dict(self) -> dict:
        # Export only instance data (avoid methods and internals)
        return {k.upper(): v for k, v in self.__dict__.items()}