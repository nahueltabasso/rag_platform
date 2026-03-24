from scrapy.exceptions import DropItem
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from crawler.src.item import ItemDTO
from crawler.src.base import CommonSpider
from crawler.src.settings import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
import httpx
import os
import chromadb
import hashlib

class ValidateFieldsPipeline:
    
    def process_item(self, item: ItemDTO, spider: CommonSpider):
        """ Process the item to save in a vector database.
        
        Args:
            item (ItemDTO): The item to process.
            spider (CommonSpider): The spider that scraped the item.
        """
        spider.logger.info(f"Processing item in ValidateFieldsPipeline")
        if item is None:
            raise DropItem("Item is None")

        url = item.get('url')
        site = item.get('site')
        title = item.get('title')
        content = item.get('content')

        if url is not None and not str(url).startswith("http"):
            raise DropItem(f"Invalid URL: {url}")
        if not site:
            raise DropItem("Missing site field")
        if not title:
            raise DropItem("Missing title field")
        if not content:
            raise DropItem("Missing content field")

        return item
    
class BaseProcessingPipeline:

    def _do_request(self, api_url: str, payload: dict, response_key: str, timeout: int = 1000):
        try:
            with httpx.Client() as client:
                response = client.post(api_url, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                return data.get(response_key)
        except httpx.RequestError as e:
            raise RuntimeError(f"An error ocurred while requesting {e.request.url!r}. Error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Error response {e.response.status_code} while requesting {e.request.url!r}. Error: {str(e)}")
        
    def generate_embedding(self, text: str):
        api_url = os.environ.get("EMBEDDING_API_URL", "")
        if api_url == '':
            raise ValueError("EMBEDDING_API_URL environment variable is not set.")
        return self._do_request(api_url, {"text": text}, 'embeddings')

    def generate_chunks(self, text: str):
        api_url = os.environ.get("CHUNKING_API_URL", "")
        if api_url == '':
            raise ValueError("CHUNKING_API_URL environment variable is not set.")
        body = {"text": text, "chunk_size": 4000, "chunk_overlap": 800}
        return self._do_request(api_url, body, 'chunks')
    
class ChunksPipeline(BaseProcessingPipeline):
    
    def process_item(self, item: ItemDTO, spider: CommonSpider):
        spider.logger.info(f"Processing item in ChunksPipeline")
        content = item.get('content')
        if content is None:
            raise DropItem("Missing content field for chunking")
        if content == "":
            raise DropItem("Empty content field for chunking")
        spider.logger.info(f"Generating chunks for content of length {len(content)}")
        chunks = self.generate_chunks(content)
        if not chunks:
            raise DropItem("Failed to generate chunks")
        spider.logger.info(f"Generated chunks: {len(chunks)}")
        item['chunks'] = chunks
        return item
        
class EmbeddingsPipeline(BaseProcessingPipeline):
    
    def process_item(self, item: ItemDTO, spider: CommonSpider):
        spider.logger.info(f"Processing item in EmbeddingsPipeline")
        chunks = item.get('chunks')
        if not chunks or not isinstance(chunks, list):
            raise DropItem("Missing or invalid chunks field for embedding generation")
        if len(chunks) == 0:
            raise DropItem("Empty chunks field for embedding generation")
        spider.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = [self.generate_embedding(chunk) for chunk in chunks]
        if not embeddings:
            raise DropItem("Failed to generate embeddings")
        spider.logger.info(f"Generated embeddings for {len(embeddings)} chunks")
        item['embeddings'] = embeddings
        return item

class SaveToChromaDBPipeline:
    
    def open_spider(self, spider: CommonSpider):
        spider.logger.info(f"Opening ChromaDB pipeline for spider: {spider.name}")
        
        self.persist_directory = CHROMA_DB_DIR
        if self.persist_directory == '':
            spider.logger.error("CHROMA_DB_DIR environment variable is not set.")
            raise ValueError("CHROMA_DB_DIR environment variable is not set.")
        self.collection_name = spider.config.collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        
        # Initialize ChromaDB Vector Store from Langchain
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        
    def close_spider(self, spider: CommonSpider):
        spider.logger.info(f"Closing ChromaDB pipeline for spider: {spider.name}")
        if self.client:
            self.client = None
        if self.vector_store:
            self.vector_store = None
        
    def process_item(self, item: ItemDTO, spider: CommonSpider):
        spider.logger.info(f"Processing item in SaveToChromaDBPipeline")
        url = item.get('url')
        site = item.get('site')
        title = item.get('title')
        date = item.get('date')
        chunks = item.get('chunks')
        
        if chunks is None or not isinstance(chunks, list) or len(chunks) == 0:
            spider.logger.error("Missing or invalid chunks field for saving to ChromaDB")
            raise DropItem("Missing or invalid chunks field for saving to ChromaDB")
        
        # Generates unique IDs to avoid duplicates in re-scrapes
        ids = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
        metadatas = [{
            "source": url,
            "title": title,
            "site": site,
            "date": date
        } for _ in chunks]
        
        # Insert texts from chunks (Chroma will generate embeddings internally now)
        self.vector_store.add_texts( # type: ignore
            texts=chunks,
            metadatas=metadatas,
            ids=ids
        )
        spider.logger.info(f"Embeddings generados e insertados para {len(chunks)} chunks")
        spider.logger.info(f"Chunks saved to ChromaDB - Collection: {self.collection_name}")
        return item