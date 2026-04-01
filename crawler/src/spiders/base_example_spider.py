from scrapy.utils.project import get_project_settings

from crawler.src.base import CustomSettings


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
