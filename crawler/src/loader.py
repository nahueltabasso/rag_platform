from typing import Any

from itemloaders import ItemLoader
from itemloaders.processors import TakeFirst, Join
from scrapy.loader import ItemLoader

# class WikipediaContentProcessor:
    
#     def __init__(self):
#         pass
    
#     def __call__(self, values: Any) -> Any:
#         with open('values.txt', 'w') as f:
#             f.write(str(type(values)))
#             f.write(str(len(values)))
#             for value in values:
#                 f.write(f"{value}\n")


class DataLoader(ItemLoader):
    
    default_output_processor = TakeFirst()
    content_out = Join("")