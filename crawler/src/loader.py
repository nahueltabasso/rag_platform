from itemloaders import ItemLoader
from itemloaders.processors import Identity, MapCompose, TakeFirst, Join
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
def clean_text(value: str) -> str:
    if not value:
        return ""

    value = value.replace("\xa0", " ")
    value = " ".join(value.split())
    return value.strip()

class DataLoader(ItemLoader):
    
    default_output_processor = TakeFirst()
    title_out = Join(" - ")
    content_in = MapCompose(clean_text)
    content_out = Join("")
    
class UrlsLoader(ItemLoader):
    
    default_output_processor = Identity()
    
    