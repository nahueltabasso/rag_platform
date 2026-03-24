from scrapy.item import Item, Field

class ItemDTO(Item):
    
    url = Field()
    title = Field()
    content = Field()
    date = Field()
    site = Field()
    chunks = Field()