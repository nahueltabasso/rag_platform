from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import os

# Test code to check the contents of the ChromaDB collection after running the spider and pipelines
COLLECTION_NAME = "wwII_wikipedia"
client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DB_DIR", ""))
collection = client.get_collection(name=COLLECTION_NAME)

total_chunks = collection.count()
print(f"Total de chunks guardados: {total_chunks}")

results = collection.get(include=["documents"])

for i, doc in enumerate(results["documents"], start=1): # type: ignore
    print(f"\nChunk {i}:\n{doc}")
    
    
# Test query to check if the chunks are retrievable
QUESTION = "¿Cuando empezo la Segunda Guerra Mundial?"

embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)

results = vector_store.similarity_search(QUESTION, k=5)

print(f"Resultados encontrados: {len(results)}")
for i, doc in enumerate(results, start=1):
    print(f"\nResultado {i}")
    print("Metadata:", doc.metadata)
    print("Contenido:", doc.page_content)

    
    