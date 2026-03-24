from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api_ai.src.settings import API_DESCRIPTION, API_NAME, API_VERSION, API_AUTHOR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, EMBEDDING_MODEL_NAME
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelService:

    def __init__(self,
                 embedding_model: OpenAIEmbeddings,
                 text_splitter: RecursiveCharacterTextSplitter):
        logger.info("ModelService Initializing")
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter

    def get_embedding(self, text: str):
        logger.info("Enter to get_embedding()")
        return self.embedding_model.embed_query(text)

    def get_chunks_from_text(self,
                             text: str,
                             chunk_size: int = 1000,
                             chunk_overlap: int = 200) -> List[str]:
        """Return chunks for `text` using a splitter configured per-call.

        Instead of mutating the shared splitter internals (not thread-safe),
        we create a new RecursiveCharacterTextSplitter per request preserving
        the original `separators` and `length_function`.
        """
        logger.info("Enter to get_chunks_from_text()")
        # preserve configuration from the base splitter
        separators = getattr(self.text_splitter, 'separators', None)
        length_fn = getattr(self.text_splitter, 'length_function', len)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
            separators=separators,
        )

        chunks = splitter.split_text(text)  # type: ignore
        return chunks

@asynccontextmanager  # type: ignore
async def lifespan(app: FastAPI):
    # load heavy resources once per process and attach to app.state
    try:
        logger.info("Load models...")
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", "", " ", ".", "!", "?", ","],
        )

        service = ModelService(embedding_model=embedding_model,
                               text_splitter=text_splitter)
        app.state.model_service = service
        yield
    finally:
        # optional cleanup hooks if embeddings or models need shutdown
        service = getattr(app.state, 'model_service', None)
        if service and hasattr(service, 'shutdown'):
            try:
                await service.shutdown()
            except Exception:
                pass

class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

class EmbeddingsRequest(BaseModel):
    text: str

app = FastAPI(title=API_NAME,
              version=API_VERSION,
              description=API_DESCRIPTION,
              author=API_AUTHOR,
              lifespan=lifespan)

def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service

@app.post('/get-embedding', status_code=200)
def embedding(req: EmbeddingsRequest, service: ModelService = Depends(get_model_service)):
    logger.info("Enter to embedding()")
    try:
        embedding = service.get_embedding(req.text)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"An error ocurred - details={e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/split-text')
def chunks(req: ChunkRequest, service: ModelService = Depends(get_model_service)):
    logger.info("Enter to chunks()")
    try:
        chunks = service.get_chunks_from_text(req.text,
                                            req.chunk_size, 
                                            req.chunk_overlap)
        return {"count": len(chunks), "chunks": chunks}
    except Exception as e:
        logger.error(f"An error ocurred - details={e}")        
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    logger.info("Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    