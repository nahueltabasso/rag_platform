# DB configuration
CHROMA_DB_PATH = "/Users/nahueltabasso/Documents/Python/rag_platform/chroma_db"

# Models configuration
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# Retriever configuration
SEARCH_TYPE = "mmr"  # Maximal Marginal Relevance
MMR_DIVERSITY_LAMBDA = 0.7 # Higher values give more weight to diversity, lower values give more weight to relevance
MMR_FETCH_K = 20 # Number of initial candidates to fetch for MMR before re-ranking
SEARCH_K = 5 # Number of final results to return after MMR re-ranking


ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.75
