
from typing import Dict, List


def format_documents(documents):
    formatted = []
    for i, doc in enumerate(documents):
        formatted.append(f"Fragmento {i+1}:\n{doc.metadata}\n{doc.page_content}\n")
    
    # for f in formatted:
    #     print(f"Formatted document:\n{f}\n{'-'*50}")
    return "\n\n".join(formatted)

def _serialize_context_docs(documents: List) -> List[Dict[str, str]]:
    serialized_docs = []
    for index, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        serialized_docs.append(
            {
                "chunk": str(index),
                "url": str(metadata.get("source") or metadata.get("url") or "Sin fuente"),
                "content": str(getattr(doc, "page_content", "")),
            }
        )
    return serialized_docs