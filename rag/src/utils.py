
def format_documents(documents):
    formatted = []
    for i, doc in enumerate(documents):
        formatted.append(f"Fragmento {i+1}:\n{doc.metadata}\n{doc.page_content}\n")
    
    # for f in formatted:
    #     print(f"Formatted document:\n{f}\n{'-'*50}")
    return "\n\n".join(formatted)