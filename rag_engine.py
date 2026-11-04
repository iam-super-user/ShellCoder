import os
import chromadb

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR      = os.path.join(BASE_DIR, "vault_data")
DB_PATH        = os.path.join(VAULT_DIR, "chroma_db")

_db         = None
_collection = None

def get_db():
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(path=DB_PATH)
    return _db

def get_collection():
    global _collection
    if _collection is None:
        _collection = get_db().get_or_create_collection(name="arsenal_docs")
    return _collection

def search_vault(query_text: str, top_k: int = 5, threshold: float = 1.2) -> str:
    """
    Searches the local RAG vault for technical context matching the query.
    Returns formatted string of results.
    """
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k * 2,
            include=["documents", "distances", "metadatas"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant data found in the vault."
            
        docs      = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        filtered = [
            (doc, dist, meta) for doc, dist, meta in zip(docs, distances, metadatas)
            if dist < threshold
        ][:top_k]

        if not filtered:
            return "No data found matching the required relevance threshold."

        context_parts = []
        for doc, dist, meta in filtered:
            filename  = meta.get('filename', os.path.basename(meta.get('source', 'unknown')))
            context_parts.append(f"[SOURCE: {filename}]\n{doc}")

        return "\n\n--- [VAULT CHUNK] ---\n\n".join(context_parts)
    except Exception as e:
        return f"Error searching vault: {str(e)}"
