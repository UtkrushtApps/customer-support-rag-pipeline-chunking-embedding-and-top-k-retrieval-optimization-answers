# retriever.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict

# Settings—must match chunk_and_embed.py
CHROMA_COLLECTION_NAME = "support_docs_optimized"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

def retrieve_topk_chunks(query: str, k: int = 5) -> List[Dict]:
    """
    Given a query string, retrieve top-k most relevant chunks from the collection.
    Returns a list of dicts: {document, metadata, score}
    """
    embedding = embedding_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    out = []
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results["distances"][0]):
        # For cosine, larger is better (Chroma returns similarity for cosine)
        out.append({
            "document": doc,
            "metadata": meta,
            "score": dist
        })
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        query = sys.argv[1]
        results = retrieve_topk_chunks(query)
        print("Top-5 retrieved chunks:")
        for idx, res in enumerate(results):
            doc = res['document']
            meta = res['metadata']
            score = res['score']
            print(f"Rank {idx+1}: ")
            if 'title' in meta:
                print(f"  Title: {meta['title']}")
            print(f"  Chunk (first 200 chars): {doc[:200].replace('\n',' ')}")
            print(f"  Score: {score:.4f}")
    else:
        print("Usage: python retriever.py <query>")
