# chunk_and_embed.py

import os
import uuid
import json
from typing import List, Dict, Generator
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Settings
CHROMA_COLLECTION_NAME = "support_docs_optimized"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db") # or as needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # as example, match pre-load
TOKENIZER_NAME = EMBEDDING_MODEL_NAME
CHUNK_SIZE = 512
CHUNK_OVERLAP = 200


def read_documents(jsonl_path: str) -> List[Dict]:
    """
    Reads documents from a JSONL file. Each line is a dict with at least 'content', may have 'title', 'id', etc.
    """
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    return documents

def chunk_document(doc: Dict, tokenizer, chunk_size: int, overlap: int) -> Generator[Dict, None, None]:
    """
    Splits the document content into chunks of given token length with overlap.
    """
    content = doc['content']
    tokens = tokenizer(content, return_attention_mask=False, return_token_type_ids=False, truncation=False)['input_ids']
    num_tokens = len(tokens)
    start = 0
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        chunk_token_ids = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        yield {
            'chunk_text': chunk_text,
            'start_token': start,
            'end_token': end,
            'metadata': {
                k: v for k, v in doc.items() if k != 'content'
            }
        }
        if end == num_tokens:
            break
        start += (chunk_size - overlap)


def add_chunks_to_chroma(chunks: List[Dict], embedding_model: SentenceTransformer, chroma_collection):
    """
    Embeds and upserts the chunks (with metadata) into Chroma DB.
    """
    texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
    # UIDs per chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            **chunk['metadata'],
            'start_token': chunk['start_token'],
            'end_token': chunk['end_token']
        } for chunk in chunks
    ]
    chroma_collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def main(doc_jsonl_path: str):
    # Prepare embedding model & tokenizer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Chroma connection
    chroma_client = PersistentClient(path=CHROMA_DB_PATH)
    if CHROMA_COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME, 
        metadata={"hnsw:space": "cosine"}
    )
    
    # Read and process documents
    documents = read_documents(doc_jsonl_path)

    all_chunks = []
    for doc in documents:
        for chunk in chunk_document(doc, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append(chunk)
            if len(all_chunks) >= 128:
                add_chunks_to_chroma(all_chunks, embedding_model, collection)
                all_chunks = []
    if all_chunks:
        add_chunks_to_chroma(all_chunks, embedding_model, collection)

    print(f"Done. {collection.count()} chunks stored in Chroma collection '{CHROMA_COLLECTION_NAME}'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python chunk_and_embed.py <documents.jsonl>")
    else:
        main(sys.argv[1])
