# Solution Steps

1. Install required Python libraries (sentence-transformers, transformers, chromadb) if not already present.

2. Prepare your document corpus in JSONL format, ensuring each line is a dict with at least a 'content' field and optionally 'title', 'id', or other metadata fields.

3. Implement the chunk_and_embed.py script as shown:

4.   - Read documents from the JSONL file.

5.   - For each document, tokenize its content using a Huggingface tokenizer for the embedding model.

6.   - Split the content into 512-token chunks with 200-token overlap, decoding each chunk back into text.

7.   - Attach document metadata (everything except 'content'), plus start/end token indices to each chunk.

8.   - Embed each chunk using sentence-transformers, batch-wise.

9.   - Insert all chunks, their embeddings, and metadata into a Chroma collection created (or recreated) with cosine similarity ('hnsw:space': 'cosine').

10.   - Batch insertions for efficiency, e.g., every 128 chunks.

11. Implement retriever.py to perform semantic retrieval:

12.   - Connect to the same Chroma DB and collection.

13.   - Embed the user's query (using the same model).

14.   - Query the collection using the query embedding to fetch the top-5 (n_results=5) most similar chunks (with documents, metadatas, distances).

15.   - Return chunk text, metadata, and score (cosine similarity).

16. Test the retrieval script interactively with provided spot-check queries, visually confirm improvement.

17. Optionally, using held-out queries and manual keywords, compute recall@5 for before/after chunking strategy to verify retrieval quality improvement.

