class TextChunker:
    def __init__(self, chunk_size=900, overlap=120):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents):
        chunks = []
        chunk_id = 0

        for doc in documents:
            text = doc["content"]
            start = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                content = text[start:end].strip()

                if content:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": doc["source"],
                        "content": content,
                    })
                    chunk_id += 1

                if end >= len(text):
                    break

                start = end - self.overlap

        return chunks
