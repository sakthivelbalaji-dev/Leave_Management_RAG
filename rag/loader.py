from pathlib import Path


class KnowledgeBaseLoader:
    def __init__(self, knowledge_base_dir="knowledge_base"):
        self.knowledge_base_dir = Path(knowledge_base_dir)

    def load_documents(self):
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        documents = []

        for path in sorted(self.knowledge_base_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append({
                    "source": str(path),
                    "content": text,
                })

        return documents
