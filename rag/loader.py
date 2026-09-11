from pathlib import Path


class KnowledgeBaseLoader:

    def __init__(self):
        # --------------------------------------------------------
        # PROJECT ROOT
        # --------------------------------------------------------
        #
        # loader.py is:
        # B:\leave_management_ai_full_code\rag\loader.py
        #
        # .parent      -> rag
        # .parent.parent -> leave_management_ai_full_code
        #
        # Therefore:
        # project_root / "knowledge_base"
        # points to:
        # B:\leave_management_ai_full_code\knowledge_base
        # --------------------------------------------------------

        project_root = Path(__file__).resolve().parent.parent

        self.knowledge_base_dir = (
            project_root / "knowledge_base"
        )

        print(
            "\n[KNOWLEDGE BASE PATH]"
        )

        print(
            self.knowledge_base_dir
        )

    # ============================================================
    # LOAD DOCUMENTS
    # ============================================================

    def load_documents(self):

        documents = []

        # --------------------------------------------------------
        # CHECK FOLDER
        # --------------------------------------------------------

        if not self.knowledge_base_dir.exists():

            raise RuntimeError(
                "Knowledge base directory does not exist:\n"
                f"{self.knowledge_base_dir}"
            )

        # --------------------------------------------------------
        # FIND TXT FILES
        # --------------------------------------------------------

        txt_files = sorted(
            self.knowledge_base_dir.glob("*.txt")
        )

        print(
            f"[KNOWLEDGE BASE] "
            f"Found {len(txt_files)} .txt file(s)"
        )

        # --------------------------------------------------------
        # LOAD EACH DOCUMENT
        # --------------------------------------------------------

        for path in txt_files:

            try:

                text = path.read_text(
                    encoding="utf-8"
                ).strip()

            except Exception as exc:

                print(
                    "[KNOWLEDGE BASE ERROR]",
                    path,
                    type(exc).__name__,
                    str(exc),
                )

                continue

            if not text:
                continue

            documents.append(
                {
                    "source": path.name,
                    "content": text,
                }
            )

            print(
                f"[KNOWLEDGE BASE] Loaded: "
                f"{path.name}"
            )

            print(
                f"[KNOWLEDGE BASE] Characters: "
                f"{len(text)}"
            )

        # --------------------------------------------------------
        # FINAL CHECK
        # --------------------------------------------------------

        if not documents:

            raise RuntimeError(
                "No usable .txt documents found in:\n"
                f"{self.knowledge_base_dir}"
            )

        print(
            f"[KNOWLEDGE BASE] "
            f"Total documents loaded: "
            f"{len(documents)}"
        )

        return documents