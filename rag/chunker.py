import re


class TextChunker:

    def __init__(
        self,
        chunk_size=600,
        overlap=100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ============================================================
    # MAIN CHUNK FUNCTION
    # ============================================================

    def chunk_documents(self, documents):

        chunks = []
        chunk_id = 0

        for doc in documents:

            text = doc.get(
                "content",
                ""
            ).strip()

            if not text:
                continue

            # ----------------------------------------------------
            # Split document into policy sections
            # ----------------------------------------------------

            sections = self._split_into_sections(text)

            for section in sections:

                section_chunks = self._split_large_section(
                    section
                )

                for content in section_chunks:

                    content = content.strip()

                    if not content:
                        continue

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "source": doc.get(
                                "source",
                                "unknown"
                            ),
                            "content": content,
                        }
                    )

                    chunk_id += 1

        print(
            f"[CHUNKER] Created {len(chunks)} chunks."
        )

        return chunks

    # ============================================================
    # SECTION SPLITTER
    # ============================================================

    def _split_into_sections(self, text):

        lines = text.splitlines()

        sections = []

        current_section = []

        for line in lines:

            stripped = line.strip()

            # ----------------------------------------------------
            # Detect policy section headings
            #
            # Examples:
            #
            # 6. LATE ARRIVAL
            # 10. GENERAL LEAVE POLICY
            # 28. FREQUENTLY ASKED QUESTIONS
            # ----------------------------------------------------

            is_heading = bool(
                re.match(
                    r"^\d+\.\s+.+",
                    stripped
                )
            )

            if is_heading:

                # Save previous section
                if current_section:

                    sections.append(
                        "\n".join(
                            current_section
                        ).strip()
                    )

                current_section = [
                    stripped
                ]

            else:

                current_section.append(
                    line
                )

        # Add final section
        if current_section:

            sections.append(
                "\n".join(
                    current_section
                ).strip()
            )

        return sections

    # ============================================================
    # SPLIT LARGE SECTION
    # ============================================================

    def _split_large_section(
        self,
        section,
    ):

        # Small enough → keep entire section together
        if len(section) <= self.chunk_size:

            return [
                section
            ]

        lines = section.splitlines()
        heading = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]).strip()
        prefix = f"{heading}\n\n" if heading else ""
        body_size = max(1, self.chunk_size - len(prefix))
        chunks = []
        start = 0

        while start < len(body):

            end = min(
                start + body_size,
                len(body)
            )

            chunk = prefix + body[
                start:end
            ].strip()

            if chunk:

                chunks.append(
                    chunk
                )

            if end >= len(body):

                break

            start = max(
                0,
                end - min(self.overlap, body_size - 1)
            )

        return chunks