class ContextBuilder:
    def build(self, results):
        parts = []

        for item in results:
            parts.append(
                f"SOURCE: {item.get('source', 'unknown')}\n"
                f"CONTENT:\n{item.get('content', '')}"
            )

        return "\n\n---\n\n".join(parts)
