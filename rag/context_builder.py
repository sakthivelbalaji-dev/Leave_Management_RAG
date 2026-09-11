class ContextBuilder:
    def build(self, results):
        parts = []

        for item in results:
            parts.append(
                f"SOURCE: {item.get('source', 'unknown')}\n"
                f"CONTENT:\n{item.get('content', '')}"
            )

        context = "\n\n---\n\n".join(parts)
        print("\nFINAL CONTEXT SENT TO LLM:")
        print(context if context else "<EMPTY>")
        print("=" * 80)
        return context
