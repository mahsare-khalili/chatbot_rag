# utils/chunking.py

def chunk_text(text: str, size: int = 500) -> list[str]:
    """
    Split `text` into chunks of at most `size` characters,
    so embeddings stay within a reasonable length.
    """
    return [text[i : i + size] for i in range(0, len(text), size)]
