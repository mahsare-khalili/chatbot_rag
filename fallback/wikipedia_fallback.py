import wikipedia

def query_wikipedia(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception:
        return "Wikipedia could not find an answer."
