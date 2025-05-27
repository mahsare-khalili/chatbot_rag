import requests

def query_duckduckgo(query: str) -> str:
    """
    Returns abstract from DuckDuckGo Instant Answer API.
    """
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    try:
        res = requests.get(url)
        data = res.json()
        return data.get("AbstractText") or "No instant answer from DuckDuckGo."
    except Exception:
        return "DuckDuckGo fallback failed."

