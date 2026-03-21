import re
import urllib.request
import urllib.parse
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup
import concurrent.futures
from rich.console import Console
from src.tools.query_refiner import refine_query

console = Console()

CURRENT_YEAR = datetime.now().year

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SKIP_DOMAINS = {
    "wikipedia.org", "yahoo.com", "msn.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com"
}

_search_cache = {}  # prevent duplicate searches in same session


def _is_skippable(url: str) -> bool:
    domain = urlparse(url).netloc.replace("www.", "")
    return any(skip in domain for skip in SKIP_DOMAINS)


def _fetch_html(url: str, timeout: int = 5) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return re.sub(r'\n{3,}', '\n\n', cleaned)


def _scrape(url: str) -> str:
    if _is_skippable(url):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    try:
        html = _fetch_html(url, timeout=5)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "aside", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else "No title"
        text = _clean_text(soup.get_text(separator="\n"))[:4000]
        return f"### {title}\nURL: {url}\n\n{text}"
    except Exception:
        return None


def _search(query: str, max_results: int = 5) -> list:
    if str(CURRENT_YEAR) not in query and str(CURRENT_YEAR - 1) not in query:
        query = f"{query} {CURRENT_YEAR}"

    encoded = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    html = _fetch_html(url, timeout=6)
    soup = BeautifulSoup(html, "html.parser")

    results = []
    for result in soup.select(".result")[:max_results]:
        title_tag = result.select_one(".result__a")
        snippet_tag = result.select_one(".result__snippet")
        if not title_tag:
            continue
        href = title_tag.get("href", "")
        if "uddg=" in href:
            href = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
        if _is_skippable(href):
            continue
        results.append({
            "title": title_tag.get_text(strip=True),
            "url": href,
            "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
        })
    return results


def web_search_tool(query: str) -> str:
    refined = refine_query(query)

    cache_key = refined.strip().lower()
    # Block duplicate queries — tell LLM it already has the answer
    if cache_key in _search_cache:
        console.print(f"[yellow][web_search] Blocked duplicate query: '{query}'[/yellow]")
        return (
            f"⚠️ You already searched for '{query}' and received results. "
            f"Do NOT search again. Use the information already provided to answer the user.\n\n"
            f"{_search_cache[cache_key]}"
        )

    with console.status(f"[bold green]Searching: '{query}'...[/bold green]", spinner="dots"):
        try:
            results = _search(query, max_results=5)
        except Exception as e:
            return f"Search failed: {str(e)}"

        if not results:
            return f"No results found for: {query}"

        # Check if snippets alone are rich enough (saves scraping time)
        total_snippet_text = " ".join(r["snippet"] for r in results)
        needs_scrape = len(total_snippet_text) < 300

        scraped = None
        if needs_scrape:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_scrape, results[0]["url"])
                try:
                    scraped = future.result(timeout=6)
                except Exception:
                    scraped = None

    # Build output
    output = f"# Search Results: {query}\n\n"
    output += "## Snippets (answer from these first)\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. {r['title']}\n"
        output += f"   URL: {r['url']}\n"
        output += f"   {r['snippet']}\n\n"

    if scraped:
        output += "## Full Content\n\n"
        output += scraped

    output += (
        "\n\n---\n"
        """ 
 ⚠️ INSTRUCTION: Answer the user now using the above results.        
## ⚠️ BEFORE YOU ANSWER — RELEVANCE CHECK REQUIRED

Original user query: "{query}"

You MUST follow these steps before responding:

1. EXAMINE: Do the results above directly answer "{query}"?
   - Look for specific names, scores, dates, facts the user asked for
   - Do NOT just summarize what was found

2. IF RESULTS ARE RELEVANT:
   - Answer the user directly using the found information
   - Cite which result you used

3. IF RESULTS ARE NOT RELEVANT or only partially match:
   - Clearly tell the user: "I couldn't find a direct answer for: {query}"
   - Briefly explain what the results contained instead
   - Suggest a refined search query the user could try

4. NEVER fabricate or assume information not present in the results
5. NEVER call web_search again for the same query
        """
    )

    output = output[:10000]
    _search_cache[cache_key] = output  # cache it
    return output