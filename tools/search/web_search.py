from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def web_search(input: str) -> dict:
    """Perform a web search and return maximum 3 results."""
    search_docs = TavilySearchResults(max_results=3).invoke(input)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc["url"]}" title="{doc["title"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}
