from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}
