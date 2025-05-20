"""LangGraph Agent"""

import os
import json
import getpass
from dotenv import load_dotenv

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama

from tools.math.multiply import multiply
from tools.math.add import add
from tools.math.subtract import subtract
from tools.math.divide import divide
from tools.math.modulus import modulus
from tools.math.power import power
from tools.math.square_root import square_root

from tools.search.arxiv_search import arxiv_search
from tools.search.web_search import web_search
from tools.search.wiki_search import wiki_search

from tools.file.analyze_csv_file import analyze_csv_file
from tools.file.analyze_excel_file import analyze_excel_file
from tools.file.analyze_image import analyze_image
from tools.file.download_file_from_url import download_file_from_url
from tools.file.save_content_to_file import save_content_to_file

# --- Load environment variables ---
load_dotenv()

# --- Constants ---
DATASET_PATH = "dataset/metadata.jsonl"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
TOOLS = [
    add,
    subtract,
    multiply,
    divide,
    modulus,
    power,
    square_root,
    web_search,
    wiki_search,
    arxiv_search,
    analyze_csv_file,
    analyze_excel_file,
    analyze_image,
    download_file_from_url,
    save_content_to_file,
]


def load_vector_store() -> InMemoryVectorStore:
    """Load vector store with dataset examples."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")
    embeddings = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embeddings)
    documents = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            content = (
                f"Question: {entry['Question']}\nFinal answer: {entry['Final answer']}"
            )
            doc = Document(page_content=content, metadata={"source": entry["task_id"]})
            documents.append(doc)
    vector_store.add_documents(documents)
    return vector_store


def get_llm(provider: str):
    """Get LLM instance based on provider."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")
        return ChatOpenAI(model="gpt-4.1", temperature=0)
    elif provider == "ollama":
        return ChatOllama(model="llama3.2", temperature=0)
    else:
        raise ValueError("Unsupported provider: choose 'openai' or 'ollama'")


def load_system_prompt() -> SystemMessage:
    """Load system prompt from file."""
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        raise FileNotFoundError(f"System prompt not found at {SYSTEM_PROMPT_PATH}.")
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return SystemMessage(content=f.read())


def build_graph(provider: str = "openai"):
    """Build and compile the LangGraph agent."""
    llm = get_llm(provider).bind_tools(TOOLS)
    vector_store = load_vector_store()
    system_msg = load_system_prompt()

    def retriever(state: MessagesState):
        """Retrieve similar examples based on user query."""
        query = state["messages"][0].content
        similar = vector_store.similarity_search(query, k=3)
        if similar:
            refs = "\n\n".join(doc.page_content for doc in similar)
            example_msg = HumanMessage(content=f"Here are similar examples:\n\n{refs}")
            return {"messages": [system_msg] + state["messages"] + [example_msg]}
        return {"messages": [system_msg] + state["messages"]}

    def assistant(state: MessagesState):
        """Call LLM to generate next message."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # --- Build graph ---
    graph = StateGraph(MessagesState)
    graph.add_node("retriever", retriever)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


def run_agent(query: str, provider: str = "openai"):
    """Run the agent on a given query."""
    graph = build_graph(provider)
    messages = [HumanMessage(content=query)]
    result = graph.invoke({"messages": messages})
    for msg in result["messages"]:
        msg.pretty_print()


# --- Run locally ---
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    run_agent(user_query)
