"""LangGraph Agent"""

from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

import os
import getpass
import json
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

load_dotenv()

tools = [
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


# Build graph function
def build_graph(provider: str = "openai"):
    """Build the graph"""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
    else:
        raise ValueError("Invalid provider!")
    llm_with_tools = llm.bind_tools(tools)

    # Create the vector store and load data from the dataset
    def load_vector_store():
        embeddings = OpenAIEmbeddings()
        vector_store = InMemoryVectorStore(embeddings)
        if not os.path.exists("dataset/metadata.jsonl"):
            raise FileNotFoundError("Dataset not found.")
        with open("dataset/metadata.jsonl", "r", encoding="utf-8") as dataset:
            documents = []
            for item in dataset:
                entry = json.loads(item)
                content = f"Question: {entry['Question']}\nFinal answer: {entry['Final answer']}"
                doc = Document(
                    page_content=content, metadata={"source": entry["task_id"]}
                )
                documents.append(doc)
        vector_store.add_documents(documents)
        return vector_store

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Node
    def retriever(state: MessagesState):
        """Retriever node"""
        with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        sys_msg = SystemMessage(content=system_prompt)
        vector_store = load_vector_store()
        similar_questions = vector_store.similarity_search(
            state["messages"][0].content, k=3
        )
        if similar_questions:
            refs = "\n\n".join([doc.page_content for doc in similar_questions])
            example_msg = HumanMessage(content=f"Here are similar examples:\n\n{refs}")
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        else:
            return {"messages": [sys_msg] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


# For local runs
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    graph = build_graph(provider="openai")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
