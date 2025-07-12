from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class StoriesRAGTool:
    """
    A tool for retrieving relevant stories using a Retrieval-Augmented Generation (RAG) approach
    with HuggingFace embeddings.

    This tool transforms queries into vector embeddings using a HuggingFace model and retrieves
    the top-k most relevant story documents from a Chroma vector database.

    Attributes:
        embedding_model (str): HuggingFace embedding model name.
        vectordb_dir (str): Directory where Chroma vector DB is stored.
        k (int): Top-k documents to retrieve.
        vectordb (Chroma): The Chroma vector DB instance.
    """

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str) -> None:
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.k = k

        self.vectordb = Chroma(
            collection_name=collection_name,
            persist_directory=self.vectordb_dir,
            embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model)
        )

        print("✅ Loaded Chroma VectorDB for stories")
        print("🔢 Total vectors:", self.vectordb._collection.count(), "\n")


@tool
def lookup_stories(query: str) -> str:
    """
    Search the fictional stories and return the top relevant matches.

    Args:
        query (str): Natural language question.

    Returns:
        str: Matching story content.
    """
    rag_tool = StoriesRAGTool(
        embedding_model=TOOLS_CFG.stories_rag_embedding_model,
        vectordb_dir=TOOLS_CFG.stories_rag_vectordb_directory,          
        k=TOOLS_CFG.stories_rag_k,                                     
        collection_name=TOOLS_CFG.stories_rag_collection_name     
    )

    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    return "\n\n".join([doc.page_content for doc in docs])
