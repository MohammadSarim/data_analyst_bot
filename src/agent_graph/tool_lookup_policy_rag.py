from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class SwissAirlinePolicyRAGTool:
    """
    A tool for retrieving relevant Swiss Airline policy documents using a 
    Retrieval-Augmented Generation (RAG) approach with HuggingFace embeddings.

    Attributes:
        embedding_model (str): HuggingFace model used for generating vector embeddings.
        vectordb_dir (str): Directory where the Chroma vector DB is stored.
        k (int): Number of top relevant documents to retrieve.
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

        print("✅ Loaded Chroma VectorDB")
        print("🔢 Total vectors:", self.vectordb._collection.count(), "\n")


@tool
def lookup_swiss_airline_policy(query: str) -> str:
    """
    Consult the Swiss airline company policy documents to check what's allowed.

    Args:
        query (str): The user query.

    Returns:
        str: The retrieved content from the policy documents.
    """
    rag_tool = SwissAirlinePolicyRAGTool(
        embedding_model=TOOLS_CFG.policy_rag_embedding_model,              
        vectordb_dir=TOOLS_CFG.policy_rag_vectordb_directory,              
        k=TOOLS_CFG.policy_rag_k,                                       
        collection_name=TOOLS_CFG.policy_rag_collection_name           
    )

    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    return "\n\n".join([doc.page_content for doc in docs])
