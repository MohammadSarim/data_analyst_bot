import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


class PrepareVectorDB:
    """
    Prepares a vector database by:
    - Loading and splitting PDF documents.
    - Generating embeddings using HuggingFace models.
    - Storing vectors in a Chroma database.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embedding_model: str,
                 vectordb_dir: str,
                 collection_name: str
                 ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name

    def path_maker(self, file_name: str, doc_dir):
        """Creates full file path."""
        return os.path.join(here(doc_dir), file_name)

    def run(self):
        """
        Main logic:
        - Loads PDFs.
        - Splits into chunks.
        - Embeds and stores into Chroma vectorDB.
        """
        vectordb_path = here(self.vectordb_dir)

        if not os.path.exists(vectordb_path):
            os.makedirs(vectordb_path)
            print(f"Directory '{self.vectordb_dir}' was created.")

            # Load PDFs
            file_list = os.listdir(here(self.doc_dir))
            docs = [PyPDFLoader(self.path_maker(fn, self.doc_dir)).load_and_split() for fn in file_list]
            docs_list = [item for sublist in docs for item in sublist]

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Embed with HuggingFace
            embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model)

            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=embedding_model,
                persist_directory=str(vectordb_path)
            )

            print("✅ VectorDB created and saved.")
            print("🔢 Total vectors:", vectordb._collection.count(), "\n")
        else:
            print(f"📁 VectorDB directory '{self.vectordb_dir}' already exists.")


if __name__ == "__main__":
    load_dotenv()

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    # --------- Run for Swiss Airline Policy ---------
    # swiss_cfg = app_config["swiss_airline_policy_rag"]
    # prepare_db_instance = PrepareVectorDB(
    #     doc_dir=swiss_cfg["unstructured_docs"],
    #     chunk_size=swiss_cfg["chunk_size"],
    #     chunk_overlap=swiss_cfg["chunk_overlap"],
    #     embedding_model=swiss_cfg["embedding_model"],
    #     vectordb_dir=swiss_cfg["vectordb"],
    #     collection_name=swiss_cfg["collection_name"]
    # )
    # prepare_db_instance.run()

    # --------- Run for Stories ---------
    stories_cfg = app_config["stories_rag"]
    prepare_db_instance = PrepareVectorDB(
        doc_dir=stories_cfg["unstructured_docs"],
        chunk_size=stories_cfg["chunk_size"],
        chunk_overlap=stories_cfg["chunk_overlap"],
        embedding_model=stories_cfg["embedding_model"],
        vectordb_dir=stories_cfg["vectordb"],
        collection_name=stories_cfg["collection_name"]
    )
    prepare_db_instance.run()
