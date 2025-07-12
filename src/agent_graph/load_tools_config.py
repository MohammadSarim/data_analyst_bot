import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

class LoadToolsConfig:
    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Load common environment variables
        #os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # ✅ Add Groq key if not already loaded

        # 🌐 Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = float(app_config["primary_agent"]["llm_temperature"])

        # 🌍 Internet Search config
        self.tavily_search_max_results = int(app_config["tavily_search_api"]["tavily_search_max_results"])

        # ✈️ Swiss Airline Policy RAG configs
        policy_cfg = app_config["swiss_airline_policy_rag"]
        self.policy_rag_llm = policy_cfg["llm"]
        self.policy_rag_llm_temperature = float(policy_cfg["llm_temperature"])
        self.policy_rag_embedding_model = policy_cfg["embedding_model"]
        self.policy_rag_vectordb_directory = str(here(policy_cfg["vectordb"]))
        self.policy_rag_unstructured_docs_directory = str(here(policy_cfg["unstructured_docs"]))
        self.policy_rag_k = int(policy_cfg["k"])
        self.policy_rag_chunk_size = int(policy_cfg["chunk_size"])
        self.policy_rag_chunk_overlap = int(policy_cfg["chunk_overlap"])
        self.policy_rag_collection_name = policy_cfg["collection_name"]

        # 📚 Stories RAG configs
        stories_cfg = app_config["stories_rag"]
        self.stories_rag_llm = stories_cfg["llm"]
        self.stories_rag_llm_temperature = float(stories_cfg["llm_temperature"])
        self.stories_rag_embedding_model = stories_cfg["embedding_model"]
        self.stories_rag_vectordb_directory = str(here(stories_cfg["vectordb"]))
        self.stories_rag_unstructured_docs_directory = str(here(stories_cfg["unstructured_docs"]))
        self.stories_rag_k = int(stories_cfg["k"])
        self.stories_rag_chunk_size = int(stories_cfg["chunk_size"])
        self.stories_rag_chunk_overlap = int(stories_cfg["chunk_overlap"])
        self.stories_rag_collection_name = stories_cfg["collection_name"]

        # 🧳 Travel SQL Agent configs
        travel_cfg = app_config["travel_sqlagent_configs"]
        self.travel_sqldb_directory = str(here(travel_cfg["travel_sqldb_dir"]))
        self.travel_sqlagent_llm = travel_cfg["llm"]
        self.travel_sqlagent_llm_temperature = float(travel_cfg["llm_temperature"])

        # 🎵 Chinook SQL Agent configs
        chinook_cfg = app_config["chinook_sqlagent_configs"]
        self.chinook_sqldb_directory = str(here(chinook_cfg["chinook_sqldb_dir"]))
        self.chinook_sqlagent_llm = chinook_cfg["llm"]
        self.chinook_sqlagent_llm_temperature = float(chinook_cfg["llm_temperature"])

        # 🧠 Graph agent config
        self.thread_id = str(app_config["graph_configs"]["thread_id"])

        # 🔑 Groq API Key
        self.groq_api_key = os.getenv("GROQ_API_KEY")  # ✅ used in all ChatGroq agents
