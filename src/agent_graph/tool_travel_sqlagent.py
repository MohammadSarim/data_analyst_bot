from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_groq import ChatGroq
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class TravelSQLAgentTool:
    """
    A tool for querying travel-related data from a SQL database using a Groq LLaMA model to
    convert natural language into SQL and return a natural language answer.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temerature: float) -> None:
        self.sql_agent_llm = ChatGroq(
            model=llm,
            temperature=llm_temerature,
            api_key=TOOLS_CFG.groq_api_key  # Make sure your tools_config.yml has this
        )

        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        print("✅ Connected to SQL Database")
        print("🧩 Tables:", self.db.get_usable_table_names())

        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(self.sql_agent_llm, self.db)

        answer_prompt = PromptTemplate.from_template(self.system_role)
        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()

        self.chain = (
            RunnablePassthrough.assign(query=write_query)
            .assign(result=itemgetter("query") | execute_query)
            | answer
        )


@tool
def query_travel_sqldb(query: str) -> str:
    """
    Query the Swiss Airline SQL Database and retrieve relevant information.
    Input should be a natural language query.
    """
    agent = TravelSQLAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,                        
        sqldb_directory=TOOLS_CFG.travel_sqldb_directory,         
        llm_temerature=TOOLS_CFG.travel_sqlagent_llm_temperature 
    )
    return agent.chain.invoke({"question": query})
