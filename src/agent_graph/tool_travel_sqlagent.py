from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent_graph.load_tools_config import LoadToolsConfig
from langchain_groq import ChatGroq
import re

TOOLS_CFG = LoadToolsConfig()

class TravelSQLAgentTool:
    def __init__(self, llm: str, sqldb_directory: str, llm_temerature: float) -> None:
        self.sql_agent_llm = ChatGroq(
            model=llm,
            temperature=llm_temerature,
            api_key=TOOLS_CFG.groq_api_key
        )

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        print("✅ Connected to SQL Database")
        print("🧩 Tables:", self.db.get_usable_table_names())

        self.write_query = create_sql_query_chain(self.sql_agent_llm, self.db)

        self.system_role = """Given the following user question, SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""

        self.answer_prompt = PromptTemplate.from_template(self.system_role)
        self.answer_chain = self.answer_prompt | self.sql_agent_llm | StrOutputParser()

    def _extract_sql(self, llm_output: str) -> str:
        # Extract the SQL query from the LLM output using regex
        match = re.search(r"(?i)sqlquery:\s*(SELECT .*?;)", llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        # fallback: return first SELECT statement found
        fallback = re.findall(r"(SELECT\s+.+?;)", llm_output, re.DOTALL | re.IGNORECASE)
        if fallback:
            return fallback[0].strip()
        raise ValueError("❌ Failed to extract SQL query from LLM output:\n" + llm_output)

    def run(self, question: str) -> str:
        llm_raw_output = self.write_query.invoke({"question": question})
        print("🧠 RAW LLM OUTPUT:\n", llm_raw_output)

        sql_query = self._extract_sql(llm_raw_output)
        print("✅ CLEANED SQL QUERY:", sql_query)

        result = self.db.run(sql_query)
        print("📊 SQL RESULT:", result)

        response = self.answer_chain.invoke({
            "question": question,
            "query": sql_query,
            "result": str(result)
        })

        return response


@tool
def query_travel_sqldb(query: str) -> str:
    """
    Query the airline SQL database using a natural language question.
    Returns a helpful final answer.
    """
    agent = TravelSQLAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,
        sqldb_directory=TOOLS_CFG.travel_sqldb_directory,
        llm_temerature=TOOLS_CFG.travel_sqlagent_llm_temperature
    )

    return agent.run(query)

