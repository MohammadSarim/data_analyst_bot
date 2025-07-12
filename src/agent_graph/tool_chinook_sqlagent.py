import os
import re
import ast
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from agent_graph.load_tools_config import LoadToolsConfig

# Load config
TOOLS_CFG = LoadToolsConfig()

# Optional helper schema if needed for table tools
class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")


def get_relevant_tables_prompt(table_names: List[str], question: str, llm) -> List[str]:
    """
    Uses an LLM to return relevant tables for the user's query.
    """
    table_list_str = "\n".join(table_names)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant that selects only the most relevant table names for answering SQL questions.
Here are the available tables:
{table_list_str}

Return a valid Python list of table names that are relevant to the user query.
Only return table names from the above list.
Example: ["Customer", "Invoice"]"""),
        ("user", question)
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({})
    try:
        parsed = ast.literal_eval(raw_response)
        return [t for t in parsed if t in table_names]
    except Exception:
        return table_names  # fallback if parsing fails


class ChinookSQLAgent:
    def __init__(self, sqldb_path: str, llm_model: str, llm_temp: float):
        self.llm = ChatGroq(model=llm_model, temperature=llm_temp, api_key=os.getenv("GROQ_API_KEY"))
        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_path}")
        self.available_tables = self.db.get_usable_table_names()

        # Schema-aware SQL query generator
        self.query_chain = create_sql_query_chain(self.llm, self.db)

        # Full chain: adds relevant tables before generating SQL
        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=self._get_relevant_tables_chain()
        ) | self.query_chain

    def _get_relevant_tables_chain(self):
        def extract_tables(inputs: dict):
            question = inputs["question"]
            return get_relevant_tables_prompt(self.available_tables, question, self.llm)
        return extract_tables


@tool
def query_chinook_sqldb(query: str) -> str:
    """
    Query the Chinook SQL Database using a natural language question.
    """
    agent = ChinookSQLAgent(
        sqldb_path=TOOLS_CFG.chinook_sqldb_directory,
        llm_model=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temp=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )

    llm_output = agent.full_chain.invoke({"question": query})

    # Extract SQL from LLM output
    match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?;", llm_output, re.IGNORECASE)
    if match:
        sql_query = match.group(0).strip()
    elif "SQLQuery:" in llm_output:
        sql_query = llm_output.split("SQLQuery:")[1].strip()
    else:
        return f"❌ Error: Could not extract a valid SQL query.\n\n🔍 LLM Output:\n{llm_output}"

    print("🧠 Final SQL to Execute:", sql_query)

    try:
        return agent.db.run(sql_query)
    except Exception as e:
        return f"🔥 Execution Error:\n{str(e)}"
