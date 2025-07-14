import os
import re
import ast
from typing import List
from pydantic import BaseModel, Field
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from src.agent_graph.load_tools_config import LoadToolsConfig

# 🔧 Load environment-specific config from YAML
TOOLS_CFG = LoadToolsConfig()


# ✅ Pydantic Model for Structured Table Response
class Table(BaseModel):
    name: str = Field(description="Name of table in SQL database.")


# ✅ Helper: Get Relevant Tables using Groq LLM
def get_relevant_tables_prompt(table_names: List[str], question: str, llm) -> List[str]:
    table_list_str = "\n".join(table_names)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant that selects only the most relevant table names for answering SQL questions.
Here are the available tables:
{table_list_str}

Return a valid Python list of table names that are relevant to the user query.
Only return table names from the above list.
Example: ["Customer", "Invoice"]
"""),
        ("user", question)
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        raw_response = chain.invoke({})
        parsed = ast.literal_eval(raw_response)
        return [t for t in parsed if t in table_names]
    except Exception as e:
        print(f"⚠️ Could not parse table list. Error: {e}")
        return table_names  # fallback to using all


# ✅ Main Agent Class
class ChinookSQLAgent:
    def __init__(self, sqldb_path: str, llm_model: str, llm_temp: float):
        self.llm = ChatGroq(
            model=llm_model,
            temperature=llm_temp,
            api_key=TOOLS_CFG.groq_api_key
        )

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_path}")
        self.available_tables = self.db.get_usable_table_names()

        self.query_chain = create_sql_query_chain(self.llm, self.db)

        # Compose runnable chain
        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=self._get_relevant_tables_chain()
        ) | self.query_chain

        self.answer_prompt = PromptTemplate.from_template(
            "Given the question and the result of the SQL query, generate a helpful final answer.\n\n"
            "Question: {question}\nSQL Result: {result}\nAnswer:"
        )
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def _get_relevant_tables_chain(self):
        def extract_tables(inputs: dict):
            question = inputs["question"]
            return get_relevant_tables_prompt(self.available_tables, question, self.llm)
        return extract_tables

    def run(self, question: str) -> str:
        try:
            # Step 1: Generate SQL
            llm_output = self.full_chain.invoke({"question": question})
            print("\n🧠 LLM SQL Generation Output:\n", llm_output)

            # Step 2: Extract SQL from Markdown blocks or fallback
            sql_match = re.search(r"```sql\s+(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_query = sql_match.group(1).strip()
            else:
                fallback_match = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)[\s\S]+?;", llm_output, re.IGNORECASE)
                if fallback_match:
                    sql_query = fallback_match.group(0).strip()
                else:
                    return f"❌ Could not extract SQL from LLM output:\n\n{llm_output}"

            print("\n✅ Final SQL to Run:\n", sql_query)

            # Step 3: Execute SQL
            try:
                sql_result = self.db.run(sql_query)
            except Exception as sql_err:
                return f"🔥 SQL Execution Error:\n{str(sql_err)}"

            # Step 4: Final Answer
            return self.answer_chain.invoke({
                "question": question,
                "result": str(sql_result)
            })

        except Exception as err:
            return f"❌ Agent runtime error: {str(err)}"


# ✅ Tool Wrapper
@tool
def query_travel_sqldb(query: str) -> str:
    """
    Query the Chinook SQL database using a natural language question.
    """
    agent = ChinookSQLAgent(
        sqldb_path=TOOLS_CFG.chinook_sqldb_directory,
        llm_model=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temp=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )
    return agent.run(query)


# ✅ MAIN BLOCK
if __name__ == "__main__":
    try:
        test_question = "In chinook DB, list the total sales per country. Which country's customers spent the most?"
        print(f"\n🔍 Test Question:\n{test_question}")
        answer = query_travel_sqldb.invoke(test_question)
        print("\n✅ Final Answer:\n", answer)
    except Exception as main_err:
        print(f"\n❌ Error in main execution: {main_err}")
