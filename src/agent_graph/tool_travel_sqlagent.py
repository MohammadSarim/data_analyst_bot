from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent_graph.load_tools_config import LoadToolsConfig
import pandas as pd
import re

TOOLS_CFG = LoadToolsConfig()

class TravelDataFrameAgentTool:
    def __init__(self, llm: str, dataframe: pd.DataFrame, llm_temperature: float) -> None:
        self.df_agent_llm = ChatGroq(
            model=llm,
            temperature=llm_temperature,
            api_key=TOOLS_CFG.groq_api_key
        )

        self.df = dataframe
        print("✅ Connected to DataFrame")
        print("🧩 Columns:", self.df.columns.tolist())

        self.analyze_data = create_pandas_dataframe_agent(
            llm=self.df_agent_llm,
            df=self.df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            max_iterations=2
        )

        self.system_role = """Given the following user question and analysis result, answer the user question.

Question: {question}
Analysis Result: {result}
Answer:"""

        self.answer_prompt = PromptTemplate.from_template(self.system_role)
        self.answer_chain = self.answer_prompt | self.df_agent_llm | StrOutputParser()

    def _extract_code(self, llm_output: str) -> str:
        """Extracts Python code from LLM output if present."""
        code_block = re.search(r"```(?:python)?\n?(.*?)\n?```", llm_output, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        action_match = re.search(r"Action Input:\s*\n?(.*?)$", llm_output, re.DOTALL | re.MULTILINE)
        if action_match:
            return action_match.group(1).strip()

        return ""

    def run(self, question: str) -> str:
        try:
            agent_response = self.analyze_data.invoke({
                "input": f"""You are a data analyst working with airline data. 
Use Python code to answer the question using the dataframe.
Question: {question}"""
            })

            print("🧠 AGENT RESPONSE:", agent_response)

            if isinstance(agent_response, dict) and "output" in agent_response:
                llm_raw_output = agent_response["output"]
            else:
                llm_raw_output = str(agent_response)

            analysis_code = self._extract_code(llm_raw_output)
            print("🧾 EXTRACTED CODE:\n", analysis_code)

            local_vars = {"df": self.df}
            if analysis_code:
                try:
                    exec(analysis_code, {}, local_vars)
                    result = local_vars.get("result", "Code executed but no `result` variable found.")
                except Exception as e:
                    result = f"Code execution error: {e}\nRaw output:\n{llm_raw_output}"
            else:
                result = llm_raw_output

            print("📊 FINAL RESULT:", result)

            response = self.answer_chain.invoke({
                "question": question,
                "result": str(result)[:1000]
            })

            return response

        except Exception as e:
            return f"Agent processing error: {str(e)}"

@tool
def query_travel_sqldb(query: str) -> str:
    """
    Query the airline DataFrame using a natural language question.
    Returns a helpful final answer based on data analysis.
    """
    df = pd.read_csv(TOOLS_CFG.travel_sqldb_directory)

    agent = TravelDataFrameAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,
        dataframe=df,
        llm_temperature=TOOLS_CFG.travel_sqlagent_llm_temperature
    )

    return agent.run(query)
