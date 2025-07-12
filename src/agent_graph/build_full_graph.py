from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_groq import ChatGroq  # ✅ Use ChatGroq instead of ChatOpenAI
from agent_graph.tool_chinook_sqlagent import query_chinook_sqldb
from agent_graph.tool_travel_sqlagent import query_travel_sqldb
from agent_graph.tool_lookup_policy_rag import lookup_swiss_airline_policy
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.tool_stories_rag import lookup_stories
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema

TOOLS_CFG = LoadToolsConfig()


def build_graph():
    """
    Builds the agent graph with Groq-based LLM and tools like SQL, RAG, and web search.
    """

    # ✅ Initialize primary Groq LLM with API key
    primary_llm = ChatGroq(
        model=TOOLS_CFG.primary_agent_llm,
        temperature=TOOLS_CFG.primary_agent_llm_temperature,
        api_key=TOOLS_CFG.groq_api_key
    )

    graph_builder = StateGraph(State)

    # ✅ Load all tool functions
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [
        search_tool,
        lookup_swiss_airline_policy,
        lookup_stories,
        query_travel_sqldb,
        query_chinook_sqldb,
    ]

    # ✅ Bind LLM with tools
    primary_llm_with_tools = primary_llm.bind_tools(tools)

    # Define chatbot node
    def chatbot(state: State):
        return {"messages": [primary_llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # Define tool node
    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # Conditional routing based on whether tools are needed
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # Link tool node back to chatbot for follow-up
    graph_builder.add_edge("tools", "chatbot")

    # Initial node
    graph_builder.add_edge(START, "chatbot")

    # ✅ Use in-memory checkpointer for simplicity
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Optional: visualize the agent structure
    plot_agent_schema(graph)

    return graph
