from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import ToolMessage, SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from operations import *  # This imports your tool functions
import os

load_dotenv()

# Initialize Groq LLaMA model
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b"
)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Register tools
tools = [add, subtract, multiply]
model = llm.bind_tools(tools)

# LLM Call Node
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
You are a helpful AI assistant that prints all results. You can use tools to answer questions.
Wait for the result of each tool call before making the next one. If you don't have tools available, answer the question directly.
""")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Decide whether to continue
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"

# Graph creation
graph = StateGraph(AgentState)
graph.add_node("our_Agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_Agent")
graph.add_conditional_edges("our_Agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_Agent")

# Compile app
app = graph.compile()

# === Chatbot Loop ===
conversation_history: list[BaseMessage] = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": conversation_history})
    
    # Get new AI message(s) from updated result
    new_messages = result["messages"]
    
    # Print and store new AI responses only
    for msg in new_messages[len(conversation_history):]:
        if isinstance(msg, AIMessage):
            print(f"\nAI: {msg.content}")
            conversation_history.append(msg)

    user_input = input("\nYou: ")

# Save chat
with open("history.txt", "w", encoding="utf-8") as file:
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"You: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n")
print("Conversation saved to history.txt")
