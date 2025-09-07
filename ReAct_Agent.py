#ReAct Agent - Reasoning and Action Agent

from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage#Passes data back to LLM after tool call
from langchain_core.messages import SystemMessage#System message to provide context to the LLM
from langchain_core.messages import BaseMessage#
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int)->int:
    """Add two numbers together"""
    return a+b

@tool
def subtract(a:int,b:int)->int:
    """Subtract two numbers"""
    return a-b

tools = [add,subtract]                  

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state:AgentState)->AgentState:

    # Build a proper list of messages: system prompt + prior messages
    msgs = [SystemMessage(content="You are a helpful assistant that can add and subtract numbers")] + list(state["messages"])
    response = model.invoke(msgs)

    # state['messages'].append(response)

    return {"messages": [response]}

def should_continue(state:AgentState)->str:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)
graph.add_node("tool_node",ToolNode(tools))
graph.add_edge(START,"our_agent")
graph.add_conditional_edges("our_agent",should_continue,
{
    "continue":"tool_node",
    "end":END
})
graph.add_edge("tool_node","our_agent")
graph.add_edge("our_agent",END)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

result = app.stream({"messages":[("user","Substract 10 and 5. Add 5+6 . Then add the results of the two operations")]},stream_mode="values")
print_stream(result)