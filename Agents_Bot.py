from typing import TypedDict,List
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message:str

llm = ChatOpenAI(model="gpt-4o-mini")

def agent(state:AgentState)->AgentState:
    """Simple Node that adds a compliment to the message"""
    state['message'] = llm.invoke(state['message'])
    print(state['message'].content)
    return state

graph = StateGraph(AgentState)
graph.add_node("agent",agent)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

agent = graph.compile()
HumanMessage = input("Enter your message: ")

while HumanMessage != "exit":
    
    agent.invoke({"message":HumanMessage})

    HumanMessage = input("\n Enter your next message: \n")