from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]


llm = ChatOpenAI(model="gpt-4o-mini")

def agent(state:AgentState)->AgentState:
    """Simple Node replies to user query"""
    response = llm.invoke(state['messages'])

    #you are storing the LLM response as an AIMessage
    state['messages'].append(AIMessage(content=response.content))

    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("agent",agent)
graph.add_edge(START,"agent")
graph.add_edge("agent",END)

agent = graph.compile()

conversation_history = []

HumanInput = input("Enter your message: ")
while HumanInput != "exit":
    conversation_history.append(HumanMessage(content=HumanInput))
    result = agent.invoke({"messages":conversation_history})
    # print(f"AI: {result['messages'][-1].content}")
    conversation_history = result['messages']
    HumanInput = input("Enter your message: ")


with open('logging.txt','w') as file:
    file.write("\n Conversation logging\n")
    for message in conversation_history:
        file.write(f"{message.type}: {message.content}\n")
    file.write("\nEnd of Conversation\n")

print("Conversation logged to logging.txt")