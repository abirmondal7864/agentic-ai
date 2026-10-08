from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END

class State(TypedDict):
    messages: Annotated[list.add_messages]

def chatbot(state:State):
    return { "messages":["Hi, This is a message from ChatBot Node"]}

def samplenode(state: State):
    return {"messages": ["Sample message appened"]}

graph_builder=StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("samplenode",samplenode)