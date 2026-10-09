from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model

load_dotenv()

llm=init_chat_model(
    model="gemini-3.6-flash",
    model_provider="google_genai"
)

class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state:State):
    response=llm.invoke(state["messages"])
    return { "messages":[response]}

def samplenode(state: State):
    print(f"\n\nInside Sample Node : {state}")
    return {"messages": [("assistant","Sample message appened")]}

graph_builder=StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("samplenode",samplenode)  

graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","samplenode")
graph_builder.add_edge("samplenode",END)

graph = graph_builder.compile()

updated_state= graph.invoke({"messages": [
    ("user","Hello my name is Abir Mondal")]})

print("\n\nupdated_state: ",updated_state)

# (START) -> chatbot -> samplenode -> (END)

# state = {messages: ["Hey there"]}
# node runs: chatbot(state:["Hey there"]) -> ["Hi, this is a message from ChatBot"]
# state ={"messages":["Hey there", "Hi this is a message from ChatBot Node"]}
