from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver


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

graph_builder=StateGraph(State)


graph_builder.add_node("chatbot",chatbot)


graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

def compile_graph_with_checkpointer(checkpoint):
        return graph_builder.compile(
            checkpointer=checkpointer
        )

DB_URI="mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer=compile_graph_with_checkpointer(checkpointer)
    config ={
        "configurable":{
            "thread_id":"abir" # user_id
        }
    }
    for chunk in graph_with_checkpointer.stream({"messages": [
        ("user","what is my name?")]}
        ,config,stream_mode="values"
        ):
        chunk["messages"][-1].pretty_print()




# (START) -> chatbot -> (END)

# state = {messages: ["Hey there"]}
# node runs: chatbot(state:["Hey there"]) -> ["Hi, this is a message from ChatBot"]

# Checkpointer(piyush) = Hey, my name is Abir Mondal
