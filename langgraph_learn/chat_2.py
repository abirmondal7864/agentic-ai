import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Optional

from google import genai
from groq import Groq
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Gemini client: primary model
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Groq client: evaluator model
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatbot(state: State):
    print("\n\nChatBot Node", state)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=state["user_query"]
    )

    return {"llm_output": response.text}


def evaluate_response(state: State):
    print("\n\nevaluate_response Node", state)

    if state.get("is_good"):
        return "endnode"

    return "chatbot_grok"


def chatbot_grok(state: State):
    print("\n\nChatbot Grok Node", state)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate the answer for correctness, relevance, "
                    "and completeness. Return only PASS or FAIL."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['user_query']}\n"
                    f"Gemini answer: {state['llm_output']}"
                )
            }
        ]
    )

    verdict = response.choices[0].message.content.strip().upper()

    return {"is_good": verdict.startswith("PASS")}


def endnode(state: State):
    print("\n\nEnd Node", state)
    print("\nGemini Answer:", state["llm_output"])
    print("Evaluation Passed:", state["is_good"])
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_grok", chatbot_grok)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "chatbot_grok")
graph_builder.add_edge("chatbot_grok", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

updated_state = graph.invoke({
    "user_query": "Hey, what is 2+2?",
    "llm_output": None,
    "is_good": None
})

print("\nUpdated State:", updated_state)
