# Chain-of-thought prompting
import os 
from dotenv import load_dotenv
from openai import OpenAI
import json 
import time 

load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

SYSTEM_PROMPT="""
You are an expert AI assistant in resolving user queries using chain of thought.
You work on START,PLAN and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be in multiple steps.
Once u think enough PLAN has been done, finally u can give an OUTPUT.

Rule:
- Strictly follow the given JSON output format
- The sequence of steps is START(where user gives input),PLAN(can be multiple times),OUTPUT(which the final answer display to user)

Output format:
{"step":"START" |"PLAN" | "OUTPUT", "content":"string"}


Examples:
START: Can you solve 2+3*5/10
PLAN: {"step":"PLAN", "content":"Seems like user is intersested in math problem"}
PLAN: {"step":"PLAN","content": "Apply BODMAS"},
PLAN: {"step":"PLAN","content": "15/10 "},
PLAN: {"step":"PLAN","content": "3 * 1.5 = 4.5"},
PLAN: {"step":"PLAN","content": "2 + 4.5 = 6.5"},
OUTPUT: {"step":"OUTPUT", "content":"The result of 2+3*5/10 is 6.5"},


"""
print("\n\n\n")
message_history=[
    {"role":"system","content":SYSTEM_PROMPT},
]
user_query=input("👉")
message_history.append({"role":"user","content":user_query})

while True:
    response=client.chat.completions.create(
        model="gemini-3.5-flash",
        response_format={"type":"json_object"},
        messages=message_history
    )

    raw_result=response.choices[0].message.content
    message_history.append({"role": "assistant","content":raw_result})
    parsed_result = json.loads(raw_result)

    # Normalize to a list so both a single step dict and a list of steps work
    steps = parsed_result if isinstance(parsed_result, list) else [parsed_result]

    has_output = False
    for item in steps:
        step = item.get("step")
        content = item.get("content")

        if step == "START":
            print("🔥", content)
        elif step == "PLAN":
            print("🧠", content)
        elif step == "OUTPUT":
            print("🤖", content)
            has_output = True

    if has_output:
        break

    message_history.append({"role": "user", "content": "continue with next step"})
    time.sleep(1)

print("\n\n\n")
# Chain-of-thought prompting: The model is asked to think step by step before giving the final answer
