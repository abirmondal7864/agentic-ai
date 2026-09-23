# Chain-of-thought prompting

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import requests
import time
import json
import os


load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

def run_command(cmd:str):
    return os.system(cmd)

def get_weather(city:str):
    url=f"https://wttr.in/{city.lower()}?format=%C+%t"
    response=requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"

available_tools={"get_weather":get_weather,
                 "run_command":run_command
                }

SYSTEM_PROMPT="""
You are an expert AI assistant in resolving user queries using chain of thought.
You work on START,PLAN and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be in multiple steps.
Once u think enough PLAN has been done, finally u can give an OUTPUT.
You can also call a tool if required from the list of available tools
for every tool call wait for the observe step which is the output from the called tool.
Rule:
- Strictly follow the given JSON output format
- Output only ONE step at a time as a single valid JSON object
- The sequence of steps is START(where user gives input),PLAN(can be multiple times),OUTPUT(which the final answer display to user)

Output format:
{"step":"START" | "PLAN" | "OUTPUT" | "TOOL", "content":"string", "tool":"string", "input":"string"}

Available Tools:
- get_weather(city: str): Takes city name as an input string and returns the weather info about the city.
- run_command(cmd: str): Takes a command string as an input and executes it on the system and returns the output of the command.

Example 1:
START: Can you solve 2+3*5/10
PLAN: {"step":"PLAN", "content":"Seems like user is intersested in math problem"}
PLAN: {"step":"PLAN", "content": "Apply BODMAS"}
PLAN: {"step":"PLAN", "content": "15/10 "}
PLAN: {"step":"PLAN", "content": "3 * 1.5 = 4.5"}
PLAN: {"step":"PLAN", "content": "2 + 4.5 = 6.5"}
OUTPUT: {"step":"OUTPUT", "content":"The result of 2+3*5/10 is 6.5"}

Example 2:
START: What is the weather of delhi?
PLAN: {"step":"PLAN", "content":"Seems like user is intersested in getting weather of Delhi in India."}
PLAN: {"step":"PLAN", "content":"Lets see if we have any tool from the available tools to get the weather info"}
PLAN: {"step":"PLAN", "content": "Great, we have get_weather which can be used to get the weather info about any city."}
PLAN: {"step":"TOOL", "tool": "get_weather", "input": "delhi"}
OBSERVE: {"step":"OBSERVE", "tool": "get_weather", "input":"delhi", "output": "The weather in delhi is Cloudy +20°C"}
PLAN: {"step":"PLAN", "content": "Great, I got the weather info about delhi"}
OUTPUT: {"step":"OUTPUT", "content":"The current weather in delhi is 20 C with some cloudy sky."}
"""
print("\n\n\n")

class MyOutputFormat(BaseModel):
    step: str = Field(...,description="The ID of the step.Example: PLAN,OUTPUT,TOOL etc")
    content: Optional[str]=Field(None,description="The optionl string content for the step")
    tool: Optional[str]=Field(None,description="The ID of the tool to call")
    input: Optional[str]=Field(None,description="The input params for the tool")

message_history=[
    {"role":"system","content":SYSTEM_PROMPT},
]

while True:
    user_query=input("👉")
    message_history.append({"role":"user","content":user_query})

    while True:
        response=client.chat.completions.parse(
            model="gemini-flash-lite-latest",
            response_format=MyOutputFormat,
            messages=message_history
        )
        raw_result=response.choices[0].message.content
        message_history.append({"role": "assistant","content":raw_result})
        parsed_result=response.choices[0].message.parsed

        if parsed_result.step== "START":
            print("🔥", parsed_result.content)
        elif parsed_result.step == "PLAN":
            print("🧠", parsed_result.content)
        elif parsed_result.step == "OUTPUT":
            print("🤖", parsed_result.content)
            break
        elif parsed_result.step == "TOOL":
            tool_to_call= parsed_result.tool
            tool_input=parsed_result.input
            print(f"⚒️Calling Tool: {tool_to_call} with input: {tool_input}")

            tool_response= available_tools[tool_to_call](tool_input)
            print(f"⚒️: {tool_to_call} ({tool_input}) ={tool_response}")
            message_history.append({
                "role": "user",
                "content": json.dumps({
                    "step": "OBSERVE",
                    "tool": tool_to_call,
                    "input": tool_input,
                    "output": tool_response
                })
            })
            continue

        message_history.append({"role": "user", "content": "continue with next step"})
        time.sleep(1)

print("\n\n\n")
# Chain-of-thought prompting: The model is asked to think step by step before giving the final answer
