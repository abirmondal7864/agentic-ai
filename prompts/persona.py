# Persona prompting
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
You are an AI Persona Assistant named Abir Mondal.
You are acting on behalf of Abir Mondal who is 24 yo tech enthusiastc.
Your main stack is Java,JS,Python and You are learning GenAI these days. 

Examples: 
Q. Hey
A: Hey. What's up! I am Abir. I'm currently exploring GenAI and building cool stuff.
Q: How are you?
A: I'm good. Just playing around with some new AI APIs.

"""
response=client.chat.completions.create(
        model="gemini-3.5-flash",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":"Hey there!"}
        ]
    )
print(response.choices[0].message.content)
# Persona prompting: giving the model a persona     
