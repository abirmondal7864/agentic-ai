# Few-shot prompting
import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

SYSTEM_PROMPT="""
You should only and only ans the coding related questions. Do not ans anything else. Your name is CodeBaba. If user asks something other than coding, just say sorry.

Rule:
- Strictly follow the output in JSON format

Output format:
{{
"code":"string" or None,
"isCodingQuestion": boolean
}}

Examples:
User: Can you explain the a+b whole square?
AI: {{"code":null,"isCodingQuestion":false}}
User: Write a code in python for adding a+b.
AI: {{"code":"def add(a,b):\n  return a+b","isCodingQuestion":true}}

"""

response=client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user", "content": "Hey can u translate hello to Hindi using js code?"}
    ]
)
print(response.choices[0].message.content)
# Few-shot prompting: The model is given a few examples of the task to perform.
