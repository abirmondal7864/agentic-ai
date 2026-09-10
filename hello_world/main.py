from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client= OpenAI()

response=client.chat.completions.create(
    model="",
    messages=[
        {"role":"user","content": "Hey There! I am Abir. Nice to meet you."}
    ]
)
print(response.choices[0].message.content)