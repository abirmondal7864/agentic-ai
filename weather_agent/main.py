from openai.types.responses import response
from google.genai._gaos.utils import url
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

client= OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)
def get_weather(city:str):
    url=f"https://wttr.in/{city.lower()}?format=%C+%t"
    response=requests.get(url)

    if response.status_code==200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"


def main():
    user_query=input("> ")
    response=client.chat.completions.create(
        model="gemini-flash-lite-latest",
        messages=[
            {"role":"user","content": user_query }
        ]
    )
    print(f"🤖: {response.choices[0].message.content}")