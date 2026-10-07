import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client=OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response=client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[
        {
            "role": "user",
            "content":[
                {
                    "type":"text",
                    "text":"What does this image depicts?"
                },
                {
                    "type":"image_url",
                    "image_url":{
                        "url":"https://images.pexels.com/photos/12130758/pexels-photo-12130758.jpeg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
    