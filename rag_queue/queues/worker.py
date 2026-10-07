import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# Load environment variables
load_dotenv()


# Gemini through OpenAI-compatible API
openai_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Test Gemini
response = openai_client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response.choices[0].message.content)


# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# Qdrant Vector Database
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)


def process_query(user_query: str):

    print("Searching Chunks:", user_query)

    search_results = vector_db.similarity_search(
        query=user_query
    )

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\n"
        f"Page Number: {result.metadata['page_label']}\n"
        f"File Location: {result.metadata['source']}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user queries
based on the available context retrieved from a PDF file.

You should only answer the user based on the following context
and navigate the user to the relevant page number to know more.

Context:
{context}
"""

    response = openai_client.chat.completions.create(
        model="gemini-3.6-flash",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    print(f"🤖: {response.choices[0].message.content}")
    return response.choices.message.content