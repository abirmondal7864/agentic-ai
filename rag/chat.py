import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types

load_dotenv()

# Gemini client
genai_client=genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Vector Embeddings
embedding_model=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_db=QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

# Take user input
user_query=input("Ask something: ")
# Relevant chunks from the vector db
search_results=vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([
    f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
    for result in search_results
])

SYSTEM_PROMPT=f"""
    You are a helpful AI Assistant who answers user query based on the available context retrived from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user to open the right page number to know more.

    Context:
    {context}
"""
response = genai_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=f"""
{SYSTEM_PROMPT}

User question:
{user_query}
"""
)

print(f"🤖: {response.text}")