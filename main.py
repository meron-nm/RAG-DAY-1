import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


documents = [
    {
        "title": "ABOUT MERON!",
        "content": "I am a third-year Computer Science student at Addis Ababa University with a strong interest in technology and AI. I am hardworking, motivated, and enjoy learning new skills and tackling challenges. I adapt quickly to new situations and am excited to contribute to projects where I can grow professionally and deepen my knowledge in tech and AI."
    },
    
]


def retrieve_doc(query):
    for doc in documents:
        if query.lower() in doc["title"].lower():
            return doc["content"]
    return "No document found."


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0, #it gives less accurate answer as it approaches to 1
    groq_api_key=groq_api_key
)

user_question = "Tell me about myself."

# WITHOUT retrieval
print("Answer WITHOUT retrieval:\n")
print(llm.invoke(user_question).content)

# WITH retrieval
context = retrieve_doc("ABOUT MERON!")

rag_prompt = f"""   
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know". 

Context:
{context}

Question:
{user_question}
"""

print("\nAnswer WITH retrieval (RAG):\n")
print(llm.invoke(rag_prompt).content)

