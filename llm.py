import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_answer(question, context):

    instructions = """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not make up information.
2. If the answer cannot be found in the context, say:
   "I don't have enough information in the provided document."
3. Keep the answer clear and concise.
"""

    prompt = f"""
Context:
{context}

User Question:
{question}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=instructions,
        input=prompt
    )

    return response.output_text