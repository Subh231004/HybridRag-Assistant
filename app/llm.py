import os
import time
import ollama

from dotenv import load_dotenv
from openai import OpenAI
from typer import prompt

load_dotenv()

# ------------------------
# OpenRouter Client
# ------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ------------------------
# Cloud Models
# ------------------------

MODELS = [

    "deepseek/deepseek-chat-v3:free",

    "qwen/qwen3-32b:free",

    "meta-llama/llama-3.3-70b-instruct:free"
]


def stream_answer(query, context, task):

    if task == "quiz":

        instruction = """
        Generate 5 MCQs with answers from context.
        """

    elif task == "summary":

        instruction = """
    Generate concise revision notes.
    """

    elif task == "long_answer":

        instruction = """
    Generate detailed exam-style answer with:
    - headings
    - examples
    - bullet points
    - conclusion
    """

    elif task == "roadmap":

        instruction = """
    Generate step-by-step roadmap.
    """

    else:

        instruction = """
    Answer using retrieved context only.
    """

    prompt = f"""
You are an intelligent document assistant.

For long answers:
- use headings
- use bullet points
- include examples
- explain step-by-step

Rules:
- Use ONLY retrieved context
- Mention source names when relevant
- Mention page numbers when available
- If information comes from one document,
  do not mix content from other documents
- Avoid repetition
- Be concise

{context}

Question:
{query}
"""
    

    # ==================================================
    # TRY OPENROUTER FIRST
    # ==================================================

    for model in MODELS:

        try:

            print(f"Trying cloud model: {model}")

            stream = client.chat.completions.create(

                model=model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                stream=True
            )

            for chunk in stream:

                if (
                    chunk.choices
                    and chunk.choices[0].delta.content
                ):

                    yield chunk.choices[0].delta.content

            return

        except Exception as e:

            print(f"Cloud model failed: {model}")
            print(str(e))

            time.sleep(1)

    # ==================================================
    # FALLBACK TO OLLAMA
    # ==================================================

    try:

        print("Using local Ollama fallback...")

        client_ollama = ollama.Client(
            host="http://host.docker.internal:11434"
        )

        response = client_ollama.chat(

            model="phi3:mini",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            stream=True
        )

        for chunk in response:

            yield chunk["message"]["content"]

    except Exception as e:

        yield f"\n\nLocal Ollama failed:\n{str(e)}"