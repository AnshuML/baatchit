import google.generativeai as genai
from langchain_community.vectorstores import Chroma
import numpy as np
import os

from config import *

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini LLM
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Google Embedding Model
class EmbeddingFunction:
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        """Internal function to call Google's embedding model"""
        result = genai.embed_content(model="models/embedding-001", content=text)
        return np.array(result["embedding"]).tolist()  # Ensure list format

def format_gemini_response(text):
    """Ensure consistent formatting of Gemini responses"""
    lines = text.split('\n')
    formatted = []

    for line in lines:
        if line.strip() == '':
            continue
        if line.startswith('**') or line.startswith('- **'):
            formatted.append(line)
        else:
            formatted.append(f"  - {line}")

    return '\n'.join(formatted)

def generate_answer(context, question):
    prompt = f"""
You are an HR Policy Assistant for Ajit Industries Pvt. Ltd.
Always respond in the same language as the question.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Use ONLY information from the CONTEXT
2. Format answer in clear bullet points
3. Include:
   - Policy Title
   - Effective Date
   - Key Details (e.g., eligibility, processes, allowances)
4. If unsure, say "I'm not sure about this policy. Please consult HR."

Example format:
- **Policy Title**: Code of Conduct
- **Effective Date**: April 1, 2023
- **Key Details**:
  - Prohibition of harassment
  - Confidentiality requirements
  - Conflict of interest rules

ANSWER:
"""
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.3)
    )
    return format_gemini_response(response.text.strip())

# MUST be at the END of the file
embed_fn = EmbeddingFunction()

