import os
os.environ["MISTRAL_API_KEY"] = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"
print(f"MISTRAL_API_KEY: {os.environ.get('MISTRAL_API_KEY')}")

api_key = os.getenv("MISTRAL_API_KEY")

import requests
from bs4 import BeautifulSoup
import re

policyLinks = [
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance"
]

all_texts = {}

for url in policyLinks:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    all_texts[url] = text
    print(f"Extracted text from: {url}")



file_name = "Text.txt"
with open(file_name, 'w') as file:
    file.write(text)

chunk_size = 512
chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

len(chunks)

from mistralai import Mistral
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

import numpy as np
text_embeddings = get_text_embedding(chunks)

len(text_embeddings)

len(text_embeddings[0].embedding)

embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])

import faiss

d = len(text_embeddings[0].embedding)
index = faiss.IndexFlatL2(d)
index.add(embeddings)

question = "How to get a scholarship"
question_embeddings = np.array([get_text_embedding([question])[0].embedding])

question_embeddings

D, I = index.search(question_embeddings, k=10)
print(I)

retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
print(retrieved_chunk)

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

from mistralai import Mistral, UserMessage
def mistral(user_message, model="mistral-small-latest", is_json=False):
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    messages = [
        UserMessage(content=user_message),
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        )
    return chat_response.choices[0].message.content

response = mistral(prompt)
print(response)