import os
import streamlit as st
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

os.environ["MISTRAL_API_KEY"] = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"
api_key = os.getenv("MISTRAL_API_KEY")

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
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy"
]

def fetch_policy(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def chunk_text(text, chunk_size=512):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embedding_batched(chunks, batch_size=10):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=batch)
        embeddings.extend([entry.embedding for entry in embeddings_batch_response.data])
        time.sleep(1)
    return np.array(embeddings)


def build_index(embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def create_policy_index():
    policy_texts = [fetch_policy(url) for url in policyLinks]
    policy_chunks = [chunk_text(text) for text in policy_texts]
    flat_chunks = [chunk for sublist in policy_chunks for chunk in sublist]
    
    embeddings = get_text_embedding(flat_chunks)
    index = build_index(embeddings)
    return index, flat_chunks

policy_index, policy_chunks = create_policy_index()

st.title("Agentic RAG: UDST Policy Chatbot")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        query_embedding = np.array([get_text_embedding([query])[0]])
        D, I = policy_index.search(query_embedding, k=3)
        retrieved_texts = "\n".join([policy_chunks[idx] for idx in I[0]])
        
        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_texts}
        ---------------------
        Given the context information, answer the query.
        Query: {query}
        Answer:
        """
        
        client = Mistral(api_key=api_key)
        messages = [UserMessage(content=prompt)]
        chat_response = client.chat.complete(model="mistral-small-latest", messages=messages)
        response = chat_response.choices[0].message.content
        
        st.text_area("Answer:", response, height=200)
    else:
        st.warning("Please enter a query.")
