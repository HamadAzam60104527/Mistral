import os
import streamlit as st
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

# Set API Key (Ensure it's set in your environment variables securely)
os.environ["MISTRAL_API_KEY"] = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"
api_key = os.getenv("MISTRAL_API_KEY")

# UDST Policy Links
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
]

# Fetch Policy Data Lazily
def fetch_policy(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

# Split into chunks
def chunk_text(text, chunk_size=512):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# Generate Embeddings (Cached for Rate Limit Control)
@st.cache_data
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return np.array([entry.embedding for entry in embeddings_batch_response.data])

# Create FAISS Index
def build_index(embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Streamlit UI
st.title("UDST Policy Chatbot")
st.sidebar.header("Select Policy")
selected_policy = st.sidebar.selectbox("Choose a Policy", policyLinks)
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        policy_text = fetch_policy(selected_policy)
        chunks = chunk_text(policy_text)
        embeddings = get_text_embedding(chunks)
        index = build_index(embeddings)
        
        # Retrieve relevant chunks
        query_embedding = np.array([get_text_embedding([query])[0]])
        D, I = index.search(query_embedding, k=2)  # Reduced to 2 chunks
        retrieved_texts = "\n".join([chunks[idx] for idx in I[0]])
        
        # Generate Answer
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
        chat_response = client.chat.complete(model="mistral-small-latest", messages=messages)  # Switched to smaller model
        response = chat_response.choices[0].message.content
        
        st.text_area("Answer:", response, height=200)
    else:
        st.warning("Please enter a query.")
