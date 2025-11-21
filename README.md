Agentic RAG

A simple Retrieval-Augmented Generation (RAG) pipeline built using LangChain, FAISS, and HuggingFace Transformers.
It loads a PDF, chunks it, embeds it, stores embeddings in FAISS, and answers questions using GPT-2 with retrieved context.

Features

PDF text extraction

Semantic text chunking

MiniLM embeddings

FAISS vector search

GPT-2 answer generation

Simple RetrievalQA chain with memory

How It Works

Load PDF

Split into chunks

Embed chunks (cached)

Store in FAISS

Retrieve top-k similar chunks

Generate answer using GPT-2

Project Structure
agentic_rag/
│── data/
│    └── agnetic_rag.pdf
│── notebooks/
│    └── agentic_rag.ipynb
│── README.md

Setup
git clone https://github.com/karthikeyan-2023/agentic_rag.git
cd agentic_rag
pip install -r requirements.txt

Usage

Open the notebook:

notebooks/agentic_rag.ipynb


Run cells to load the PDF, build the FAISS store, and query the model.

Example Query
query = "Phases of NLP"
response = qa_chain.invoke({"query": query})
print(response["result"])
