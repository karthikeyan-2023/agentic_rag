# Agentic RAG

A simple Retrieval-Augmented Generation (RAG) pipeline built using **LangChain**, **FAISS**, and **HuggingFace Transformers**.  
This project loads a PDF, chunks it, embeds the text, stores embeddings in FAISS, and answers questions using GPT-2 with retrieved context.

---

## 📌 Features
- PDF text extraction  
- Semantic text chunking  
- Embeddings using MiniLM  
- FAISS vector search  
- GPT-2 based answer generation  
- Simple RetrievalQA chain with memory  

---

## 🚀 How It Works
1. Load PDF  
2. Split PDF into chunks  
3. Generate embeddings (cached)  
4. Store embeddings in FAISS  
5. Retrieve similar chunks  
6. Generate final answer using GPT-2  

---

