****📘 Agentic RAG — End-to-End Retrieval Augmented Generation Pipeline********

Welcome to Agentic RAG, a lightweight but fully functional Retrieval-Augmented Generation system built using LangChain, FAISS, and HuggingFace Transformers.
This project demonstrates how to take any PDF, chunk it, embed it, store it in a vector database, and then query it using an LLM that retrieves real context before answering.

This repo is ideal for anyone learning RAG, prototyping retrieval systems, or testing local LLM workflows.

****🌟 Key Features****
**🔍 PDF → Text Loader**

Extracts clean text from PDFs using PyPDFLoader.
(Example PDF included: agnetic_rag.pdf)

**🧩 Semantic Chunking**

Breaks long documents into manageable chunks using
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).

**🧠 Embeddings + Cache
**
Embeddings are created using:

sentence-transformers/all-MiniLM-L6-v2


Includes an in-memory cache so repeated chunk embeddings don’t get recomputed.

**📡 FAISS Vector Store**

Stores all document embeddings for efficient similarity search.
Search retrieves top-k relevant chunks (k=2 by default).

**🗣️ RetrievalQA with GPT-2**

Uses HuggingFace’s GPT-2 via a text-generation pipeline for answer creation.
Integrates a simple prompt template and conversation memory.

**📝 Conversational Memory**

Uses LangChain’s

ConversationBufferMemory


to maintain chat continuity.
**
📂 Project Structure**
agentic_rag/
│
├── data/
│   └── agnetic_rag.pdf          # Sample PDF used in the notebook
│
├── notebook/
│   └── agentic_rag.ipynb        # Main Colab / Jupyter Notebook
│
├── README.md                    # You're reading this
│
└── requirements.txt             # Optional (I can generate this if you want)

**🚀 How It Works (Pipeline Overview)******
1. Install dependencies
%pip install faiss-cpu langchain sentence-transformers transformers pypdf

2. Load PDF
loader = PyPDFLoader("/content/agnetic_rag.pdf")
documents = loader.load()

3. Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

4. Create embeddings + cache
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

5. Store embeddings in FAISS
vector_store = FAISS.from_texts(
    [chunk.page_content for chunk in chunks],
    embedding_model
)

6. Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

7. Prompt template
prompt_template = """
Use the following context to answer the question. If you don't know the answer, say so.

Context:
{context}

Question:
{question}
"""

8. Load GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

9. Set conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="result"
)

10. Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    memory=memory
)

💬 Example Query
result = qa_chain.invoke({"query": "Phases of NLP"})
print(result["result"])


Retrieves context + generates answer based on your uploaded PDF
(agnetic_rag.pdf) — which includes topics like lexical analysis, tokenization, POS tagging, syntax, and semantic challenges.

📦 Clone This Repository
git clone https://github.com/karthikeyan-2023/agentic_rag.git
cd agentic_rag
