import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG CHATBOT APP"

PDF_PATH = "islr.pdf"


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

print(f"Loaded {len(docs)} pages")


# --------------------------------------------------
# 2. Split documents
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = splitter.split_documents(docs)

print(f"Created {len(splits)} chunks")


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

# 3) Embed + index

emb = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://127.0.0.1:11434"
)

# Test embedding
test_embedding = emb.embed_query("Hello world")
print(f"Embedding dimension: {len(test_embedding)}")

# Create persistent Chroma database
vs = Chroma(
    collection_name="rag_collection",
    embedding_function=emb,
    persist_directory="./chroma_db"
)

# Add documents in batches
batch_size = 25

for i in range(0, len(splits), batch_size):

    batch = splits[i:i + batch_size]

    print(
        f"Adding chunks "
        f"{i + 1}-{min(i + batch_size, len(splits))}"
        f"/{len(splits)}"
    )

    vs.add_documents(batch)

print("✅ All chunks embedded and stored!")

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 5. Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not found in the context, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


# --------------------------------------------------
# 6. LLM
# --------------------------------------------------

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


# --------------------------------------------------
# 7. Format retrieved documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# 8. RAG chain
# --------------------------------------------------

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()


# --------------------------------------------------
# 9. Ask questions
# --------------------------------------------------

print("\nPDF RAG ready.")
print("Ask a question (Ctrl+C to exit).")

while True:

    q = input("\nQ: ")

    if not q.strip():
        continue

    try:
        ans = chain.invoke(q.strip())
        print("\nA:", ans)

    except Exception as e:
        print("\nError:", e)