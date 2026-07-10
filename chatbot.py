import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# VECTOR STORE CREATION (CACHED)
# -----------------------------
@st.cache_resource
def create_vector_store(_file_bytes):
    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(_file_bytes)

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc.page_content for doc in chunks]
    db = FAISS.from_texts(texts, embeddings)

    return db, len(chunks)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="PDF Q&A (Gemini)",
    layout="centered"
)

st.title("📄 PDF → FAISS → Gemini Q&A")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        db, chunk_count = create_vector_store(uploaded_file.getvalue())

    st.success("✅ PDF processed successfully!")
    st.write(f"📦 Total chunks created: **{chunk_count}**")

    st.divider()
    st.subheader("💬 Ask questions about the PDF")

    question = st.text_input("Enter your question")

    if question:
        with st.spinner("Thinking..."):
            retriever = db.as_retriever(search_kwargs={"k": 3})

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0
            )

            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful assistant. "
                    "Answer ONLY using the provided context. "
                    "If the answer is not in the context, say 'I don't know.'"
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{question}"
                )
            ])

            chain = prompt | llm
            try:
                response = chain.invoke({
                    "context": context,
                    "question": question
                })
            except Exception as e:
                st.exception(e)
                st.stop()

        st.markdown("### ✅ Answer")
        st.write(response.content)

