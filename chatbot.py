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

    texts = [
        doc.page_content.strip()
        for doc in chunks
        if doc.page_content.strip()
    ]

    if len(texts) == 0:
        st.error("No readable text found in the PDF.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(texts, embeddings)

    return db, len(chunks)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="PDF Q&A",
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

            retriever = db.as_retriever(
                search_kwargs={"k": 3}
            )

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0
            )

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

            response = chain.invoke({
                "context": context,
                "question": question
            })

        st.markdown("### ✅ Answer")
        st.write(response.content)
