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
st.markdown("""
<style>
.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
}

[data-testid="metric-container"]{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,.1);
}
</style>
""", unsafe_allow_html=True)
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

    # Create embeddings
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
    page_title="PDF RAG Chatbot",
    layout="centered"
)

st.markdown("""
<div class="title">
🤖 PDF RAG Chatbot
</div>

<div class="subtitle">
Ask questions from any PDF using Gemini AI
</div>
""", unsafe_allow_html=True)
with st.sidebar:

    st.title("📚 PDF RAG")

    st.markdown("---")

    st.success("Gemini AI")

    st.success("FAISS")

    st.success("LangChain")

    st.success("Sentence Transformers")

    st.markdown("---")

    st.info("Upload a PDF and ask questions.")

st.subheader("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "",
    type=["pdf"]
)
if uploaded_file:

    with st.spinner("Processing PDF..."):
        db, chunk_count = create_vector_store(uploaded_file.getvalue())

    st.success("✅ PDF processed successfully!")
   c1,c2,c3 = st.columns(3)

c1.metric("Chunks",chunk_count)

c2.metric("Model","Gemini")

c3.metric("Retriever","FAISS")

    st.divider()
   st.subheader("💬 Ask Your Question")

    question = st.chat_input("Ask anything about your PDF...")

    if question:
        with st.chat_message("user"):
             st.write(question)
        with st.spinner("Thinking..."):

            retriever = db.as_retriever(search_kwargs={"k": 3})

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. "
                        "Answer ONLY using the provided context. "
                        "If the answer is not found in the context, reply with 'I don't know.'"
                    ),
                    (
                        "human",
                        "Context:\n{context}\n\nQuestion:\n{question}"
                    )
                ]
            )

            chain = prompt | llm

            response = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

      with st.chat_message("assistant"):
    st.write(response.content)
