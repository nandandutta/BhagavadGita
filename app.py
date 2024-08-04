import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from typing import List
import streamlit as st

# Load environment variables
load_dotenv()

# Set Pinecone API key and environment
os.environ['PINECONE_API_KEY'] = '4961199f-ac64'  # Replace with your actual API key
os.environ['PINECONE_ENVIRONMENT'] = 'us-east-1'

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])

# Define index name and namespace
index_name = "bhagavadgita"
namespace = "2MAN3D"

# Connect to the index
index = pc.Index(index_name)

# Define a function to download embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the embeddings
embeddings = download_hugging_face_embeddings()

# Define a custom Pinecone retriever
class CustomPineconeRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Retrieve relevant documents from Pinecone
        return self.vectorstore.similarity_search(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# Load the index into PineconeVectorStore
docsearch = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
retriever = CustomPineconeRetriever(vectorstore=docsearch)

# Define a simple prompt template
PROMPT_TEMPLATE = """
You are Krishna from the Bhagavad Gita. Provide answers based only on the Bhagavad Gita. Do not use mythology or references outside of the Bhagavad Gita.

Context: {context}
Query: {query}

Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "query"]
)

# Initialize the LLM
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Create a simple LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=PROMPT
)

# Streamlit app
st.set_page_config(page_title="Bhagavad Gita Assistant", page_icon="📖", layout="wide")

st.title("📖 Bhagavad Gita Assistant")
st.markdown("Welcome to the Bhagavad Gita Assistant on LLAMA 2. Ask your questions and get insightful answers based on the Bhagavad Gita.")

# Tabs for Chat, Project Details, Mechanism, Logic, and Tech Used
tabs = st.tabs(["Chat", "Project Details", "Mechanism", "Logic", "Tech Used"])

with tabs[0]:
    # Input for the query
    user_query = st.text_input("Enter your query:", placeholder="e.g., Who is Arjuna?")
    submit_query = st.button("Submit")

    if submit_query and user_query:
        test_context = "You are Krishna from the Bhagavad Gita act as Krishna."
        
        try:
            # Retrieve relevant documents from Pinecone
            relevant_docs = retriever.get_relevant_documents(user_query)
            context_from_docs = " ".join([doc.page_content for doc in relevant_docs])
            enriched_context = test_context + " " + context_from_docs
            input_data = {"context": enriched_context, "query": user_query}

            # Call the LLMChain with the input data
            test_result = llm_chain(input_data)
            st.subheader("Response")
            st.write(test_result['text'])
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tabs[1]:
    st.header("Project Details")
    st.markdown("""
    **Project Name:** Bhagavad Gita Assistant
    **Creator:** Nandan
    
    This project aims to provide insightful answers based on the Bhagavad Gita by using advanced AI models and vector search technologies.
    """)

with tabs[2]:
    st.header("Mechanism")
    st.markdown("""
    The assistant retrieves relevant text from a pre-indexed database of the Bhagavad Gita using Pinecone. 
    It then uses an AI model to generate a response based on the retrieved text and the user's query.
    """)

with tabs[3]:
    st.header("Logic")
    st.markdown("""
    1. **User Query:** The user inputs a query.
    2. **Semantic Search:** The query is used to perform a semantic search on a vector database (Pinecone) containing pre-indexed chunks of the Bhagavad Gita text.
    3. **Retrieve Similar Chunks:** The search retrieves chunks of text that are semantically similar to the user's query.
    4. **Generate Response:** The retrieved chunks, along with the user query, are sent to the AI model (LLAMA 2) to generate a final response based on the Bhagavad Gita.
    """)

with tabs[4]:
    st.header("Tech Used")
    st.markdown("""
    - **Streamlit:** For the web interface.
    - **LangChain:** For prompt templates and chains.
    - **Pinecone:** For vector search and retrieval.
    - **CTransformers:** For loading and using the AI model (LLAMA 2).
    - **Hugging Face:** For text embeddings.
    """)

# Feedback section
st.markdown("---")
st.subheader("Feedback")
st.markdown("We value your feedback. Please leave your comments below:")

feedback = st.text_area("Your feedback:", placeholder="Write your comments here...")

if st.button("Submit Feedback"):
    if feedback:
        st.success("Thank you for your feedback!")
        st.markdown(f"**Feedback submitted:** {feedback}")
    else:
        st.error("Please write some feedback before submitting.")

# Comments section
st.markdown("---")
st.subheader("Comments")

# Simulated comments for display purposes
comments = [
    "This assistant is really helpful!",
    "I love how it integrates context from the Bhagavad Gita.",
    "The answers are very insightful."
]

for comment in comments:
    st.markdown(f"**User Comment:** {comment}")

# Add love symbol at the bottom
st.markdown("❤️ Made with love by Nandan")
