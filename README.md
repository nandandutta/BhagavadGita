# 🎈📖 Bhagavad Gita Assistant

Welcome to the Bhagavad Gita Assistant on LLAMA 2. Ask your questions and get insightful answers based on the Bhagavad Gita.

Project Details

Project Name: Bhagavad Gita Assistant 
Creator: Nandan

Mechanism

The assistant retrieves relevant text from a pre-indexed database of the Bhagavad Gita using Pinecone. It then uses an AI model to generate a response based on the retrieved text and the user's query.

Logic

User Query: The user inputs a query.
Semantic Search: The query is used to perform a semantic search on a vector database (Pinecone) containing pre-indexed chunks of the Bhagavad Gita text.
Retrieve Similar Chunks: The search retrieves chunks of text that are semantically similar to the user's query.
Generate Response: The retrieved chunks, along with the user query, are sent to the AI model (LLAMA 2) to generate a final response based on the Bhagavad Gita.

Tech Used

Streamlit: For the web interface.
LangChain: For prompt templates and chains.
Pinecone: For vector search and retrieval.
CTransformers: For loading and using the AI model (LLAMA 2).
Hugging Face: For text embeddings.

This project aims to provide insightful answers based on the Bhagavad Gita by using advanced AI models and vector search technologies.

### How to run it on your own machine

Download LLM Model From Hugging face

meta-llama/Meta-Llama-3.1-8B


1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run app.py
   ```
