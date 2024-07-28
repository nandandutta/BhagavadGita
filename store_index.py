from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import os
from pinecone import Pinecone, ServerlessSpec

# Set your Pinecone API key and environment directly in the script
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "4961199f-ac64-44c4-9fda-f2decb00ac27")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "us-east-1")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, if not create it
index_name = "bhagavadgita"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Replace with the actual dimension of your embeddings
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_API_ENV
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Load PDF and split text
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Use the correct method to obtain embeddings
vectors = embeddings.embed_documents([t.page_content for t in text_chunks])
ids = [f"doc_{i}" for i in range(len(text_chunks))]

# Split vectors into smaller batches
batch_size = 1000  # Adjust batch size as needed
for i in range(0, len(vectors), batch_size):
    batch_ids = ids[i:i + batch_size]
    batch_vectors = vectors[i:i + batch_size]
    # Upsert vectors into Pinecone index
    index.upsert(vectors=list(zip(batch_ids, batch_vectors)))

print("Indexing completed.")

