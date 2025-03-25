# script2_query_qdrant.py
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model (must match the upload script)
model_name = "BAAI/bge-large-en-v1.5"  # Using the same model as upload
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Qdrant client with your cloud cluster
qdrant_client = QdrantClient(
    url="https://b97a3564-c617-4e7e-89f6-ee19b78d5066.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.0xWFxe2UIYoO4ct4On-feLJ5ELE2jv3A8QZ8Zp1dG-o"
)
print("Connected to Qdrant client")

def search_qdrant(query_text, limit=2):
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query_text)
    print(f"Query embedding size: {len(query_embedding)}")  # Debug to confirm size

    # Search the collection
    try:
        search_results = qdrant_client.search(
            collection_name="example_Egypt",
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        # Display results
        print(f"\nQuery: '{query_text}'")
        print(f"Top {limit} results:")
        for result in search_results:
            print(f"ID: {result.id}")
            print(f"Score: {result.score:.4f}")
            print(f"Text: {result.payload['text']}")
            print("---")
        return search_results
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return None

if __name__ == "__main__":
    queries = [
       "When did Ancient Egypt begin?",
        "Who was Cleopatra?",
        "How did Egypt become part of the Islamic world?"
    ]
    
    for query in queries:
        search_qdrant(query, limit=2)