# script1_upload_qa_to_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings

# Your Q&A data
documents = [
    {
        "id": 1,
        "text": "Question: What is cryptocurrency? Answer: Cryptocurrency is a digital or virtual form of money that uses cryptography for security and operates on decentralized networks, typically based on blockchain technology. Bitcoin, created in 2009, was the first cryptocurrency."
    },
    {
        "id": 2,
        "text": "Question: How are new cryptocurrencies created? Answer: New cryptocurrencies can be created through processes like mining (e.g., solving complex mathematical problems to validate transactions, as with Bitcoin) or by launching a new token via an Initial Coin Offering (ICO) or token generation event on an existing blockchain like Ethereum."
    },
    {
        "id": 3,
        "text": "Question: What is a crypto wallet? Answer: A crypto wallet is a software program or physical device that stores private and public keys, allowing users to send, receive, and manage their cryptocurrencies. Wallets can be hot (online) or cold (offline) depending on their connectivity."
    },
    {
        "id": 4,
        "text": "Question: Why do cryptocurrency prices fluctuate so much? Answer: Cryptocurrency prices fluctuate due to factors like supply and demand, market speculation, regulatory news, technological developments, and macroeconomic trends. Their volatility is often amplified by the relatively small market size compared to traditional assets."
    }
]

# Initialize embedding model
model_name = "BAAI/bge-large-en"
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

def upload_to_qdrant():
    collection_name = "example_kb"
    
    # Recreate collection to ensure it exists with correct config
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' recreated successfully")
    except Exception as e:
        print(f"Error recreating collection: {e}")
        return

    # Process documents and upload
    points = []
    print("Generating embeddings...")
    for doc in documents:
        embedding = embeddings.embed_query(doc["text"])
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=embedding,
                payload={"text": doc["text"]}
            )
        )
    print(f"Prepared {len(points)} points for upload")

    # Upload points to Qdrant
    try:
        qdrant_client.upload_points(
            collection_name=collection_name,
            points=points,
            wait=True 
        )
        print(f"Uploaded {len(points)} points to Qdrant collection '{collection_name}'")
    except Exception as e:
        print(f"Error uploading points: {e}")
        return

    # Verify the upload
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        print(f"Collection status: {collection_info.status}")
        print(f"Points count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error checking collection: {e}")

if __name__ == "__main__":
    upload_to_qdrant()