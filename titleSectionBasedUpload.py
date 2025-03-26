# script1_upload_bitcoin_to_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Bitcoin history data with titles
bitcoin_data = [
    {
        "title": "Bitcoin’s Historical Journey",
        "text": (
            "As we venture further into Bitcoin's timeline, it's crucial to note that its history is not just a series of technological advancements but also a compelling narrative of how society's approach to money and finance is evolving. "
            "From its mysterious origins to its modern-day impact, Bitcoin's historical journey offers invaluable lessons in innovation, resilience, and the ever-changing dynamics of value and trust. Let's delve into the key moments that have defined this digital currency's fascinating story."
        )
    },
    {
        "title": "The Precursors to Bitcoin",
        "text": (
            "Before Bitcoin became a reality, the idea of digital money had been toyed with for years. Concepts like 'bit gold' and 'b-money' were formulated but never fully developed. These prototypes, while not as successful, paved the way for what was to come."
        )
    },
    {
        "title": "The Enigmatic Creator: Satoshi Nakamoto",
        "text": (
            "In 2008, an individual or group under the pseudonym 'Satoshi Nakamoto' published the Bitcoin whitepaper titled 'Bitcoin: A Peer-to-Peer Electronic Cash System.' This groundbreaking document presented a solution to the double-spending problem, enabling transactions without a central authority."
        )
    },
    {
        "title": "Bitcoin's First Steps (2009-2010)",
        "text": (
            "Genesis Block: On January 3, 2009, the first-ever Bitcoin block was mined, marking the birth of Bitcoin's blockchain. "
            "First Transaction: Later that year, Satoshi sent 10 BTC to computer scientist Hal Finney, marking the first Bitcoin transaction. "
            "Bitcoin Pizza Day: In 2010, a user traded 10,000 BTC for two pizzas, giving Bitcoin its first tangible value. Bitcoin Pizza Day is celebrated annually on May 22 and marks the anniversary of the first-ever real-world Bitcoin transaction."
        )
    },
    {
        "title": "Growing Pains and Recognition (2011-2012)",
        "text": (
            "Altcoins: With Bitcoin's success, other cryptocurrencies, known as altcoins, began to emerge. Litecoin, one of the earliest, claimed faster transaction speeds. "
            "Silk Road Controversy: Bitcoin's anonymity features became popular on the Silk Road, a dark web marketplace. This association brought scrutiny but also heightened interest in the currency. "
            "Bitcoin Foundation: To standardize and promote Bitcoin, industry members formed the Bitcoin Foundation in 2012."
        )
    },
    {
        "title": "Adoption and Challenges (2013-2016)",
        "text": (
            "All-time Highs: Bitcoin reached $1,000 for the first time in late 2013, driven by growing adoption and media attention. "
            "Mt. Gox Disaster: Once the world's largest Bitcoin exchange, Mt. Gox filed for bankruptcy in 2014 after losing around 850,000 BTC, shaking trust in the ecosystem. "
            "Regulatory Scrutiny: As Bitcoin's popularity surged, regulators worldwide started examining the cryptocurrency, leading to both challenges and legitimacy."
        )
    },
    {
        "title": "Mainstream Acceptance (2017-Present)",
        "text": (
            "Bull Run of 2017: Bitcoin's price soared to almost $20,000 by the end of 2017, driven by retail and institutional interest. "
            "Scaling Solutions: Recognizing Bitcoin's scalability issues, the community developed solutions like the Lightning Network to handle more transactions. "
            "Institutional Adoption: Corporate giants like Tesla and MicroStrategy began adding Bitcoin to their balance sheets, marking a significant shift in its acceptance."
        )
    },
    {
        "title": "Bitcoin Price Trajectory",
        "text": (
            "Bitcoin's price trajectory has been nothing short of a roller coaster ride since its inception. Beginning as a virtual unknown, valued at mere pennies, it saw its first surge in 2011, crossing the $1 threshold. Several highs and lows followed, shaped by regulatory news, technological developments, and market sentiment. The landmark moment came in late 2017 when it peaked near $20,000, drawing global attention. "
            "In late 2021, the price of BTC reached approximately $64,000. However, in 2022, Bitcoin experienced notable price volatility, with its value dropping from its peak and reaching a low of around $16,000. By 2023, it had stabilized at approximately $26,000."
        )
    }
]

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Max characters per chunk
    chunk_overlap=50  # Overlap to maintain context
)

# Split each section's text into chunks and create documents
documents = []
id_counter = 1
for section in bitcoin_data:
    chunks = text_splitter.split_text(section["text"])
    for chunk in chunks:
        documents.append({
            "id": id_counter,
            "title": section["title"],
            "text": chunk
        })
        id_counter += 1

# Initialize embedding model
model_name = "BAAI/bge-large-en-v1.5"
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
    collection_name = "bitcoin_kb"
    
    # Recreate collection to ensure it exists with correct config
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,  # Matches BAAI/bge-large-en-v1.5
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
                payload={
                    "title": doc["title"],
                    "text": doc["text"]
                }
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
    
    