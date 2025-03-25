# script1_upload_qa_to_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Egypt history text (combined paragraphs)
egypt_text = (
    "Egypt boasts one of the world's oldest and most influential civilizations, with its history stretching back over 5,000 years. Ancient Egypt emerged along the Nile River around 3100 BCE, when Upper and Lower Egypt unified under the first pharaoh, Narmer. This marked the beginning of a remarkable era defined by the construction of monumental pyramids, such as those at Giza, and the development of hieroglyphic writing, advanced mathematics, and a complex polytheistic religion. The Old Kingdom (c. 2686–2181 BCE), Middle Kingdom (c. 2055–1650 BCE), and New Kingdom (c. 1550–1070 BCE) saw Egypt flourish as a powerful empire, with iconic rulers like Hatshepsut, Akhenaten, and Ramses II shaping its legacy. After centuries of foreign rule—including by the Persians, Greeks under Alexander the Great, and Romans—Egypt became a center of Islamic culture following the Arab conquest in 641 CE. Today, its ancient wonders and rich history continue to captivate the world. "
    "Egypt’s history took a significant turn following the decline of the New Kingdom around 1070 BCE, entering a period known as the Third Intermediate Period (c. 1070–664 BCE). During this time, Egypt faced internal divisions and foreign invasions, including rule by Libyan and Nubian dynasties. The once-mighty civilization struggled to maintain its sovereignty, eventually falling under the control of the Assyrian Empire in the 7th century BCE. This era of instability paved the way for the Late Period (c. 664–332 BCE), during which Egypt briefly regained its independence under native rulers like the Saite kings before succumbing to Persian domination in 525 BCE. The arrival of Alexander the Great in 332 BCE marked the end of Persian rule and the beginning of the Ptolemaic dynasty, a Greek-speaking lineage that ruled Egypt after Alexander’s death. The most famous Ptolemaic ruler, Cleopatra VII, attempted to restore Egypt’s power through alliances with Rome, but her defeat in 31 BCE by Octavian (later Augustus) led to Egypt’s annexation as a Roman province, ending its era as an independent kingdom. "
    "In the centuries that followed, Egypt transformed into a vital part of the Roman and later Byzantine Empires, serving as a breadbasket due to its fertile Nile Delta. Christianity spread rapidly, with Alexandria becoming a major center of early Christian thought until the Arab conquest in 641 CE ushered in Islamic rule under the Rashidun Caliphate. This shift marked the beginning of Egypt’s integration into the Islamic world, with Cairo—founded in 969 CE by the Fatimid dynasty—emerging as a cultural and political hub. Over the medieval period, Egypt was governed by a succession of Muslim dynasties, including the Ayyubids and Mamluks, who fortified its defenses against Crusaders and Mongols. The Ottoman Empire absorbed Egypt in 1517, though it retained a degree of autonomy under local rulers. In the 19th century, Muhammad Ali Pasha, an ambitious Ottoman governor, modernized Egypt, laying the groundwork for its emergence as a semi-independent state. British occupation in 1882 shifted control again, lasting until Egypt gained formal independence in 1922, though foreign influence persisted until the 1952 revolution established a republic, cementing its modern identity."
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Max characters per chunk
    chunk_overlap=50  # Overlap to maintain context
)

# Split the text into chunks
chunks = text_splitter.split_text(egypt_text)
documents = [{"id": i + 1, "text": chunk} for i, chunk in enumerate(chunks)]

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
    collection_name = "example_Egypt"
    
    # Recreate collection to ensure it exists with correct config
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,  # Used for BAAI/bge-large-en-v1.5
                distance=models.Distance.COSINE
            )# openAi uses default 1534 vectors
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