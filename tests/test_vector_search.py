import os
import time
from langchain_nebius import NebiusEmbeddings
from pydantic import SecretStr
from pymilvus import MilvusClient
# Configuration constants
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "my_rag_collection"
DOCUMENT_DIR = "data/"
EMBEDDING_DIMENSION = 4096

milvus_client = MilvusClient(uri=MILVUS_URI)

TEXT_MAX_LENGTH = 65000
CHUNK_SIZE = 100
BATCH_SIZE = 5


embedding_model = NebiusEmbeddings(
    api_key=SecretStr(os.getenv("NEBIUS_API_KEY", os.getenv("OPENAI_API_KEY"))),
    model="Qwen/Qwen3-Embedding-8B",
    base_url="https://api.studio.nebius.ai/v1"
)

def generate_embedding(text):
    """Generate embedding for a single text."""
    return embedding_model.embed_query(text)

def verify_insertion():
    """Verify that data was successfully inserted into Milvus."""
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)
    print(f"Collection stats: {stats}")
    
    # Test search functionality
    test_query = "What are competence standards and their purpose?"
    test_embedding = generate_embedding(test_query)
    
    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[test_embedding],
        limit=3,
        output_fields=["text", "metadata"]
    )
    
    print(f"\nTest search results for '{test_query}':")
    for i, result in enumerate(search_results[0]):
        print(f"Result {i+1}:")
        print(f"  Score: {result['distance']:.4f}")
        print(f"  Text preview: {result['entity']['text'][:300]}...")
        print(f"  Metadata: {result['entity']['metadata']}")
        print("-" * 50)

if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
   
    print("\n Starting test search")
    print("=" * 30)
    verify_insertion()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")