from pymilvus import MilvusClient
import os
# Initialize client
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI)
collection_name = "my_rag_collection"

# Delete the collection
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' has been deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")