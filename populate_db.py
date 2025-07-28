import time
import os
from pathlib import Path
from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes, clean_dashes, group_broken_paragraphs
from langchain_unstructured import UnstructuredLoader
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from langchain_nebius import NebiusEmbeddings
from pydantic import SecretStr
import os

# Initialize Milvus client and collection setup
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI)
collection_name = "my_rag_collection"

# Initialize embedding model
# embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_model = NebiusEmbeddings(
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")),
    model="Qwen/Qwen3-Embedding-8B" 
)



def emb_text(text):
    """Generate embeddings for text using the sentence transformer model."""
    return embedding_model.embed_query(text)
    # return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def create_collection():
    """Create collection if it doesn't exist."""
    if milvus_client.has_collection(collection_name):
        milvus_client.load_collection(collection_name=collection_name)
        return
    
    # Create Milvus collection schema
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=4096)  # Qwen/Qwen3-Embedding-8B dimension
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)  # Maximum allowed for VARCHAR
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    # Create index for vector search
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="AUTOINDEX",
    )

    # Create and load collection
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    milvus_client.load_collection(collection_name=collection_name)

# Document directory
directory_path = "data/"  

def main():
    """Main function to load documents and insert them into Milvus."""
    create_collection()
    
    # Check if collection already has data
    stats = milvus_client.get_collection_stats(collection_name)
    if stats['row_count'] > 0:
        print(f"Collection already contains {stats['row_count']} documents. Skipping insertion.")
        return
    
    docs = unstructured_document_loader()
    
    # Prepare data for insertion
    data_to_insert = []
    
    print(f"Processing {len(docs)} documents for insertion...")
    
    for i, doc in enumerate(docs):
        # Check text length and truncate if necessary
        text_content = doc.page_content
        if len(text_content) > 65000:  # Leave some buffer below 64KB limit
            text_content = text_content[:65000]
            print(f"Document {i+1} truncated from {len(doc.page_content)} to {len(text_content)} characters")
        
        # Generate embedding for the document content
        embedding = emb_text(text_content)
        
        # Prepare the data entry
        data_entry = {
            "id": i,
            "vector": embedding,
            "text": text_content,
            "metadata": doc.metadata if doc.metadata else {}
        }
        
        data_to_insert.append(data_entry)
        
        # Print progress every 100 documents
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(docs)} documents")
    
    print(f"Inserting {len(data_to_insert)} documents into Milvus...")
    
    # Insert data into Milvus
    insert_result = milvus_client.insert(
        collection_name=collection_name,
        data=data_to_insert
    )
    
    print(f"Successfully inserted {insert_result['insert_count']} documents")
    print(f"Primary keys: {insert_result['ids'][:10]}...")  # Show first 10 IDs
    
    return docs

def unstructured_document_loader():
    """Load documents using UnstructuredLoader."""
    # Collect file paths for PDF, DOCX, and HTML files
    file_extensions = ["*.pdf", "*.docx", "*.html"]
    file_paths = []

    for ext in file_extensions:
        file_paths.extend(Path(directory_path).glob(ext))

    # Convert Path objects to strings
    file_paths = [str(file) for file in file_paths]
    
    # Configure UnstructuredLoader with post-processors
    loader = UnstructuredLoader(
        file_paths, 
        chunking_strategy="by_title", 
        include_orig_elements=False,
        post_processors=[
            clean_extra_whitespace, 
            replace_unicode_quotes, 
            clean_dashes, 
            group_broken_paragraphs
        ]
    )
    
    docs = loader.load()
    print(f"Number of LangChain documents: {len(docs)}")
    print(f"Length of first document: {len(docs[0].page_content)} characters")
    print(f"First document preview: {docs[0].page_content[:200]}...")
    
    return docs

def verify_insertion():
    """Verify that data was successfully inserted into Milvus."""
    # Get collection statistics
    stats = milvus_client.get_collection_stats(collection_name)
    print(f"Collection stats: {stats}")
    
    # Test search functionality with a sample query
    test_query = "Questions by staff to other staff"
    test_embedding = emb_text(test_query)
    
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[test_embedding],
        limit=3,
        output_fields=["text", "metadata"]
    )
    
    print(f"\nTest search results for '{test_query}':")
    for i, result in enumerate(search_results[0]):
        print(f"Result {i+1}:")
        print(f"  Score: {result['distance']:.4f}")
        print(f"  Text preview: {result['entity']['text'][:200]}...")
        print(f"  Metadata: {result['entity']['metadata']}")
        print("-" * 50)


if __name__ == "__main__":
    start_time = time.time()
    
    print("="*60)
    print("STARTING DOCUMENT PROCESSING AND MILVUS INSERTION")
    print("="*60)
    
    main()
    
    print("\n" + "="*50)
    print("VERIFYING DATA INSERTION")
    print("="*50)
    verify_insertion()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")