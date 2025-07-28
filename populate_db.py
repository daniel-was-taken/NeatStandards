import time
import os
from pathlib import Path
from dotenv import load_dotenv
from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes, clean_dashes, group_broken_paragraphs
from langchain_unstructured import UnstructuredLoader
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from langchain_nebius import NebiusEmbeddings
from pydantic import SecretStr
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Milvus client and collection setup
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI)
collection_name = "my_rag_collection"

# Initialize embedding model
# embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embedding_model = NebiusEmbeddings(
    api_key=SecretStr(os.getenv("NEBIUS_API_KEY", os.getenv("OPENAI_API_KEY"))),
    model="Qwen/Qwen3-Embedding-8B",
    base_url="https://api.studio.nebius.ai/v1"
)



def emb_text(text):
    """Generate embeddings for text using the sentence transformer model."""
    return embedding_model.embed_query(text)
    # return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def emb_text_batch(texts):
    """Generate embeddings for multiple texts in batch - more efficient."""
    return embedding_model.embed_documents(texts)

def process_embeddings_in_batches(texts_to_embed, batch_size=50):
    """Process embeddings in batches with error handling and fallback."""
    all_embeddings = []
    
    print(f"Generating embeddings in batches of {batch_size}...")
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_end = min(i + batch_size, len(texts_to_embed))
        
        print(f"Processing embedding batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size} (documents {i+1}-{batch_end})")
        
        try:
            batch_embeddings = emb_text_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Add a small delay between batches to be respectful to the API
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            print("Falling back to individual processing for this batch...")
            
            # Fallback to individual processing for this batch
            for j, text in enumerate(batch_texts):
                try:
                    embedding = emb_text(text)
                    all_embeddings.append(embedding)
                    print(f"  Individual embedding {i+j+1} completed")
                    time.sleep(2)  # Longer delay for individual requests
                except Exception as individual_error:
                    print(f"  Failed to process document {i+j+1}: {individual_error}")
                    # Use a zero vector as fallback
                    all_embeddings.append([0.0] * 4096)
    
    return all_embeddings

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
    
    # Prepare texts for batch processing
    texts_to_embed = []
    doc_data = []
    
    print(f"Preparing {len(docs)} documents for batch processing...")
    
    for i, doc in enumerate(docs):
        # Check text length and truncate if necessary
        text_content = doc.page_content
        if len(text_content) > 65000:  # Leave some buffer below 64KB limit
            text_content = text_content[:65000]
            print(f"Document {i+1} truncated from {len(doc.page_content)} to {len(text_content)} characters")
        
        texts_to_embed.append(text_content)
        doc_data.append({
            "id": i,
            "text": text_content,
            "metadata": doc.metadata if doc.metadata else {}
        })
        
        # Print progress every 500 documents
        if (i + 1) % 500 == 0:
            print(f"Prepared {i + 1}/{len(docs)} documents")
    
    # Process embeddings in batches
    all_embeddings = process_embeddings_in_batches(texts_to_embed, batch_size=25)  # Smaller batch size for better reliability
    
    # Prepare data for insertion
    data_to_insert = []
    
    print(f"Preparing {len(doc_data)} documents for Milvus insertion...")
    
    for i, (doc_info, embedding) in enumerate(zip(doc_data, all_embeddings)):
        data_entry = {
            "id": doc_info["id"],
            "vector": embedding,
            "text": doc_info["text"],
            "metadata": doc_info["metadata"]
        }
        data_to_insert.append(data_entry)
        
        # Print progress every 500 documents
        if (i + 1) % 500 == 0:
            print(f"Prepared {i + 1}/{len(doc_data)} entries for insertion")
    
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