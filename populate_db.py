import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings
from langchain_unstructured import UnstructuredLoader
from pydantic import SecretStr
from pymilvus import MilvusClient, DataType
from unstructured.cleaners.core import (
    clean_extra_whitespace, 
    replace_unicode_quotes
)

# Load environment variables
load_dotenv()

# Configuration constants
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "my_rag_collection"
DOCUMENT_DIR = "data/"
EMBEDDING_DIMENSION = 4096
TEXT_MAX_LENGTH = 65000
CHUNK_SIZE = 100
BATCH_SIZE = 5

# Chunking configuration
MAX_CHARACTERS = 1500
COMBINE_TEXT_UNDER_N_CHARS = 200

# Initialize clients
milvus_client = MilvusClient(uri=MILVUS_URI)

embedding_model = NebiusEmbeddings(
    api_key=SecretStr(os.getenv("NEBIUS_API_KEY", os.getenv("OPENAI_API_KEY"))),
    model="Qwen/Qwen3-Embedding-8B",
    base_url="https://api.studio.nebius.ai/v1"
)

def clean_text(text):
    """Simple text cleaning for educational documents."""
    import re
    
    # Basic cleaning without problematic functions
    text = clean_extra_whitespace(text)
    text = replace_unicode_quotes(text)
    
    # Simple normalizations
    text = re.sub(r'[\r\n]+', ' ', text)  # Convert newlines to spaces
    text = re.sub(r'\s+', ' ', text)      # Multiple spaces to single space
    
    return text.strip()


def generate_embedding(text):
    """Generate embedding for a single text."""
    return embedding_model.embed_query(text)


def generate_embeddings_batch(texts):
    """Generate embeddings for multiple texts efficiently."""
    return embedding_model.embed_documents(texts)


def process_embeddings_in_batches(texts, batch_size=BATCH_SIZE):
    """Process embeddings in batches with error handling."""
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Generating embeddings in {total_batches} batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches}")
        
        try:
            batch_embeddings = generate_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            time.sleep(1.5)  # API rate limiting
            
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}. Processing individually...")
            
            for j, text in enumerate(batch_texts):
                try:
                    embedding = generate_embedding(text)
                    all_embeddings.append(embedding)
                    time.sleep(2)
                except Exception as individual_error:
                    print(f"Failed to process document {i+j+1}: {individual_error}")
                    all_embeddings.append([0.0] * EMBEDDING_DIMENSION)
    
    return all_embeddings

def create_collection():
    """Create Milvus collection if it doesn't exist."""
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.load_collection(collection_name=COLLECTION_NAME)
        return
    
    # Create collection schema
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    # Create vector index
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="AUTOINDEX",
    )

    # Create and load collection
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    milvus_client.load_collection(collection_name=COLLECTION_NAME)

def load_documents():
    """Load documents from the data directory."""
    file_extensions = ["*.pdf", "*.docx", "*.html"]
    file_paths = []

    for ext in file_extensions:
        file_paths.extend(Path(DOCUMENT_DIR).glob(ext))

    file_paths = [str(file) for file in file_paths]
    
    loader = UnstructuredLoader(
        file_paths, 
        chunking_strategy="by_title",
        include_orig_elements=False
    )
    
    docs = loader.load()
    print(f"Loaded {len(docs)} initial documents")
    
    # Apply additional cleaning and chunking
    final_chunks = []
    
    for doc in docs:
        # Clean text
        cleaned_text = clean_text(doc.page_content)
        
        # Skip very short chunks
        if len(cleaned_text) < 50:
            continue
            
        # Split if too large
        if len(cleaned_text) <= MAX_CHARACTERS:
            doc.page_content = cleaned_text
            final_chunks.append(doc)
        else:
            # Split large chunks on sentence boundaries
            chunks = _split_large_chunk(cleaned_text, doc.metadata)
            final_chunks.extend(chunks)
    
    print(f"Final processed chunks: {len(final_chunks)}")
    if final_chunks:
        avg_length = sum(len(doc.page_content) for doc in final_chunks) / len(final_chunks)
        print(f"Average chunk length: {avg_length:.0f} characters")
    
    return final_chunks


def _split_large_chunk(text, metadata):
    """Split large text into smaller chunks."""
    from langchain.schema import Document
    
    chunks = []
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        potential_chunk = current_chunk + sentence + '. '
        
        if len(potential_chunk) > MAX_CHARACTERS and len(current_chunk) > COMBINE_TEXT_UNDER_N_CHARS:
            if current_chunk.strip():
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata=metadata.copy()
                ))
            current_chunk = sentence + '. '
        else:
            current_chunk = potential_chunk
    
    # Add remaining content
    if current_chunk.strip():
        chunks.append(Document(
            page_content=current_chunk.strip(),
            metadata=metadata.copy()
        ))
    
    return chunks


def prepare_document_data(docs, start_idx=0):
    """Prepare document data for insertion."""
    texts_to_embed = []
    doc_data = []
    
    for i, doc in enumerate(docs):
        text_content = doc.page_content
        if len(text_content) > TEXT_MAX_LENGTH:
            text_content = text_content[:TEXT_MAX_LENGTH]
            print(f"Document {start_idx + i + 1} truncated to {TEXT_MAX_LENGTH} characters")
        
        texts_to_embed.append(text_content)
        doc_data.append({
            "id": start_idx + i,
            "text": text_content,
            "metadata": doc.metadata or {}
        })
    
    return texts_to_embed, doc_data


def process_document_chunk(docs, chunk_idx, chunk_num, total_chunks):
    """Process a single chunk of documents."""
    print(f"\nProcessing chunk {chunk_num}/{total_chunks}")
    
    # Prepare document data
    texts_to_embed, doc_data = prepare_document_data(docs, chunk_idx)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts_to_embed)} documents...")
    embeddings = process_embeddings_in_batches(texts_to_embed)
    
    # Prepare data for insertion
    data_to_insert = []
    for doc_info, embedding in zip(doc_data, embeddings):
        data_to_insert.append({
            "id": doc_info["id"],
            "vector": embedding,
            "text": doc_info["text"],
            "metadata": doc_info["metadata"]
        })
    
    # Insert into Milvus
    insert_result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    return insert_result['insert_count']

def main():
    """Main function to process and insert documents into Milvus."""
    create_collection()
    
    # Check if collection already has data
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)
    if stats['row_count'] > 0:
        print(f"Collection already contains {stats['row_count']} documents. Skipping insertion.")
        return
    
    # Load documents
    docs = load_documents()
    if not docs:
        print("No documents found to process.")
        return
    
    # Process documents in chunks
    total_docs = len(docs)
    total_chunks = (total_docs + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_inserted = 0
    
    print(f"Processing {total_docs} documents in {total_chunks} chunks of {CHUNK_SIZE}")
    
    for chunk_idx in range(0, total_docs, CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, total_docs)
        chunk_num = chunk_idx // CHUNK_SIZE + 1
        current_chunk = docs[chunk_idx:chunk_end]
        
        # Process chunk
        chunk_inserted = process_document_chunk(current_chunk, chunk_idx, chunk_num, total_chunks)
        total_inserted += chunk_inserted
        
        print(f"Chunk {chunk_num} complete: {chunk_inserted} docs inserted")
        print(f"Progress: {total_inserted}/{total_docs} ({(total_inserted/total_docs)*100:.1f}%)")
        
        # Memory cleanup
        del current_chunk
        if chunk_num < total_chunks:
            time.sleep(2)
    
    print(f"\nSuccessfully processed {total_inserted} documents!")


def verify_insertion():
    """Verify that data was successfully inserted into Milvus."""
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)
    print(f"Collection stats: {stats}")
    
    # Test search functionality
    test_query = "Why should reasonable adjustments be made?"
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
        print(f"  Text preview: {result['entity']['text'][:200]}...")
        print(f"  Metadata: {result['entity']['metadata']}")
        print("-" * 50)


if __name__ == "__main__":
    start_time = time.time()
    
    print("Starting document processing and Milvus insertion")
    print("=" * 60)
    
    main()
    
    print("\nVerifying data insertion")
    print("=" * 30)
    verify_insertion()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")