# Competence Standards RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to assist in writing competence standards in higher education. This application uses Chainlit for the user interface, Milvus for vector storage, and Nebius AI for embeddings and chat completion.

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Starting the Application](#starting-the-application)
- [Database Population](#database-population)
- [Data Management](#data-management)
- [Testing](#testing)

## 🎯 About the Project

This RAG chatbot is specifically designed to assist with competence standards in higher education settings. It provides intelligent responses based on a curated knowledge base of documents.

The system uses advanced natural language processing to understand queries and retrieve relevant information from the document corpus, providing contextually appropriate responses.

## ✨ Features

- **Interactive Chat Interface**: Built with Chainlit for an intuitive user experience
- **Vector Search**: Powered by Milvus for efficient similarity search
- **Advanced Embeddings**: Uses Nebius AI Qwen3-Embedding-8B model
- **Document Processing**: Supports multiple formats (PDF, DOCX, HTML)
- **Authentication**: Integrated with Chainlit's authentication system
- **Persistent Storage**: Database integration with PostgreSQL
- **Containerized Deployment**: Docker Compose setup for easy deployment
- **Testing Suite**: Comprehensive testing with RAGAS evaluation


## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/daniel-was-taken/prod-rag-chat.git
   cd prod-rag-chat
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Create environment file**
   ```bash
   cp .env.example .env  # If available, or create manually
   ```

2. **Configure environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```env
   # Nebius AI Configuration
   NEBIUS_API_KEY=your_nebius_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Fallback
   
   # Milvus Configuration
   MILVUS_URI=http://localhost:19530
   
   # Chainlit Configuration
   CHAINLIT_AUTH_SECRET=your_auth_secret_here
   
   # Database Configuration (if using PostgreSQL)
   DATABASE_URL=postgresql://user:password@localhost:5432/dbname
   ```

## 🎬 Starting the Application

### Method 1: Using Docker Compose (Recommended)

1. **Start the infrastructure services**
   ```bash
   docker-compose up -d
   ```
   This will start:
   - Milvus vector database
   - etcd (for Milvus coordination)
   - MinIO (for Milvus storage)

2. **Wait for services to be ready** 

3. **Start the Chainlit application**
   ```bash
   chainlit run app.py -w
   ```

### Method 2: Manual Setup

1. **Install and configure Milvus standalone**
   Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)

2. **Start the application**
   ```bash
   chainlit run app.py -w
   ```

The application will be available at `http://localhost:8000`

## 📊 Database Population

The system automatically populates the vector database on first startup. However, you can manually manage the data:

### Automatic Population

The application automatically checks if the Milvus collection exists and has data. If not, it runs the population script automatically.

### Manual Population

To manually populate or repopulate the database:

```bash
python populate_db.py
```

### Adding New Documents

1. **Add documents to the data directory**
   ```bash
   # Place your documents in the data/ folder
   cp your_new_document.pdf data/
   cp your_new_document.docx data/
   ```

2. **Supported file formats:**
   - PDF files (`.pdf`)
   - Microsoft Word documents (`.docx`)
   - HTML files (`.html`)

3. **Repopulate the database**
   ```bash
   # Delete existing collection
   python delete_collection.py
   
   # Repopulate with new documents
   python populate_db.py
   ```

### Database Configuration

The population script uses the following configuration:

- **Embedding Model**: Qwen/Qwen3-Embedding-8B (4096 dimensions)
- **Chunk Size**: 1500 characters maximum
- **Combine Threshold**: 200 characters minimum
- **Batch Size**: 5 documents per batch
- **Collection Name**: `my_rag_collection`

### Document Processing Pipeline

1. **Loading**: Documents are loaded using UnstructuredLoader
2. **Cleaning**: Text is cleaned and normalized
3. **Chunking**: Documents are split into manageable chunks
4. **Embedding**: Chunks are converted to vector embeddings
5. **Storage**: Embeddings are stored in Milvus with metadata

## 🗃️ Data Management


### Deleting the Collection

```bash
python delete_collection.py
```


### Updating Documents

To update the document corpus:

1. Add/remove documents in the `data/` directory
2. Delete the existing collection: `python delete_collection.py`
3. Restart the application (it will automatically repopulate)

## 🧪 Testing

The project includes comprehensive testing:

### Running Unit Tests

```bash

# Run specific test files
python -m unittest tests/test_chainlit.py -v
```

### RAGAS Evaluation

Evaluate the RAG system performance:

```bash
# Run RAGAS evaluation
python tests/test_ragas.py

# Or use the Jupyter notebook
jupyter notebook tests/test_ragas.ipynb
```

### Manual Testing

Test individual components:

```bash
# Test vector search functionality
python tests/test_vector_search.py

```


## 📄 License

This project is licensed under the MIT license.


