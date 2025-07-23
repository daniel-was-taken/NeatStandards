import os
from typing import Dict, List, Optional
from operator import itemgetter

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from pydantic import SecretStr


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_nebius import ChatNebius
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from chainlit.input_widget import Select, Switch, Slider

from langchain_core.documents import Document
from typing_extensions import List

from populate_db import main

# Initialize Milvus client and embedding model
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI)
collection_name = "my_rag_collection"

# Initialize collection once at startup
if not milvus_client.has_collection(collection_name):
    main()
else:
    # Check if collection has data, populate if empty
    stats = milvus_client.get_collection_stats(collection_name)
    if stats['row_count'] == 0:
        main()
    milvus_client.load_collection(collection_name=collection_name)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize LLM
model = ChatNebius(
    model="Qwen/Qwen3-14B",
    streaming=True,
    temperature=0.7,
    top_p=0.95,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")),
)

# Define application steps

def emb_text(text: str) -> List[float]:
    """Generate embeddings for text using the sentence transformer model."""
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def retrieve_relevant_documents(query: str, limit: int = 5) -> List[Dict]:
    """Retrieve relevant documents from Milvus based on the query."""
    try:
        query_embedding = emb_text(query)
        search_results = milvus_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["text", "metadata"]
        )
        documents = []
        for result in search_results[0]:
            doc_info = {
                "text": result['entity']['text'],
                "metadata": result['entity']['metadata'],
                "score": result['distance']
            }
            documents.append(doc_info)

        return documents
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def format_docs_with_id(docs: List[Dict]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        # Extract title and page_number from metadata, with fallbacks
        metadata = doc.get('metadata', {})
        title = metadata.get('filename', 'Unknown Document')  # Use filename as fallback for title
        page_number = metadata.get('page_number', 'Unknown')
        score = doc.get('score', 'N/A')  # Use score if available
        text_content = doc.get('text', '')

        formatted_doc = f"[{i+1}] Source: {title}, Page: {page_number}, Score: {score}\nContent: {text_content}"
        formatted.append(formatted_doc)
    
    print(f"Formatted documents: {formatted}")
    return "\n\n".join(formatted)


def setup_rag_chain():
    """Setup the RAG chain with context retrieval."""
    
    def get_context_and_history(inputs):
        """Retrieve context and get conversation history."""
        query = inputs["question"]
        relevant_docs = retrieve_relevant_documents(query, limit=5)
        print("Relevant documents:", relevant_docs[0] if relevant_docs else "No documents found")
        
        # Convert dictionaries to Document objects for LangChain
        doc_objects = []
        for doc in relevant_docs:
            doc_obj = Document(
                page_content=doc.get('text', ''),
                metadata=doc.get('metadata', {})
            )
            doc_objects.append(doc_obj)

        # Format citations for reference
        citations = format_docs_with_id(relevant_docs)
        
        # Add citations to the last document's metadata so it's available to the prompt
        if doc_objects:
            doc_objects[-1].metadata['formatted_citations'] = citations

        return {
            "question": query,
            "context": doc_objects,
            "history": cl.user_session.get("messages", [])
        }
    system_prompt = """You are a helpful assistant specialising in disability support, reasonable adjustments, and equality legislation.

When answering questions, you should:
1. Use the provided context documents to inform your response
2. Be accurate and helpful
3. Cite relevant documents in the format [1], [2], etc.
4. If the context doesn't contain relevant information, say so clearly
5. Always reply in English
6. Provide clear recommendations wherever applicable
7. Do not make assumptions about the user's knowledge or background
8. If the user asks for a specific law or regulation, provide a brief explanation and cite relevant documents if available.
9. Do not overlook the importance of accessibility and inclusivity in your responses.
10. Do not overemphasize disability in your responses, but rather focus on the support and adjustments that can be made to ensure equality and inclusivity.
11. If the user asks about a specific disability, provide general information and resources, but do not make assumptions about the individual's experience or needs.
12. If the user query explicitly asks for a disability-related topic, provide a well-informed response based on the context documents.

Context documents:
{context} 

Please provide a clear response using the above context"""

    # Get the current settings to check if Think mode is enabled
    settings = cl.user_session.get("settings", {})
    use_think = settings.get("Think", True)  # Default to True as per the initial setting
    
    if not use_think:
        system_prompt = '/no_think ' + system_prompt

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    
    # Use a custom chain that properly handles our context and history
    def process_input_and_format(inputs):
        context_data = get_context_and_history(inputs)
        return {
            "context": context_data["context"],
            "question": context_data["question"],
            "history": context_data["history"]
        }
    
    chain = RunnableLambda(process_input_and_format) | question_answer_chain
    
    return chain


# ============== Application Setup ==============


# Authentication
@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("admin", os.getenv("PASSWORD")):
        return cl.User(
            identifier="admin",
            metadata={"role": "admin", "provider": "credentials"}
        )
    return None

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.PersistedUser]:
    return default_user

# Starters
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Considerations for Autistic People",
            message="What considerations should be made for autistic people?",
        ),
        cl.Starter(
            label="Explain Equality Act 2010",
            message="Explain the Equality Act 2010 in simple terms.",
        ),
    ]


# Chat lifecycle
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Switch(id="Think", label="Use Deep Thinking", initial=True),
        ]
    ).send()

    # Store initial settings
    cl.user_session.set("settings", {"Think": True})  # Set the default value

    """Initialize chat session with RAG chain."""
    chain = setup_rag_chain()
    cl.user_session.set("chain", chain)
    cl.user_session.set("messages", [])


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    # Store the settings in the user session so they can be accessed in setup_rag_chain
    cl.user_session.set("settings", settings)
    
    # Update the chain with the new settings
    chain = setup_rag_chain()
    cl.user_session.set("chain", chain)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume chat with conversation history."""
    messages = []
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    
    for message in root_messages:
        if message["type"] == "user_message":
            messages.append(HumanMessage(content=message["output"]))
        else:
            messages.append(AIMessage(content=message["output"]))

    cl.user_session.set("messages", messages)
    

    chain = setup_rag_chain()
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    
    """Handle incoming messages with RAG and conversation history."""
    chain = cl.user_session.get("chain")
    messages = cl.user_session.get("messages", [])
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["</think> "]
    )
    
    try:
        # Get the relevant documents for citations
        relevant_docs = retrieve_relevant_documents(message.content, limit=5)
        citations = format_docs_with_id(relevant_docs)
        
        answer = await chain.ainvoke({"question": message.content}, config=RunnableConfig(callbacks=[cb]))
        
        async with cl.Step(name="References") as step:
            if relevant_docs:
                step.output = citations
            else:
                step.output = "No relevant documents found for this query."

        # Update conversation history
        messages.append(HumanMessage(content=message.content))
        messages.append(AIMessage(content=answer))

        cl.user_session.set("messages", messages)
        
    except Exception as e:
        await cl.Message(content=f"Sorry, I encountered an error: {str(e)}").send()
