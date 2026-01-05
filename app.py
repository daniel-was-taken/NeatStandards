import os
from typing import Dict, List, Optional
from operator import itemgetter
from dotenv import load_dotenv

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_nebius import ChatNebius

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from pymilvus import MilvusClient

from sentence_transformers import SentenceTransformer
from langchain_nebius import NebiusEmbeddings

from chainlit.input_widget import Select, Switch, Slider

from langchain_core.documents import Document
from typing_extensions import List

# from populate_db import main

# Initialize Milvus client and embedding model
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
milvus_client = MilvusClient(uri=MILVUS_URI, token=os.getenv("MILVUS_API_KEY"))
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

# embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

embedding_model = NebiusEmbeddings(
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")),
    model="Qwen/Qwen3-Embedding-8B" 
)


# Initialize LLM
model = ChatNebius(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    streaming=True,
    temperature=0.2,
    max_tokens=8192,
    top_p=0.95,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")),
)

# Define application steps

def emb_text(text: str) -> List[float]:
    """Generate embeddings for text using the sentence transformer model."""
    return embedding_model.embed_query(text)
    # return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

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
    system_prompt = """You are an expert assistant for staff in UK higher education institutions. Help develop inclusive, non-discriminatory competence standards that comply with UK equality legislation (for example: the Equality Act 2010). Advise on reasonable adjustments and support to remove barriers and promote fairness for all students.

Rules:
1. Use ONLY the provided context documents as your source of information: {context}
2. If the context does not contain relevant information, respond exactly:
   "I could not find relevant information about this topic in the provided documents."
3. Do not guess or include information from outside the provided documents.
4. Answer in clear, plain English. Define technical or legal terms when needed.
5. Provide practical, actionable guidance and examples for writing competence standards.
6. Emphasise removing barriers via reasonable adjustments and support; treat disability within the broader goal of equality and inclusivity.
7. Do not assume the user's prior knowledge; maintain a neutral, professional, and respectful tone.

Format requirements:
- Structure all responses using the RESPONSE TEMPLATE provided below.
- Use bolding for headers (e.g. **Summary**)
- Ensure there is a blank line before and after lists.

Response template:

**Summary**

[Insert a concise 1-3 sentence answer here]

**Key Guidance**

* [Actionable point 1]
* [Actionable point 2]
* [Actionable point 3]

"""

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
            label="Reviewing Existing Standards",
            message="How can we review existing competence standards to ensure they are inclusive?",
        ),
        cl.Starter(
            label="When No Adjustments are Possible",
            message="What should we do if a competence standard cannot be adjusted for a student?",
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
    # print("on_settings_update", settings)
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

    settings = await cl.ChatSettings(
        [
            Switch(id="Think", label="Use Deep Thinking", initial=True),
        ]
    ).send()

    # Store initial settings
    cl.user_session.set("settings", {"Think": True})  # Set the default value
    # TODO: # Reinitialize the chain with the current settings
    chain = setup_rag_chain()
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with RAG and conversation history."""
    chain = cl.user_session.get("chain")
    messages = cl.user_session.get("messages", [])
    
    # 1. Initialize callback with stream_final_answer=True
    # This automatically creates an empty message and streams tokens into it
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
    )
    
    try:
        # Get relevant documents first (fast)
        relevant_docs = retrieve_relevant_documents(message.content, limit=5)
        citations = format_docs_with_id(relevant_docs)
        
        # 2. Invoke the chain with the callback
        # The chain will stream chunks to 'cb', which updates the UI in real-time
        # We assign the final result to 'res' just to store it in history
        answer = await chain.ainvoke(
            {"question": message.content}, 
            config=RunnableConfig(callbacks=[cb])
        )
        
        await cl.Message(answer).send()
        
        # 'res' is usually a dict if the chain returns a dict, or a string.
        # Based on your StrOutputParser usage, it should be a string.
        # If your chain returns a dict, you might need to extract the text.
        # answer = res if isinstance(res, str) else res.get("output", "") or res.get("text", "")

        # 3. Add References as a Step (Collapsible element under the message)
        # Note: Since the message is already sent by the callback, we just append a step.
        async with cl.Step(name="References") as step:
            if relevant_docs:
                step.output = citations
            else:
                step.output = "No relevant documents found for this query."

        # 4. Update History
        messages.append(HumanMessage(content=message.content))
        messages.append(AIMessage(content=answer))
        cl.user_session.set("messages", messages)
        
    except Exception as e:
        await cl.Message(content=f"Sorry, I encountered an error: {str(e)}").send()
