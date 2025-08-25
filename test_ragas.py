from datasets import Dataset
from app import retrieve_relevant_documents, emb_text, model, embedding_model
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda
from langchain_core.documents import Document


def setup_standalone_rag_chain():
    """Setup a standalone RAG chain for testing without Chainlit session."""
    
    def get_context_and_history(inputs):
        """Retrieve context without session history."""
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

        return {
            "question": query,
            "context": doc_objects,
            "history": []  # Empty history for testing
        }
    
    system_prompt = """You are a helpful assistant specialising in developing non-discriminatory competence standards and disability support, reasonable adjustments, and equality legislation.

When answering questions, you should:
1. Use the provided context documents to inform your response
2. Be accurate and helpful
3. If the context doesn't contain relevant information, say so clearly
4. Always reply in English
5. Provide clear recommendations and examples wherever applicable
6. Do not make assumptions about the user's knowledge or background
7. If the user asks for a specific law or regulation, provide a brief explanation and cite relevant documents if available.
8. Do not overemphasize disability in your responses, but rather focus on the support and adjustments that can be made to ensure equality and inclusivity.
9. If the user query explicitly asks for a disability-related topic, provide a well-informed response based on the context documents.

Context documents:
{context} 

Please provide a clear response using the above context
"""

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

# Setup the RAG chain
rag_chain = setup_standalone_rag_chain()

questions = ["What is a 'reasonable adjustment'?", 
             "To whom do competence standards apply?",
             "Do competence standards vary by subject?",
            ]
ground_truths = [
    """The reasonable adjustments duty contains three requirements, which relate to changing
how things are done, changing the built environment to avoid such a substantial
disadvantage and providing auxiliary aids and services. Specifically:
1. A duty to make reasonable adjustments to any provision, criterion or practice (PCP)
which places disabled students at a substantial disadvantage
2. A duty to make reasonable adjustments to physical features
3. A duty to provide auxiliary aids (including services) """,

    """The Act does not specify to whom competence standards may be applied but it is clear that 
anti-discrimination provisions apply to prospective and current students (and in some cases 
former students). 
Providers commonly apply competence standards to:13 
+ Applicants, to determine whether they have the knowledge and skills necessary to 
participate in and complete a course of study 
+ Students, to determine whether they are ready to progress to the next year/stage of
study, and to determine whether they have demonstrated the requirements in order to be
awarded a qualification that necessitates a competence standard to be applied.""",

    """Competence standards can - and should - vary between courses of study.  What may 
constitute a competence standard in one subject area may not be justifiable in another. """]


answers = []
contexts = []

def clean_answer(answer):
    """Remove <think></think> tags and content from the answer."""
    import re
    # Remove everything between <think> and </think> tags, including the tags themselves
    cleaned = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL)
    return cleaned.strip()

# Inference
for query in questions:
    # Get answer from the RAG chain
    answer = rag_chain.invoke({"question": query})
    # Clean the answer to remove thinking content
    cleaned_answer = clean_answer(answer)
    answers.append(cleaned_answer)
    
    # Get relevant documents for context
    relevant_docs = retrieve_relevant_documents(query, limit=5)
    context_texts = [doc['text'] for doc in relevant_docs]
    contexts.append(context_texts)

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "reference": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    llm=model,
    embeddings=embedding_model,
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

# df = result.to_pandas()

evaluation_results = result.to_pandas()

display_columns = ['user_input', 'answer_relevancy', 'faithfulness', 'context_precision', 'context_recall']
formatted_results = evaluation_results[display_columns].to_markdown(index=False, numalign="left", stralign="left")

print(formatted_results)
