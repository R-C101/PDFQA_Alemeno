import os
import uuid
import torch
import shutil
import hashlib
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def load_local_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Load local embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector_stores(base_directory='./chroma_dbs'):
    """
    Load or create multiple Chroma vector stores for different PDF collections.
    """
    embeddings = load_local_embeddings()
    
    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)
    
    # Dictionary to store vector stores for different document collections
    vector_stores = {
        'google': {
            'store': Chroma(
                embedding_function=embeddings,
                persist_directory=os.path.join(base_directory, 'google')
            ),
            'keywords': ['google', 'alphabet', 'search', 'engine', 'cloud','youtube', 
                         'android', 'chrome', 'maps', 'pixel', 'gmail', 'drive', 
                         'assistant', 'play', 'google', 'tech', 'technology', 'googles', "google's"]
        },
        'tesla': {
            'store': Chroma(
                embedding_function=embeddings,
                persist_directory=os.path.join(base_directory, 'tesla')
            ),
            'keywords': ['tesla', 'elon', 'musk', 'electric', 'vehicle', 'car', 'teslas', 'tesla\'s']
        },
        'uber': {
            'store': Chroma(
                embedding_function=embeddings,
                persist_directory=os.path.join(base_directory, 'uber')
            ),
            'keywords': ['uber', 'ride', 'sharing',  'taxi', 'transportation','ubers', 'uber\'s']
        }
    }
    
    return vector_stores

def add_docs_to_index(vector_store, docs_directory):
    """
    Load PDF documents and add them to the specified vector store.
    """
    # Load documents
    loader = PyPDFDirectoryLoader(docs_directory)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=500
    )
    texts = text_splitter.split_documents(documents)
    
    # Add to vector store
    vector_store.add_documents(texts)
    
    return len(texts)

def choose_vector_store(question: str, vector_stores: Dict) -> Tuple[Chroma, str]:
    """
    Choose the appropriate vector store based on keywords in the question.
    Returns the vector store and the chosen category.
    """
    # Lowercase the question for case-insensitive matching
    lower_question = question.lower()
    
    # Split the question into words
    question_words = set(lower_question.split())
    
    # Check for keyword matches
    best_match = None
    max_keyword_matches = 0
    
    for category, data in vector_stores.items():
        # Count keyword matches
        keyword_matches = len(set(data['keywords']) & question_words)
        
        if keyword_matches > max_keyword_matches:
            max_keyword_matches = keyword_matches
            best_match = category
    
    # If no specific keywords found, return the first vector store as default
    if best_match is None:
        best_match = list(vector_stores.keys())[0]
    
    return vector_stores[best_match]['store'], best_match

def load_local_llm(model_name='meta-llama/Llama-2-7b-chat-hf'):
    """
    Load a local LLM using Hugging Face Transformers.
    Requires login to access Llama models.
    """
    # Ensure you have the right access and login
    hf = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={"temperature": 0, "max_new_tokens": 300}
    )
    
    return hf

def get_answer(question: str, base_directory: str, llm, embeddings: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Retrieve context from the most appropriate vector store and generate answer.
    """
    # Load vector stores
    vector_stores = load_vector_stores(base_directory, embeddings)
    
    # Choose the most appropriate vector store
    vector_store, chosen_category = choose_vector_store(question, vector_stores)
    
    # Add documents to the chosen index if not already done
    docs_directory = os.path.join('pdfs', chosen_category)
    
    # Index documents if the vector store is empty
    if vector_store._collection.count() == 0:
        add_docs_to_index(vector_store, docs_directory)
    
    # Create prompt template
    prompt_template = """
    Use the following context to answer the question as detailed as possible. 
    If the answer is not in the context, say "Answer is not available in the context".

    Context:
    {context}

    Question: 
    {question}

    Helpful Answer:
    """

    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=['context', 'question']
    )

    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
        chain_type_kwargs={'prompt': prompt}
    )

    # Generate response
    result = qa_chain({'query': question})
    retrieved_docs = qa_chain.retriever.get_relevant_documents(query=question) 
    
    return result['result'], retrieved_docs, chosen_category
