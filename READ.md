# RAG PDF Chatbot

A  Retrieval-Augmented Generation (RAG) chatbot built from scratch using Python.

This project demonstrates the core architecture of a RAG system without using frameworks such as LangChain or LlamaIndex. The application reads a PDF document, extracts its text, splits the text into chunks, generates embeddings, stores the embeddings in a FAISS vector index, retrieves the most relevant chunks for a user query, and uses an LLM to generate a context-grounded answer.

## Overview

Large Language Models (LLMs) have powerful language-generation capabilities, but they do not automatically have access to information contained in private documents.

Retrieval-Augmented Generation addresses this problem by retrieving relevant information from an external knowledge source and providing that information to the LLM as context.

This project implements the following RAG pipeline:

```text
PDF Document
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Embedding Generation
     |
     v
FAISS Vector Index
     |
     |
User Query
     |
     v
Query Embedding
     |
     v
Similarity Search
     |
     v
Relevant Chunks
     |
     v
LLM + Retrieved Context
     |
     v
Final Answer



Key Features
- PDF document ingestion
- Text extraction using pypdf
- Character-based text chunking
- Chunk overlap for improved context preservation
- Semantic embedding generation
- 384-dimensional embeddings using Sentence Transformers
- FAISS vector similarity search
- Top-K document retrieval
- Query embedding
- Context-aware LLM responses
- Environment-variable based API key management
- Command-line interface


Technology Stack
Component	Technology
Language	Python
PDF Processing	pypdf
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Search	FAISS
Numerical Processing	NumPy
LLM	OpenAI API
Environment Variables	python-dotenv
IDE	VS Code
Version Control	Git / GitHub
