More on RAG 
============

In the previous RAG lecture, we saw how combining retrieval with a language model can produce more accurate, context-specific responses. 
In this follow-up, we dig deeper: we’ll unpack the main architectural components of a full-featured RAG system, understand why and how documents are processed before retrieval, explore more advanced retrieval paradigms (like graph-based RAG), and survey modern frameworks that make building RAG systems easier.

By the end of this module, students should be able to:

* Understand each component in a RAG pipeline;
* Learn preprocessing for your documents (splitting, chunking, embedding, indexing);
* Recognize when to use vector-based or graph-based retrieval;
* Use frameworks to build maintainable, extensible RAG applications;


Examples of Retrieval-Augmented Generation (RAG)
-------------------------------------------------

**Healthcare and Research Assistants**

* Query medical literature and summarize latest research papers.
* Retrieve treatment guidelines from proprietary clinical data.

**Customer Support Chatbots**

* Pull answers from company documentation or ticket history.
* Use retrieved context to generate accurate and personalized support responses.

**Enterprise Knowledge Assistants**

* Access internal wikis, design docs, and project repositories.
* Help employees query domain-specific knowledge without searching manually.

**Legal Document Analysis**

* Extract relevant clauses from long contracts.
* Generate summaries and answer compliance-related questions.

**Code Assistants**

* Answer questions using project-specific codebases and API docs.
* Help developers with accurate debugging suggestions based on the repository.



Core Components of a RAG System 
================================

RAG empowers an LLM to produce grounded, context-aware responses by pulling in information from outside sources. 
This is made possible through a set of architectural components that operate together in a defined workflow. Let's take a closer look at each compoment.

.. figure:: ./images/RAG.png
    :width: 310px
    :align: center


Data sources and knowledge bases
---------------------------------
The knowledge base can both contain structured and unstructured data. It could be in a form of csv files with rows and columns
or it could be a text document like PDF or HTML files. You can also connect your knowledge base to streaming data sources or APIs.

A raw document (e.g., a long PDF, research paper, book chapter) may be thousands of words — far exceeding what an embedding model or LLM can reasonably handle at once.
If you treat entire documents as single units, retrieval will likely retrieve entire documents — which are large, and may contain a lot of irrelevant data, covering the relevant information.
Instead, if you divide documents into smaller, semantically coherent **chunks** of text (e.g. 500–2,000 characters, or number-of-sentences), then retrieval returns just a handful of chunks relevant to the query. This improves both precision (less irrelevant text) and efficiency (fewer tokens, faster inference).
Thus — **document splitting** or **chunking** is fundamental to making RAG practical, scalable, and precise.

In the last lecture, we used several Tapis code snippets as documents and generated embeddings for them.
In this session, we will learn how to split these documents into smaller chunks by grouping sentences into fixed-size token blocks, preparing them for more efficient retrieval and RAG processing.

.. code-block:: python3 

    import re

    def chunk_text(text, max_tokens=100):
        # Simple sentence split
        sentences = re.split(r'(?<=[.!?]) +', text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence exceeds the limit, start a new chunk
            if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


.. code-block:: python3 

        tapis_documents = [
        "Tapis is an NSF-funded web-based API framework for securely managing computational workloads across infrastructure and institutions, so that experts can focus on their research instead of the technology needed to accomplish it.",
        "As part of work funded by the National Science Foundation starting in 2019, Tapis is delivering a version 3 (“v3”) of its platform with several new capabilities, including a multi-site Security Kernel, Streaming Data APIs, and first-class support for containerized applications.",
        "Python code for generating a Tapis token: from tapipy.tapis import Tapis ...",  # shortened for example
        ]

        # Step 1: Chunk the document
        chunks = chunk_text(doc)
        print(chunks)


.. code-block:: python3 

    Output - > ['Tapis is an NSF-funded web-based API framework for securely managing computational workloads across infrastructure and institutions, so that experts can focus on their research instead of the technology needed to accomplish it. As part of work funded by the National Science Foundation starting in 2019, Tapis is delivering a version 3 (“v3”) of its platform with several new capabilities, including a multi-site Security Kernel, Streaming Data APIs, and first-class support for containerized applications.ine.']

**Embeddings (Vectorization)**
Embeddings are the backbone of retrieval.Each chunk is turned into a numerical vector (an embedding) capturing semantic meaning.
You can pull the embedding model by execing into the Ollama container and and doing a 
**olama pull nomic-embed-text**
The code below generates an embedding vector for a given text chunk using a local Ollama API. This embedding can then be used in a RAG system for semantic search or retrieval.


.. code-block:: python3 

    import requests
    import numpy as np

    def embed_with_ollama(text, model="nomic-embed-text"):
        """
        Generate embeddings for a text chunk using Ollama local API.
        Ollama must be running on localhost:11434.
        """
        url = "http://172.17.0.1:11434/api/embed"  
        payload = {
            "model": model,
            "input": text
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()  # This is already a Python dict

        # Access embeddings directly
        return np.array(data["embeddings"][0])  # single embedding vector

To view chunks and its associated embedding you can use the code below

.. code-block:: python3 

    def process_document_for_rag(text, chunk_size=100, embedding_model="nomic-embed-text"):
        """
        1. Chunk the document
        2. Generate embeddings for each chunk
        3. Return a list of dicts: {"chunk": text, "embedding": vector}
        """
        chunks = chunk_text(text, max_tokens=chunk_size)
        embedded_chunks = []

        for chunk in chunks:
            emb = embed_with_ollama(chunk, model=embedding_model)
            embedded_chunks.append({
                "chunk": chunk,
                "embedding": emb
            })

        return embedded_chunks


.. code-block:: python3 

    if __name__ == "__main__":
    tapis_documents = [
        "Tapis is an NSF-funded web-based API framework ...",
        "As part of work funded by the National Science Foundation ...",
        "Python code for generating a Tapis token: from tapipy.tapis import Tapis ..."
    ]

    rag_chunks = []

    for doc in tapis_documents:
        chunks = process_document_for_rag(doc, chunk_size=20)
        rag_chunks.extend(chunks)

    for i, c in enumerate(rag_chunks):
        print(f"Chunk {i+1}:\n{c['chunk']}\nEmbedding: {(c['embedding'])}\n")


Index / Vector Store
--------------------
These embeddings are then stored in a specialized database that can supporting efficient similarity queries (nearest neighbors, etc.). This enables rapid retrieval of relevant chunks given a query. 
Some of the examples of vectore are ChromaDB, Neo4j, FAISS (Facebook AI Similarity Search)

User Query
----------
The input question or request provided by the user.  
It triggers both the retrieval of relevant context and the generation process.

Retriever
---------
Finds the most relevant information from an external knowledge source.  
Typically performs a semantic similarity search using embeddings. 
An embedding is a way of converting text (or other data like images or audio) into a numerical vector—a list of numbers—that captures its meaning and relationships in a high-dimensional space.

LLM (Generator)
---------------
The model (e.g., GPT-style, or other LLM) receives the assembled prompt (query + context) and generates the response. Because of the external context, the output is more likely to be factually grounded and relevant.


Graph RAG
==========
While the **classical** RAG pipeline (documents → split → embed → vector store → retrieve → LLM) works for many tasks, there are scenarios where more sophisticated retrieval / knowledge integration approaches are beneficial.

Graph-based RAG (Graph RAG)
Graph RAG involves representing knowledge not just as isolated chunks, but as nodes and edges — e.g., entities, relationships, concepts — forming a knowledge graph. Retrieval then becomes graph traversal + semantic search rather than just nearest-neighbor in vector space.
This is especially useful when your data is structured, relational, or has complex interdependencies (e.g., medical records, relational databases, multimodal data, or domains where relationships matter as much as entities).
Recent research such as DSRAG: A Domain‑Specific Retrieval Framework Based on Document-derived Multimodal Knowledge Graph shows how combining a multimodal knowledge graph (text, tables, images) with semantic search can improve domain-specific question answering. 
arXiv
Graph RAG can help with context coherence, relational reasoning, and reduce hallucinations by grounding answers in structured relationships rather than free-text chunks.

.. figure:: ./images/graph_rag.png
    :width: 310px
    :align: left

This knowledge graph shows 42 nodes and 81 edges (relationships) between the nodes. 
Demo LLM-Builder

Frameworks and Tools to build RAG Applications
======================================

`LangChain <https://docs.langchain.com/oss/python/langchain/overview>`_ : The most popular framework that make building RAG applications super easy. 
------------
LangChain is a widely used framework for building RAG pipelines, chatbots, summarization tools, and more. It provides abstractions for document loading, splitting, embedding generation, vector store integration, retrievers, prompt templates, LLM wrappers, and chaining of operations.
LangaChain can be integrated with OpenAI, Anthropic, `Sambanova <https://sambanova.ai/blog/tacc-deploys-sambanova-suite-ai-inference-for-scientific-research>`

.. code-block:: python3 

    !pip install langchain-sambanova

.. code-block:: python3 

    import getpass
    import os

    if not os.getenv("SAMBANOVA_API_KEY"):
        os.environ["SAMBANOVA_API_KEY"] = getpass.getpass("Enter your SambaNova API key: ")

.. code-block:: python3 

    import os
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://tejas.tacc.utexas.edu/v1/60709c21-409c-44a5-8a9d-1638fad5d5a6",
    )

    # Function to get a response for a user question
    def ask_question(question: str):
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-405B-Instruct',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            top_p=0.1
        )
        return response.choices[0].message.content

    # Example: single user input
    user_question = input("Enter your question: ")
    answer = ask_question(user_question)
    print("\nAssistant:", answer)
        

Output -> 
Enter your question:  what is Tapis?

Assistant: Tapis can refer to different things, but here are a few possible meanings:

1. Tapis (textile): Tapis is a type of traditional textile art form that originated in Southeast Asia, particularly in Indonesia and Malaysia. It is a kind of woven cloth, often made from silk or cotton, that features intricate designs and patterns. Tapis textiles are highly valued for their beauty and cultural significance.

2. Tapis (software): Tapis is also the name of a software framework designed for building scalable, distributed applications. It is an open-source platform that provides a set of tools and APIs for developing data-intensive applications, particularly in the fields of science and engineering.

3. Tapis (French word): In French, "tapis" means "carpet" or "rug". It can also refer to a tapestry or a woven wall hanging.

Without more context, it's difficult to determine which definition is most relevant. If you have any additional information or clarification, I'd be happy to try and provide a more specific answer.

