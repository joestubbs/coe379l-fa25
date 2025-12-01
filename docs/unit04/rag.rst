Introduction to RAG 
=====================

In this lecture, we introduce Retrival Augmented Generation (RAG) and its usefulness over LLMs. 
We'll motivate this with a brief demo of the Graph RAG application.

Limitations of LLMS without RAG?
------------------------------------

**Static Knowledge**
LLMs are trained on large-scale corpora that are static and fixed at training time. They do not have access to private, proprietary, or newly updated data during inference unless explicitly integrated through mechanisms like RAG.

**Hallucinations**
When unsure, LLMs may generate incorrect but confident responses, especially on domain-specific or technical questions.

**No Source Traceability**
Answers are not grounded in provable sources—no citations or evidence.

**Hard to Update Knowledge**
Updating model knowledge typically requires fine-tuning or retraining, which is costly and not scalable.


Motivation behind RAG
-----------------------
LLMs are powerful language processors but not reliable knowledge retrieval systems. RAG bridges this gap.”

* RAG combines LLM reasoning with real-time retrieval from documents, databases, or web sources.

* It Improves factual accuracy by grounding answers in retrieved text.

* Makes knowledge dynamically updatable—just update your document store or embeddings.

* Reduces memory pressure, allowing even smaller models to perform better when paired with retrieval.

* Practical for real-world use cases that require up-to-date or organization-specific information.


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



RAG Components
==============

.. figure:: ./images/RAG.png
    :width: 310px
    :align: left

User Query
----------
The input question or request provided by the user.  
It triggers both the retrieval of relevant context and the generation process.

Retriever
---------
Finds the most relevant information from an external knowledge source.  
Typically performs a semantic similarity search using embeddings. 
An embedding is a way of converting text (or other data like images or audio) into a numerical vector—a list of numbers—that captures its meaning and relationships in a high-dimensional space.

Index / Vector Store
--------------------
A specialized database that stores embeddings of documents.  
Allows fast retrieval of conceptually similar content based on the query. Some of the examples of vectore are ChromaDB, Neo4j, FAISS (Facebook AI Similarity Search)

Embedding Model
---------------
Transforms text into high-dimensional vectors that capture meaning.  
Used both during document ingestion and at query time.

LLM (Generator)
---------------
Produces the final response using the retrieved information and input query.  
Relies on its internal reasoning but grounds the answer in external sources.

Prompt Constructor
------------------
Builds a structured input combining the user query and retrieved content.  
Ensures the LLM uses context effectively and follows task instructions.

Post-Processing *(Optional)*
----------------------------
Refines, formats, or validates the LLM's response before returning it.  
May include citation extraction, reranking, or response safety checks.

Document Store
--------------
Stores original knowledge sources such as PDFs, websites, or datasets.  
Content is pre-processed and indexed for retrieval via the vector store.

Feedback / Monitoring
---------------------
Tracks system performance and answer quality to enable continuous improvement.  
Used to tune retrieval accuracy, update data, or refine prompting strategies.
