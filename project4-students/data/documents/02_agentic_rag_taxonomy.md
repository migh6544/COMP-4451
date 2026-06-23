# Agentic RAG Taxonomy

## RAG Paradigms
Naive RAG uses a simple retrieve-read workflow, often with keyword search or static top-k retrieval. Advanced RAG improves retrieval with dense embeddings, reranking, and iterative or multi-hop search. Modular RAG decomposes the system into replaceable components such as loaders, chunkers, retrievers, rerankers, generators, and evaluators. Graph RAG adds entity and relationship structure to support relational reasoning. Agentic RAG adds an autonomous control layer that decides when to retrieve, how to reformulate, which tools to use, and when the answer is ready.

## Components of an Agent
An agentic RAG system typically contains an LLM with a defined role and task, short-term memory for the current interaction, possible long-term memory or persistent state, planning or reflection mechanisms, and tools such as vector search, keyword search, SQL, APIs, web search, or calculators.

## Agentic Design Patterns
Reflection lets a model critique and revise its output. Planning lets a model decompose a complex task into smaller steps. Tool use lets the model extend beyond static pretrained knowledge. Multi-agent collaboration distributes specialized tasks across agents, such as one agent for retrieval, one for critique, and one for synthesis.

## Workflow Patterns
Prompt chaining executes fixed sequential stages. Routing sends different queries to different specialized pipelines. Parallelization runs independent subtasks at the same time. Orchestrator-worker patterns use a central controller to delegate subtasks dynamically. Evaluator-optimizer loops generate an answer, evaluate it, and refine it until quality improves or a stopping condition is reached.

## Architecture Types
Single-agent Agentic RAG uses one controller to route retrieval and synthesis. Multi-agent RAG assigns specialized retrieval or reasoning tasks to different agents. Hierarchical Agentic RAG uses top-level agents to coordinate lower-level agents. Corrective RAG evaluates retrieved documents and refines poor retrieval. Adaptive RAG chooses whether to use no retrieval, single-step retrieval, or multi-step retrieval based on query complexity. Graph-based Agentic RAG combines graph retrieval with unstructured document retrieval.

## Trade-Offs
Agentic RAG improves adaptability, context coverage, and multi-step reasoning. It also increases orchestration complexity, latency, cost, and evaluation difficulty. The best architecture depends on query type, corpus structure, and required reliability.
