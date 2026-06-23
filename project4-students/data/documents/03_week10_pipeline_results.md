# Week 10 RAG Pipeline Results

## Benchmark Setup
The Week 10 RAG evaluation used a forty-eight question benchmark over OpenAI policy documents. The pipelines were evaluated with shared components where applicable: a GPT-4o-mini generator, OpenAI text-embedding-3-large embeddings in the experimental runs, an MS MARCO MiniLM cross-encoder reranker, and a 400 token chunk size with 50 token overlap for the later pipelines.

## Hybrid Retrieval and Reranking
The no-agent hybrid retrieval pipeline used a fixed retrieval path. It combined vector retrieval and BM25-style keyword retrieval, fused candidates, reranked them with a cross-encoder, and generated an answer from the final top-k chunks. It achieved answer correctness around 0.746, answer relevance around 0.906, faithfulness around 0.908, semantic similarity around 0.859, and ROUGE-1 F1 around 0.563. The fixed top-k strategy was limited when evidence was split across multiple chunks or when a query required comparison.

## ReAct Baseline
The ReAct baseline used iterative retrieval. The model could reason, search, observe results, and decide when enough context had been collected. It achieved the strongest general-purpose results in the comparison: answer correctness around 0.830, answer relevance around 0.969, faithfulness around 0.917, semantic similarity around 0.867, and ROUGE-1 F1 around 0.573. It had the lowest abstention rate and an average cost near 3.4 pipeline LLM calls per query.

## ReAct with Query Decomposition
The ReAct plus decomposition pipeline first split complex questions into atomic subquestions. It then ran ReAct retrieval and focused generation for each subquestion before synthesizing a final answer. It improved context relevance to around 0.885 and performed well when decomposition was actually triggered. However, it increased cost to about 6.7 LLM calls per query and retrieved a broader chunk pool. Its overall correctness landed between the hybrid baseline and the simpler ReAct baseline.

## Main Conclusion
ReAct was the best general-purpose pipeline on this benchmark because it gave the best accuracy-to-cost balance. Decomposition was valuable for multi-part, comparative, or enumeration-heavy questions, but it should be selectively triggered rather than applied to every query.
