# RAG Metrics Interpretation Guide

## Context Relevance
Context relevance measures whether the retrieved chunks are useful for answering the question. Low context relevance usually points to a retrieval failure. A model can generate a fluent answer, but if the retrieved context is wrong or incomplete, correctness will suffer.

## Faithfulness
Faithfulness measures whether the answer's claims are supported by the retrieved context. Faithfulness is not the same as correctness. A system can be perfectly faithful to the wrong retrieved passage and still be incorrect against the reference answer. A system can also give a true answer but receive lower faithfulness if the supporting sentence was not included in the retrieved context.

## Answer Correctness
Answer correctness compares the generated answer against the reference or gold answer. It is often the most discriminating metric because it penalizes missing claims, wrong numbers, omitted list items, and incorrect comparisons. In the Week 10 benchmark, correctness separated the stronger pipelines more clearly than faithfulness or semantic similarity.

## Semantic Similarity
Semantic similarity measures embedding-level closeness between the answer and the reference. It is useful as a rough topical signal but can be misleading. Wrong legal numbers, incomplete lists, or partially answered questions can still produce high semantic similarity because the answer and reference share vocabulary and topic.

## ROUGE-1 F1
ROUGE-1 F1 measures lexical unigram overlap. It is weak as a standalone quality signal in RAG. Correct paraphrases and well-organized explanations can have low ROUGE, while incomplete answers can score moderately if they share key words with the reference.

## Abstention
Abstention should be evaluated separately from answer quality. A good RAG system should abstain when retrieval fails. However, an over-eager abstention gate can zero out judged metrics even when the generated text overlaps the reference and is substantially correct.

## Common Failure Patterns
Enumeration questions fail when the model summarizes instead of extracting every item. Comparative questions fail when the system does not retrieve evidence for both sides and perform a set difference. Legal or policy questions fail when the generator substitutes common prior knowledge, such as a typical thirty-day notice period, for the specific number in the source. These failures should be addressed with better retrieval coverage and stricter prompts.
