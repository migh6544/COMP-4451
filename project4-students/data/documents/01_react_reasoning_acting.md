# ReAct: Reasoning and Acting

## Core Idea
ReAct is a prompting and agent-control pattern that interleaves reasoning traces with actions and observations. A ReAct trajectory is commonly represented as Thought, Act, and Observation. The Thought step lets the language model plan, track progress, reformulate a search, identify missing evidence, or synthesize the next step. The Act step invokes a tool or performs a task-specific operation. The Observation step returns external information from the environment or retrieval system.

## Why ReAct Matters for RAG
Traditional chain-of-thought reasoning can be internally coherent but ungrounded. In a RAG system, this creates risk when the model reasons from memory instead of from the indexed sources. ReAct reduces this weakness by letting the model search, observe retrieved evidence, and update its plan. The loop supports the principle of reason to act and act to reason.

## ReAct Compared with CoT and Act-Only
Chain-of-thought uses reasoning but no external action. Act-only systems use tools but do not explicitly plan or explain their intermediate state. ReAct combines both. The result is more interpretable because developers can inspect the agent trajectory. It is also more diagnosable because failures can be assigned to poor search results, bad reasoning, loop repetition, or unsupported synthesis.

## Known Failure Modes
ReAct is not automatically better in every case. It can repeat prior thoughts or actions, follow poor search results, or over-generate when retrieval is noisy. It can also be more expensive than a single fixed retrieval pass because each ReAct step may require another language-model call. These trade-offs make ReAct useful when the query needs iterative evidence gathering, but less necessary for simple factual questions.

## Design Implication
A good ReAct RAG system should expose the retrieval trajectory, limit the step budget, ground final answers in retrieved context, and abstain when useful evidence is not found. ReAct should be paired with strict source-grounded prompts so the final answer does not drift beyond the retrieved chunks.
