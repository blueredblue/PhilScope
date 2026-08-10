# PhilScope

**A Discourse-Level Framework for Philosophical Authorship Analysis**

PhilScope is a research-oriented pipeline designed to analyze philosophical academic papers, extract core argumentative prompts, generate synthetic essay corpora, and ultimately classify authorship between humans and AI by analyzing discourse-level narrative and argumentative choices.

Inspired by the **StoryScope** (Russell et al., 2026) [1] research methodology, this project automates the transition from raw academic literature to structured, prompt-driven synthetic data to study model behavior in domain-specific, unstructured writing tasks.

## Project Aim
The ultimate goal of PhilScope is to move beyond superficial stylistic detection of AI writing. The aim is to determine if AI-generated philosophical essays can be distinguished from human-authored ones by analyzing **discourse-level narrative and argumentative choices** (e.g., argumentative structure, dialectical progression, handling of philosophical tensions, and moral ambiguity). By constructing a parallel corpus of human and AI-generated essays, PhilScope seeks to develop models capable of classifying authorship based on underlying philosophical construction rather than surface-level heuristics.

### Methodological Foundation

PhilScope builds upon the pipeline architecture and evaluation methodology introduced by Russell et al. in StoryScope [1]. These methods are adapted
for philosophical writing, with particular emphasis on argumentative structure, dialectical progression, philosophical tensions, and normative reasoning.

## Status
🚧 **Work in Progress:** This project is actively under development.

## Key Technical Contributions

*   **Automated Document Pipeline:** Developed a scalable pipeline to extract, clean, and structure raw text from academic PDF repositories.
*   **Prompt Synthesis Engine:** Engineered an LLM-based extraction layer that distills complex philosophical arguments into concise, high-quality essay prompts, ensuring argumentative focus while retaining philosophical nuance.
*   **Multi-Model Generation Architecture:** Implemented a robust orchestration framework utilizing `litellm` to parallelize synthetic essay generation across heterogeneous models (Gemini, Chat-GPT, Deepseek, Ling).
*   **Production-Grade Engineering:** Integrated sophisticated rate-limiting, error handling, and idempotency features to maintain pipeline stability during large-scale generation runs.
*   **Evaluation Corpus:** Curating a synthetic dataset to benchmark the inherent structural, dialectical, and normative biases across diverse Large Language Models (LLMs).

## Technical Stack

*   **Language:** Python
*   **LLM Orchestration:** `litellm`
*   **PDF Processing:** `pdfplumber`
*   **APIs:** Google Generative AI, Openrouter, Groq
*   **Data Management:** JSON-based persistence layer for failure logging and incremental processing.

## References

[1] Russell, Jenna, et al. *StoryScope: [Investigating idiosyncrasies in AI fiction.]*. [arXiv]. [[DOI or URL](https://arxiv.org/abs/2604.03136)] (2026)
---
*Developed for research purposes to evaluate LLM performance in specialized, complex, and open-ended philosophical analysis.*
