# PhilScope

**A Discourse-Level Framework for Philosophical Authorship Analysis**

PhilScope is a research-oriented pipeline designed to analyze philosophical academic papers, extract core argumentative prompts, generate synthetic essay corpora, and ultimately classify authorship between humans and AI by analyzing discourse-level narrative and argumentative choices.

Inspired by the [StoryScope](https://arxiv.org/abs/2412.02983) research methodology, this project automates the transition from raw academic literature to structured, prompt-driven synthetic data to study model behavior in domain-specific, unstructured writing tasks.

## Project Aim
The ultimate goal of PhilScope is to move beyond superficial stylistic detection of AI writing. The aim is to determine if AI-generated philosophical essays can be distinguished from human-authored ones by analyzing **discourse-level narrative and argumentative choices** (e.g., argumentative structure, dialectical progression, handling of philosophical tensions, and moral ambiguity). By constructing a parallel corpus of human and AI-generated essays, PhilScope seeks to develop models capable of classifying authorship based on underlying philosophical construction rather than surface-level heuristics.

## Status
🚧 **Work in Progress:** This project is actively under development.

## Key Technical Contributions

*   **Automated Document Pipeline:** Developed a scalable pipeline to extract, clean, and structure raw text from academic PDF repositories.
*   **Prompt Synthesis Engine:** Engineered an LLM-based extraction layer that distills complex philosophical arguments into concise, high-quality essay prompts, ensuring argumentative focus while retaining philosophical nuance.
*   **Multi-Model Generation Architecture:** Implemented a robust orchestration framework utilizing `litellm` to parallelize synthetic essay generation across heterogeneous models (Gemini, Llama, Qwen).
*   **Production-Grade Engineering:** Integrated sophisticated rate-limiting, error handling, and idempotency features to maintain pipeline stability during large-scale generation runs.
*   **Evaluation Corpus:** Curating a synthetic dataset to benchmark the inherent structural, dialectical, and normative biases across diverse Large Language Models (LLMs).

## Technical Stack

*   **Language:** Python
*   **LLM Orchestration:** `litellm`
*   **PDF Processing:** `pdfplumber`
*   **APIs:** Google Generative AI, Openrouter, Groq
*   **Data Management:** JSON-based persistence layer for failure logging and incremental processing.

---
*Developed for research purposes to evaluate LLM performance in specialized, complex, and open-ended philosophical analysis.*
