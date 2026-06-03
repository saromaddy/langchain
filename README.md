# Understanding AI Agents and working on Multi agents

This repository contains experiments and learning materials for LangChain, AI Agents, and Multi-Agent systems.

## Topics Explored

### 1. Model Integrations (`langchainintro.ipynb`)
Exploring how to initialize and invoke different LLMs using LangChain's unified interface:
- **Ollama (Llama 3.2)**: Local model execution.
- **Google Gemini**: Integration via Google Generative AI (e.g., gemini-2.5-flash-lite).
- **GROQ (Qwen)**: Fast inference using the Groq API.

### 2. LangChain Tools (`3-tools.ipynb`)
Equipping AI agents with tools (like custom Python functions) so they can interact with the external world, make decisions, and fetch information dynamically.

### 3. LangChain Messages (`4-messages.ipynb`)
Understanding `Messages`, the fundamental unit of context for models in LangChain. They carry content and metadata to represent the state of a conversation:
- **Role**: Identifies the message type (e.g., System, User, AI, Tool).
- **Content**: The actual data (text, images, audio, etc.).
- **Metadata**: Optional fields like token usage and message IDs.