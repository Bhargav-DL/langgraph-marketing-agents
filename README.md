# Multi-Agent Marketing Content Generator

This project implements a multi-agent content generation system for marketing using **Qwen 2.5 7B**, **LangGraph**, and **web search**. It simulates a marketing team: Search → Researcher → Strategist → Copywriter → SEO Editor, producing high-quality, researched content.

## Features
- Loads Qwen 2.5 7B with 4-bit quantization (runs on Colab/GPU).
- Real-time web search via DuckDuckGo.
- LangGraph orchestrates the agent workflow.
- Gradio interface for easy interaction.

## Requirements
- Python 3.10+
- CUDA-capable GPU (e.g., T4, A100) recommended.
- See `requirements.txt` for Python packages.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marketing-agent-crew.git
   cd marketing-agent-crew