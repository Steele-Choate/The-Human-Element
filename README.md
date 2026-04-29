# Evolving Distinguishing Minds

## Project Overview

Evolving Distinguishing Minds is a Python-based multiplayer text adventure and experimental human-AI identity judgment system. The project places the player in a simulated research facility where different drone characters may be controlled by humans, baseline AI agents, or multi-agent AI systems.

The goal of the project is to evaluate whether a structured multi-agent reasoning pipeline can improve human-AI identity classification compared to a single-prompt baseline model. The system compares baseline and agentic AI behavior using metrics such as classification accuracy, deception success rate, estimated token usage, and response latency.

## Features

- Text-based exploration game with a Tkinter graphical interface
- Room-based map navigation
- Inventory system with item tooltips
- Puzzle-solving mechanics
- Human, baseline AI, and multi-agent AI drone roles
- Human-vs-AI identity judgment system
- Baseline single-prompt AI response generation
- Multi-agent pipeline with persona, context, critic, and decision stages
- Optional LangGraph support for the agentic workflow
- Ollama backend support using `llama3:8b`
- Local GPT-2 fallback through Hugging Face Transformers
- Logging of experiment results to JSON and CSV files
- Plot generation for performance, latency, token usage, and deception metrics
- Optional multiplayer socket support

## Project Structure

project-folder/
│
├── main.py
├── requirements.txt
├── README.md
│
├── rooms/
│   └── room image files
│
├── items/
│   └── item image files
│
├── strategy_memory/
│   └── saved agent strategy memory files
│
├── human_response_bank.json
├── human_prompt_bank.json
├── agentic_eval_log.json
├── identity_experiment_log.json
└── generated plot/image output files

Note: Some JSON log files and generated plot files are created automatically when the program runs.
