# ARIS: Automated Research Ideation System

An autonomous, multi-agent pipeline that uses AI to analyze academic literature, mathematically identify research gaps, and generate highly novel research proposals.

## Table of Contents
- [Why ARIS is Needed](#why-aris-is-needed)
- [What is ARIS?](#what-is-aris)
- [How it Works](#how-it-works)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start Guide](#quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Installation (Ubuntu / Linux)](#installation-ubuntu--linux)
  - [Installation (Windows)](#installation-windows)
  - [Running the Application](#running-the-application)
- [License](#license)


## Why ARIS is Needed
Academic research often suffers from a severe bottleneck during the ideation phase. Researchers spend countless hours manually searching through literature, reading abstracts, and attempting to map out the boundary of human knowledge just to find a viable gap. Even after finding a gap, generating novel, structurally sound research ideas that haven't already been explored is a daunting task. ARIS was built to solve this problem by automating the literature review and ideation process, allowing researchers to focus on execution and experimentation rather than getting stuck in the literature phase.

## What is ARIS?
ARIS is an autonomous, multi-agent pipeline powered by Google's Gemini models and LangGraph. It acts as an AI research assistant that takes a simple high-level research topic and autonomously traverses the academic landscape. It reads recent literature, mathematically identifies unaddressed research gaps, and generates highly novel, peer-review-style research proposals.

## How it Works
The ARIS pipeline operates through a directed acyclic graph (DAG) of specialized nodes:

1. **Query Expansion:** Takes the user's broad topic and expands it into highly targeted search queries focusing on limitations, bottlenecks, and open problems.
2. **Data Retrieval:** Searches Semantic Scholar (with an automatic fallback to ArXiv) to retrieve the most relevant and highly cited recent academic papers.
3. **Knowledge Extraction:** Analyases the abstracts and text of the retrieved papers to extract specific limitations, future work suggestions, and failing methodologies.
4. **Limitation Embedding & Clustering:** Converts the extracted text into high-dimensional vectors and uses unsupervised clustering to group similar limitations together, mathematically defining "Research Gaps".
5. **Gap Validation:** An LLM agent reviews the clusters to ensure they represent genuine, distinct, and unsolved academic problems.
6. **Idea Generation:** Generates novel research ideas specifically targeted to solve the validated gaps.
7. **Novelty Evaluation & Critic Loop:** Checks the generated ideas against the original literature embeddings using cosine similarity to ensure they are actually novel. Ideas that are too structurally or semantically similar to existing work are sent back to the generation node for refinement.
8. **Report Generation:** Compiles the successfully validated ideas into a comprehensive, readable research report.

## Key Features
* **Multi-Agent Architecture:** Utilizes LangGraph to orchestrate complex reasoning steps across multiple autonomous nodes.
* **Aggressive API Resilience:** Built-in exponential backoff, instant model fallbacks (e.g., from Gemini 2.5 Pro to Gemini 2.5 Flash), and data source fallbacks (Semantic Scholar to ArXiv) to ensure the pipeline never hangs due to server overloads or rate limits.
* **Stateless TF-IDF Fallback:** If the external embedding API fails, ARIS automatically shifts to local HashingVectorizer TF-IDF embeddings to keep the clustering and novelty evaluation processes running smoothly without external dependencies.
* **Iterative Critic Loop:** Ideas are not just generated; they are rigorously evaluated for novelty and sent back for revision if they fail the threshold.
* **Interactive UI:** A clean, responsive Streamlit interface that allows users to configure thresholds, model parameters, and monitor the pipeline execution in real-time.

## File Structure 

```text
aris/
│
├── app.py                     # Main Streamlit application
├── config.py                  # Global configuration and hyperparameter settings
├── graph.py                   # LangGraph pipeline definition
├── state.py                   # State schemas and definitions
├── styles.py                  # CSS styling for the Streamlit interface
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment variables
│
├── nodes/                     # LangGraph Pipeline Nodes
│   ├── query_expansion.py     # Expands topics to search queries
│   ├── data_retrieval.py      # Semantic Scholar/ArXiv API interactions
│   ├── knowledge_extraction.py# Extracts limitations from papers
│   ├── limitation_embedding.py# Embeds limitations into vectors
│   ├── gap_clustering.py      # Groups limitations into research gaps
│   ├── gap_validation.py      # Validates gaps via LLM agent
│   ├── idea_generation.py     # Generates novel research ideas
│   ├── idea_embedding.py      # Embeds generated ideas
│   ├── novelty_evaluation.py  # Evaluates semantic and structural novelty
│   ├── critic_loop.py         # Routes failed ideas back for revision
│   └── evaluation_metrics.py  # Calculates final pipeline metrics
│
└── utils/                     # Helper Utilities
    ├── gemini_client.py       # Resilient Gemini API client (with fallbacks)
    ├── semantic_scholar.py    # Resilient Semantic Scholar API client (with ArXiv fallback)
    ├── embeddings.py          # Math for vector distance and semantic similarity
    └── clustering.py          # TF-IDF, K-Means, and Silhouette score logic
```

## Quick Start Guide

### Prerequisites
* Python 3.10 or higher
* A Google Gemini API Key

### Installation (Ubuntu / Linux)

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Sahasrangshu-Roy/ARIS.git
cd ARIS
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Installation (Windows)

1. Clone the repository and navigate to the project directory:
```powershell
git clone https://github.com/Sahasrangshu-Roy/ARIS.git
cd ARIS 
```

2. Create a virtual environment and activate it:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install the required dependencies:
```powershell
pip install -r requirements.txt
```

### Configuration

Set up your environment variables. Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional: Semantic Scholar API Key for higher rate limits
SEMANTIC_SCHOLAR_API_KEY=your_s2_api_key_here
```

### Running the Application

Start the Streamlit interface by running the following command:
```bash
streamlit run app.py
```

This will launch the ARIS dashboard in your default web browser. From there, you can input your research topic, adjust parameters like the novelty threshold or maximum iterations, and start the automated ideation pipeline.

## License
This project is licensed under the [MIT License](LICENSE).
