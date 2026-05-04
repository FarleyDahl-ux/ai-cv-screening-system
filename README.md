# ai-cv-screening-system
36121 Artificial Intelligence Principles and Applications - Autumn 2026 - Assignment 3 - AT3_Groups 28 

# AI-Powered CV Screening System

This project is a prototype recruitment decision-support system developed for Assessment 3: Real World Applications of Artificial Intelligence.

The system analyses CVs and a job description, extracts structured candidate and role information, ranks candidates using AI-based scoring methods, audits potential demographic bias, and supports plain-English recruiter queries using embeddings and retrieval-augmented generation.

## Project Modules

1. CV Parser  
   Extracts structured information from raw CV text using an LLM.

2. Job Description Analyser  
   Extracts role requirements and assigns importance weights.

3. Candidate Ranker  
   Compares candidates against the job profile using Naïve Bayes and a shallow MLP.

4. Bias Detection Module  
   Tests whether candidate scores change when demographic proxies are altered.

5. Recruiter Query Interface  
   Allows plain-English semantic search over scored candidate profiles.

## System Design

Each module is independent. Modules communicate by reading and writing JSON files in the shared_outputs folder. This allows each team member to develop and test their own component separately.

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
