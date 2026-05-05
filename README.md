# AI-Powered CV Screening System

This project is a prototype recruitment decision-support system developed for **Assessment 3: Real World Applications of Artificial Intelligence**.

The system analyses CVs and a job description, extracts structured candidate and role information, ranks candidates using AI-based scoring methods, audits potential demographic bias, and supports plain-English recruiter queries using embeddings and retrieval-augmented generation.

The aim is not to replace human recruiters, but to demonstrate how different AI techniques can be combined into a transparent decision-support workflow.

---

## Project Overview

A recruiter provides:

1. A set of raw CV text files
2. A plain-text job description

The system then:

1. Parses each CV into structured candidate data
2. Analyses the job description into weighted requirements
3. Scores each candidate using two AI approaches
4. Tests whether demographic proxy changes affect candidate scores
5. Provides a web interface for recruiter search and shortlist explanation

Each module reads and writes JSON files rather than passing Python objects directly. This means each team member can develop and test their own component independently.

---

## AI Techniques Used

This project demonstrates several AI approaches covered in the course:

- Large Language Models and Generative AI
- Natural language processing
- Knowledge representation
- Probabilistic reasoning
- Naïve Bayes-style scoring
- Multilayer perceptron neural networks
- Bias detection and responsible AI
- Embeddings
- Semantic search
- Retrieval-augmented generation

---

## Project Modules

### 1. CV Parser

The CV Parser reads raw CV text files and uses a large language model to extract structured information such as skills, years of experience, education level, job titles, and a completeness score.

**AI connection:** Generative AI and natural language processing.

### 2. Job Description Analyser

The Job Description Analyser reads a plain-text job advertisement and identifies role requirements, including which skills are essential, which are desirable, and how strongly each requirement should be weighted.

**AI connection:** LLM-based information extraction and knowledge representation.

### 3. Candidate Ranker

The Candidate Ranker compares each parsed CV against the structured job profile and scores candidates using two approaches:

- Naïve Bayes-style probabilistic scoring
- A shallow multilayer perceptron model

The system also compares the two model outputs and flags uncertainty when the models disagree.

**AI connection:** Probabilistic reasoning and multilayer neural networks.

### 4. Bias Detection Module

The Bias Detection Module creates counterfactual versions of candidate profiles by changing names, universities, and employment gaps while keeping the underlying CV information constant. It then checks whether the candidate scores shift across variant groups.

**AI connection:** Responsible AI, AI ethics, counterfactual fairness testing, and statistical analysis.

### 5. Recruiter Query Interface

The Recruiter Query Interface allows a recruiter to search the candidate pool using plain English questions, such as:

- “Who has the strongest Python background?”
- “Show me candidates with stakeholder management experience.”
- “Who is strongest for SQL and dashboard reporting?”

The system embeds candidate records, retrieves relevant candidates using semantic similarity, and uses GPT to generate a shortlist explanation.

**AI connection:** Embeddings, semantic search, retrieval-augmented generation, and Generative AI.

---

## System Design

Each module is independent. Modules communicate by reading and writing JSON files in the `shared_outputs` folder.

This structure allows each team member to work on a separate task without needing to import or modify another person’s code.

```text
ai-cv-screening-system/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── cvs_raw/
│   └── job_description/
│
├── shared_outputs/
│   ├── parsed_candidates/
│   ├── job_profile/
│   ├── scored_candidates/
│   ├── bias_reports/
│   └── recruiter_queries/
│
├── task1_cv_parser/
│   └── cv_parser.py
│
├── task2_job_analyser/
│   └── job_analyser.py
│
├── task3_candidate_ranker/
│   └── candidate_ranker.py
│
├── task4_bias_detection/
│   └── bias_detector.py
│
└── task5_recruiter_query/
    ├── recruiter_query.py
    ├── recruiter_query_web.py
    ├── templates/
    │   └── index.html
    └── static/
        └── style.css
```

---

## Setup Instructions

### 1. Clone or download the repository

Download or clone this repository to your computer.

### 2. Create an OpenAI API key file

Create a file called:

```text
.env.local
```

Place it in the root folder of the project, next to `README.md` and `requirements.txt`.

Inside `.env.local`, add:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not upload `.env.local` to GitHub. This file is ignored by `.gitignore`.

The repository includes `.env.example` as a safe template.

### 3. Install dependencies

Open a terminal in the project root folder and run:

```bash
pip install -r requirements.txt
```

---

## How to Run the Full System

Run each module in order from the root project folder:

```bash
python task1_cv_parser/cv_parser.py
python task2_job_analyser/job_analyser.py
python task3_candidate_ranker/candidate_ranker.py
python task4_bias_detection/bias_detector.py
python task5_recruiter_query/recruiter_query_web.py
```

Then open the web interface in a browser:

```text
http://127.0.0.1:5000
```

---

## Running the Web Interface

When the web interface opens, enter a recruiter query such as:

```text
who has the strongest Python background?
```

or:

```text
show me candidates with stakeholder management experience
```

If the CVs, job description, candidate scores, or explanations have changed, tick:

```text
Rebuild embedding index
```

before searching.

This rebuilds the semantic search index from the latest candidate JSON files.

---

## Important Note About Multiple Job Descriptions

The current prototype supports **one active job description per run**.

To test a different job description:

1. Place the desired job description in `data/job_description/`
2. Rerun the job description analyser
3. Rerun the candidate ranker
4. Rerun the bias detector
5. Rebuild the embedding index in the web interface

Recommended command sequence:

```bash
python task2_job_analyser/job_analyser.py
python task3_candidate_ranker/candidate_ranker.py
python task4_bias_detection/bias_detector.py
python task5_recruiter_query/recruiter_query_web.py
```

---

## Output Files

The system writes outputs to the `shared_outputs` folder.

### CV Parser output

```text
shared_outputs/parsed_candidates/
```

Example:

```json
{
  "candidate_id": "cv_001",
  "name": "Aisha Khan",
  "skills": ["python", "sql", "excel"],
  "experience_years": 3,
  "education_level": "bachelor",
  "job_titles": ["data analyst", "junior business analyst"],
  "completeness_score": 0.9
}
```

### Job Description Analyser output

```text
shared_outputs/job_profile/
```

Example:

```json
{
  "job_id": "jd_001",
  "title": "Senior Data Scientist",
  "requirements": [
    {
      "skill": "python",
      "weight": 1.0,
      "essential": true
    }
  ],
  "experience_years_required": 4,
  "education_required": "bachelor"
}
```

### Candidate Ranker output

```text
shared_outputs/scored_candidates/
```

Example:

```json
{
  "candidate_id": "cv_001",
  "score": 0.82,
  "confidence": "high",
  "breakdown": {
    "skills": 0.91,
    "experience": 0.85,
    "education": 0.70
  }
}
```

### Bias Detection output

```text
shared_outputs/bias_reports/
```

Example files:

```text
bias_report.json
bias_score_chart.png
bias_scored_variants.json
```

### Recruiter Query output

```text
shared_outputs/recruiter_queries/
```

Example files:

```text
candidate_embedding_index.json
recruiter_query_result.json
```

---

## Prototype Screenshots

The screenshots below show the final web interface for the AI-powered CV screening prototype.

### AI-Powered CV Screening Search

![AI-Powered CV Screening Search](screenshots/AI-Powered%20CV%20Screening%20Search.png)

### Recruiter Query

![Recruiter query](screenshots/Recruiter%20query.png)

### Explanation Panel

![Explanation panel](screenshots/Explaination%20panel.png)

### Candidate Cards

![Candidate cards](screenshots/Candidate%20cards.png)

### Job Profile

![Job Profile](screenshots/Job%20Profile.png)

### Bias Audit

![Bias Audit](screenshots/Bias%20Audit.png)

---

## Responsible AI Notes

This prototype is intended as a decision-support system, not an automated hiring tool.

The system includes a bias audit that checks whether changing proxy attributes such as names, universities, and employment gaps changes candidate scores. In the current design, these fields are intentionally excluded from the main scoring function.

This reduces the risk of direct proxy bias, but it does not prove the system is completely bias-free. Bias can still enter through CV content, training data, extraction errors, job description wording, or structural inequalities in education and work experience.

Human review remains essential.

---

## Limitations

This is a classroom prototype and has several limitations:

- The test CVs are synthetic.
- The MLP model uses synthetic training examples rather than real recruitment labels.
- The ranking scores should not be treated as validated hiring decisions.
- The bias audit is small-scale and should not be interpreted as proof of fairness.
- The local embedding index is suitable for demonstration but not production use.
- A real recruitment system would require privacy controls, access logging, validation, human review, and legal compliance.

---

## Contributors

- Team member 1: CV Parser
- Team member 2: Job Description Analyser
- Team member 3: Candidate Ranker
- Team member 4: Bias Detection Module
- Team member 5: Recruiter Query Interface
