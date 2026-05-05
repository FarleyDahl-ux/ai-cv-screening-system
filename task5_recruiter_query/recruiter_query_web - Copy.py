import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file
from openai import OpenAI


# ------------------------------------------------------------
# Task 5: Recruiter Query Web Interface
# ------------------------------------------------------------
# Reads:
# - Parsed candidate JSON files from shared_outputs/parsed_candidates/
# - Scored candidate JSON files from shared_outputs/scored_candidates/
# - Bias report and chart from shared_outputs/bias_reports/
#
# Builds:
# - A local embedding index saved to shared_outputs/recruiter_queries/
#
# Allows:
# - Plain-English recruiter queries through a graphical web interface
# - Semantic search over candidate records
# - GPT-generated shortlist explanation
# - Display of Task 4 bias audit summary and chart
#
# Writes:
# - recruiter_query_result.json
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = BASE_DIR / "shared_outputs" / "parsed_candidates"
SCORED_DIR = BASE_DIR / "shared_outputs" / "scored_candidates"
OUTPUT_DIR = BASE_DIR / "shared_outputs" / "recruiter_queries"
ENV_PATH = BASE_DIR / ".env.local"

BIAS_REPORT_DIR = BASE_DIR / "shared_outputs" / "bias_reports"
BIAS_REPORT_FILE = BIAS_REPORT_DIR / "bias_report.json"
BIAS_CHART_FILE = BIAS_REPORT_DIR / "bias_score_chart.png"

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

INDEX_FILE = OUTPUT_DIR / "candidate_embedding_index.json"
QUERY_RESULT_FILE = OUTPUT_DIR / "recruiter_query_result.json"

app = Flask(__name__)


def load_api_key() -> None:
    """
    Loads environment variables from .env.local.
    The .env.local file should contain:
    OPENAI_API_KEY=your_api_key_here
    """
    load_dotenv(dotenv_path=ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY was not found. Check that .env.local exists "
            "and contains OPENAI_API_KEY=your_key_here"
        )


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Loads a JSON file.
    """
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Saves a JSON file with readable formatting.
    """
    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_bias_report() -> Dict[str, Any]:
    """
    Loads the Task 4 bias report if it exists.
    Returns a fallback object if Task 4 has not been run yet.
    """
    if not BIAS_REPORT_FILE.exists():
        return {
            "available": False,
            "overall_finding": (
                "Bias audit has not been run yet. Run "
                "task4_bias_detection/bias_detector.py to generate the bias report."
            ),
            "candidate_count": 0,
            "variant_count": 0,
            "bias_flags": [],
            "important_design_note": ""
        }

    report = load_json(BIAS_REPORT_FILE)
    report["available"] = True

    return report


def load_candidates() -> Dict[str, Dict[str, Any]]:
    """
    Loads parsed candidate JSONs from Task 1.
    Returns a dictionary keyed by candidate_id.
    """
    candidate_files = sorted(CANDIDATE_DIR.glob("*.json"))

    if not candidate_files:
        raise FileNotFoundError(
            f"No parsed candidate JSON files found in: {CANDIDATE_DIR}"
        )

    candidates = {}

    for file_path in candidate_files:
        candidate = load_json(file_path)
        candidates[candidate["candidate_id"]] = candidate

    return candidates


def load_scored_results() -> Dict[str, Dict[str, Any]]:
    """
    Loads scored candidate JSONs from Task 3.
    Excludes ranking_summary.json because it does not end with _score.json.
    """
    score_files = sorted(SCORED_DIR.glob("*_score.json"))

    if not score_files:
        raise FileNotFoundError(
            f"No scored candidate JSON files found in: {SCORED_DIR}"
        )

    scored_results = {}

    for file_path in score_files:
        result = load_json(file_path)
        scored_results[result["candidate_id"]] = result

    return scored_results


def combine_candidate_records(
    candidates: Dict[str, Dict[str, Any]],
    scored_results: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combines Task 1 and Task 3 records into one searchable candidate record.
    """
    combined_records = []

    for candidate_id, candidate in candidates.items():
        score = scored_results.get(candidate_id)

        if score is None:
            print(f"Warning: no scored result found for {candidate_id}. Skipping.")
            continue

        combined_records.append(
            {
                "candidate_id": candidate_id,
                "name": candidate.get("name", "unknown"),
                "skills": candidate.get("skills", []),
                "experience_years": candidate.get("experience_years", 0),
                "education_level": candidate.get("education_level", "unknown"),
                "job_titles": candidate.get("job_titles", []),
                "completeness_score": candidate.get("completeness_score", 0),
                "score": score.get("score", 0),
                "confidence": score.get("confidence", "unknown"),
                "breakdown": score.get("breakdown", {}),
                "method_comparison": score.get("method_comparison", {}),
                "explanation": score.get("explanation", "")
            }
        )

    if not combined_records:
        raise ValueError("No combined candidate records could be created.")

    return combined_records


def candidate_record_to_search_text(record: Dict[str, Any]) -> str:
    """
    Converts a candidate JSON record into searchable text for embedding.

    This text representation is what semantic search uses.
    """
    skills = ", ".join(record.get("skills", []))
    job_titles = ", ".join(record.get("job_titles", []))
    breakdown = record.get("breakdown", {})

    return f"""
Candidate ID: {record.get("candidate_id")}
Name: {record.get("name")}
Job titles: {job_titles}
Skills: {skills}
Experience years: {record.get("experience_years")}
Education level: {record.get("education_level")}
Overall ranking score: {record.get("score")}
Confidence: {record.get("confidence")}
Skill score: {breakdown.get("skills")}
Experience score: {breakdown.get("experience")}
Education score: {breakdown.get("education")}
Scoring explanation: {record.get("explanation")}
"""


def get_embedding(client: OpenAI, text: str) -> List[float]:
    """
    Creates one embedding vector for a text string.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Calculates cosine similarity between two vectors.
    """
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def build_embedding_index(
    client: OpenAI,
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Builds an embedding index from candidate records.
    """
    index = []

    for record in records:
        print(f"Embedding candidate: {record['candidate_id']}")

        search_text = candidate_record_to_search_text(record)
        embedding = get_embedding(client, search_text)

        index.append(
            {
                "candidate_id": record["candidate_id"],
                "name": record["name"],
                "search_text": search_text,
                "embedding": embedding,
                "record": record
            }
        )

    return index


def load_or_build_index(
    client: OpenAI,
    rebuild: bool = False
) -> List[Dict[str, Any]]:
    """
    Loads the existing embedding index if available.
    Otherwise builds a new one.

    Set rebuild=True if candidate records have changed.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()
    scored_results = load_scored_results()
    records = combine_candidate_records(candidates, scored_results)

    if INDEX_FILE.exists() and not rebuild:
        print(f"Loading existing embedding index: {INDEX_FILE}")
        saved_index = load_json(INDEX_FILE)
        return saved_index["index"]

    print("Building new embedding index...")
    index = build_embedding_index(client, records)

    save_json(
        {
            "embedding_model": EMBEDDING_MODEL,
            "index": index
        },
        INDEX_FILE
    )

    print(f"Saved embedding index: {INDEX_FILE}")

    return index


def search_candidates(
    client: OpenAI,
    query: str,
    index: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches candidates using semantic similarity.
    """
    query_embedding = get_embedding(client, query)

    results = []

    for item in index:
        similarity = cosine_similarity(query_embedding, item["embedding"])

        results.append(
            {
                "candidate_id": item["candidate_id"],
                "name": item["name"],
                "similarity": round(similarity, 4),
                "record": item["record"]
            }
        )

    ranked = sorted(
        results,
        key=lambda item: item["similarity"],
        reverse=True
    )

    return ranked[:top_k]


def synthesise_recommendation(
    client: OpenAI,
    query: str,
    retrieved_candidates: List[Dict[str, Any]]
) -> str:
    """
    Uses GPT to write a plain-English recruiter shortlist recommendation.
    """
    candidate_summaries = []

    for rank, item in enumerate(retrieved_candidates, start=1):
        record = item["record"]

        candidate_summaries.append(
            {
                "rank": rank,
                "candidate_id": record["candidate_id"],
                "name": record["name"],
                "semantic_similarity": item["similarity"],
                "overall_score": record["score"],
                "confidence": record["confidence"],
                "skills": record["skills"],
                "experience_years": record["experience_years"],
                "education_level": record["education_level"],
                "job_titles": record["job_titles"],
                "breakdown": record["breakdown"],
                "explanation": record["explanation"]
            }
        )

    prompt = f"""
You are assisting a recruiter who is searching a candidate pool.

Recruiter query:
{query}

Retrieved candidates:
{json.dumps(candidate_summaries, indent=2)}

Write a concise recommendation that:
1. Directly answers the recruiter's query.
2. Explains why the top candidates were retrieved.
3. Uses both semantic similarity and candidate score.
4. Mentions any important limitations or uncertainty.
5. Avoids making claims about protected characteristics.
6. Does not overstate certainty.
7. Makes clear this is decision support, not an automatic hiring decision.

Use plain English.
"""

    response = client.responses.create(
        model=GPT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful recruitment decision-support assistant. "
                    "You explain shortlist results transparently and avoid unfair, unsupported, "
                    "or discriminatory claims."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main web page route.
    Displays the recruiter search interface, Task 5 results, and Task 4 bias audit.
    """
    load_api_key()
    client = OpenAI()

    query = ""
    recommendation = None
    results = []
    error = None
    rebuild_checked = False
    selected_top_k = "3"

    bias_report = load_bias_report()
    bias_chart_available = BIAS_CHART_FILE.exists()

    if request.method == "POST":
        try:
            query = request.form.get("query", "").strip()
            selected_top_k = request.form.get("top_k", "3")
            rebuild_checked = request.form.get("rebuild_index") == "on"

            try:
                top_k = int(selected_top_k)
            except ValueError:
                top_k = 3
                selected_top_k = "3"

            if not query:
                error = "Please enter a recruiter query."
            else:
                embedding_index = load_or_build_index(
                    client=client,
                    rebuild=rebuild_checked
                )

                results = search_candidates(
                    client=client,
                    query=query,
                    index=embedding_index,
                    top_k=top_k
                )

                recommendation = synthesise_recommendation(
                    client=client,
                    query=query,
                    retrieved_candidates=results
                )

                output = {
                    "query": query,
                    "top_k": top_k,
                    "retrieved_candidates": [
                        {
                            "rank": rank + 1,
                            "candidate_id": item["candidate_id"],
                            "name": item["name"],
                            "similarity": item["similarity"],
                            "score": item["record"]["score"],
                            "confidence": item["record"]["confidence"],
                            "skills": item["record"]["skills"],
                            "experience_years": item["record"]["experience_years"],
                            "education_level": item["record"]["education_level"],
                            "breakdown": item["record"]["breakdown"],
                            "explanation": item["record"]["explanation"]
                        }
                        for rank, item in enumerate(results)
                    ],
                    "recommendation": recommendation
                }

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                save_json(output, QUERY_RESULT_FILE)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        query=query,
        recommendation=recommendation,
        results=results,
        error=error,
        rebuild_checked=rebuild_checked,
        selected_top_k=selected_top_k,
        bias_report=bias_report,
        bias_chart_available=bias_chart_available
    )


@app.route("/bias-chart")
def bias_chart():
    """
    Serves the Task 4 bias chart image if it exists.
    """
    if not BIAS_CHART_FILE.exists():
        return "Bias chart not found. Run Task 4 first.", 404

    return send_file(BIAS_CHART_FILE, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)