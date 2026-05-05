import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# ------------------------------------------------------------
# Task 1: CV Parser
# ------------------------------------------------------------
# Reads raw CV text files from data/cvs_raw/
# Uses OpenAI to extract structured candidate information
# Writes one JSON file per CV to shared_outputs/parsed_candidates/
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "cvs_raw"
OUTPUT_DIR = BASE_DIR / "shared_outputs" / "parsed_candidates"
ENV_PATH = BASE_DIR / ".env.local"

MODEL_NAME = "gpt-4o-mini"


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


def create_candidate_id(file_path: Path, index: int) -> str:
    """
    Creates a stable candidate ID from the file number.
    Example: cv_001, cv_002
    """
    return f"cv_{index:03d}"


def read_cv_text(file_path: Path) -> str:
    """
    Reads a raw CV text file.
    """
    return file_path.read_text(encoding="utf-8")


def clean_skill_list(skills: List[str]) -> List[str]:
    """
    Normalises skill names so later modules receive consistent values.
    """
    cleaned = []

    for skill in skills:
        skill_clean = skill.strip().lower()

        if skill_clean and skill_clean not in cleaned:
            cleaned.append(skill_clean)

    return cleaned


def clean_job_titles(job_titles: List[str]) -> List[str]:
    """
    Normalises job titles so later modules receive consistent values.
    """
    cleaned = []

    for title in job_titles:
        title_clean = title.strip().lower()

        if title_clean and title_clean not in cleaned:
            cleaned.append(title_clean)

    return cleaned


def validate_candidate_json(candidate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs simple validation and cleanup after the model response.
    This makes the JSON safer for Task 3, Task 4, and Task 5.
    """

    required_fields = [
        "candidate_id",
        "name",
        "skills",
        "experience_years",
        "education_level",
        "job_titles",
        "completeness_score",
    ]

    for field in required_fields:
        if field not in candidate_data:
            raise ValueError(f"Missing required field: {field}")

    candidate_data["skills"] = clean_skill_list(candidate_data["skills"])
    candidate_data["job_titles"] = clean_job_titles(candidate_data["job_titles"])

    candidate_data["experience_years"] = float(candidate_data["experience_years"])

    candidate_data["education_level"] = str(candidate_data["education_level"]).strip().lower()

    candidate_data["completeness_score"] = float(candidate_data["completeness_score"])
    candidate_data["completeness_score"] = max(0.0, min(1.0, candidate_data["completeness_score"]))

    return candidate_data


def extract_candidate_data(client: OpenAI, cv_text: str, candidate_id: str) -> Dict[str, Any]:
    """
    Sends the CV text to OpenAI and extracts structured candidate data.
    """

    schema = {
        "name": "candidate_cv_profile",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "The candidate ID provided by the system, such as cv_001."
                },
                "name": {
                    "type": "string",
                    "description": "The candidate's full name. Use 'unknown' if not available."
                },
                "skills": {
                    "type": "array",
                    "description": "A list of technical, analytical, communication, and professional skills found in the CV.",
                    "items": {
                        "type": "string"
                    }
                },
                "experience_years": {
                    "type": "number",
                    "description": "Estimated total years of relevant professional experience."
                },
                "education_level": {
                    "type": "string",
                    "description": "Highest education level. Use one of: high school, certificate, diploma, bachelor, master, phd, unknown.",
                    "enum": [
                        "high school",
                        "certificate",
                        "diploma",
                        "bachelor",
                        "master",
                        "phd",
                        "unknown"
                    ]
                },
                "job_titles": {
                    "type": "array",
                    "description": "Previous or current job titles listed in the CV.",
                    "items": {
                        "type": "string"
                    }
                },
                "completeness_score": {
                    "type": "number",
                    "description": "A score from 0 to 1 estimating how complete the CV is for recruitment screening."
                }
            },
            "required": [
                "candidate_id",
                "name",
                "skills",
                "experience_years",
                "education_level",
                "job_titles",
                "completeness_score"
            ]
        },
        "strict": True
    }

    prompt = f"""
You are extracting structured information from a candidate CV.

Use only the information in the CV.
Do not invent qualifications, skills, job titles, or experience.
If information is missing, use reasonable conservative estimates only where necessary.

The candidate_id must be exactly: {candidate_id}

CV text:
{cv_text}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "You are a careful CV information extraction assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema["strict"]
            }
        }
    )

    raw_output = response.output_text
    candidate_data = json.loads(raw_output)

    return validate_candidate_json(candidate_data)


def save_candidate_json(candidate_data: Dict[str, Any], output_path: Path) -> None:
    """
    Saves structured candidate data to a JSON file.
    """
    output_path.write_text(
        json.dumps(candidate_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def parse_all_cvs() -> None:
    """
    Main workflow for Task 1.
    """

    load_api_key()

    client = OpenAI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_files = sorted(INPUT_DIR.glob("*.txt"))

    if not cv_files:
        print(f"No CV text files found in: {INPUT_DIR}")
        return

    print(f"Found {len(cv_files)} CV file(s).")

    for index, cv_file in enumerate(cv_files, start=1):
        candidate_id = create_candidate_id(cv_file, index)
        output_file = OUTPUT_DIR / f"{candidate_id}.json"

        print(f"Processing {cv_file.name} as {candidate_id}...")

        try:
            cv_text = read_cv_text(cv_file)
            candidate_data = extract_candidate_data(client, cv_text, candidate_id)
            save_candidate_json(candidate_data, output_file)

            print(f"Saved: {output_file}")

        except Exception as error:
            print(f"Error processing {cv_file.name}: {error}")

    print("Task 1 CV parsing complete.")


if __name__ == "__main__":
    parse_all_cvs()