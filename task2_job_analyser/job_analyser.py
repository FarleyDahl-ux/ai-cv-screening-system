import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# ------------------------------------------------------------
# Task 2: Job Description Analyser
# ------------------------------------------------------------
# Reads a plain text job description from data/job_description/
# Uses OpenAI to extract structured job requirements
# Writes a single JSON file to shared_outputs/job_profile/
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "job_description"
OUTPUT_DIR = BASE_DIR / "shared_outputs" / "job_profile"
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


def find_job_description_file() -> Path:
    """
    Finds the first .txt job description file in data/job_description/.
    For the first version, we expect one job description only.
    """
    jd_files = sorted(INPUT_DIR.glob("*.txt"))

    if not jd_files:
        raise FileNotFoundError(
            f"No job description .txt file found in: {INPUT_DIR}"
        )

    if len(jd_files) > 1:
        print(
            "Multiple job description files found. "
            f"Using the first one: {jd_files[0].name}"
        )

    return jd_files[0]


def read_job_description(file_path: Path) -> str:
    """
    Reads the raw job description text.
    """
    return file_path.read_text(encoding="utf-8")


def create_job_id(file_path: Path) -> str:
    """
    Creates a simple job ID.

    If the file is called jd_001.txt, the job_id becomes jd_001.
    Otherwise, it defaults to jd_001.
    """
    stem = file_path.stem.strip().lower()

    if re.match(r"^jd_\d{3}$", stem):
        return stem

    return "jd_001"


def clean_skill_name(skill: str) -> str:
    """
    Normalises requirement skill names.
    """
    return skill.strip().lower()


def normalise_education_level(value: str) -> str:
    """
    Normalises education levels for downstream scoring.
    """
    value = value.strip().lower()

    mapping = {
        "high school": "high school",
        "secondary school": "high school",
        "certificate": "certificate",
        "cert": "certificate",
        "diploma": "diploma",
        "associate": "diploma",
        "bachelor": "bachelor",
        "bachelors": "bachelor",
        "undergraduate": "bachelor",
        "master": "master",
        "masters": "master",
        "postgraduate": "master",
        "phd": "phd",
        "doctorate": "phd",
        "doctoral": "phd",
        "unknown": "unknown",
    }

    return mapping.get(value, "unknown")


def validate_job_profile(job_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs simple validation and cleanup after the model response.
    This makes the JSON safer for Task 3.
    """

    required_fields = [
        "job_id",
        "title",
        "requirements",
        "experience_years_required",
        "education_required",
    ]

    for field in required_fields:
        if field not in job_profile:
            raise ValueError(f"Missing required field: {field}")

    job_profile["job_id"] = str(job_profile["job_id"]).strip()
    job_profile["title"] = str(job_profile["title"]).strip()

    cleaned_requirements = []

    for requirement in job_profile["requirements"]:
        if "skill" not in requirement:
            continue

        skill = clean_skill_name(str(requirement["skill"]))

        if not skill:
            continue

        weight = float(requirement.get("weight", 0.5))
        weight = max(0.0, min(1.0, weight))

        essential = bool(requirement.get("essential", False))

        cleaned_requirements.append(
            {
                "skill": skill,
                "weight": weight,
                "essential": essential,
            }
        )

    if not cleaned_requirements:
        raise ValueError("No valid requirements were extracted.")

    # Remove duplicate skills by keeping the highest weight.
    deduped = {}

    for requirement in cleaned_requirements:
        skill = requirement["skill"]

        if skill not in deduped:
            deduped[skill] = requirement
        else:
            if requirement["weight"] > deduped[skill]["weight"]:
                deduped[skill] = requirement

            # If any duplicate marks it as essential, keep essential as true.
            deduped[skill]["essential"] = (
                deduped[skill]["essential"] or requirement["essential"]
            )

    job_profile["requirements"] = list(deduped.values())

    job_profile["experience_years_required"] = float(
        job_profile["experience_years_required"]
    )

    if job_profile["experience_years_required"] < 0:
        job_profile["experience_years_required"] = 0.0

    job_profile["education_required"] = normalise_education_level(
        str(job_profile["education_required"])
    )

    return job_profile


def extract_job_profile(
    client: OpenAI,
    job_description_text: str,
    job_id: str
) -> Dict[str, Any]:
    """
    Sends the job description to OpenAI and extracts a weighted job profile.
    """

    schema = {
        "name": "job_description_profile",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job ID provided by the system, such as jd_001."
                },
                "title": {
                    "type": "string",
                    "description": "The job title from the job description."
                },
                "requirements": {
                    "type": "array",
                    "description": "A weighted list of job requirements extracted from the job description.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill": {
                                "type": "string",
                                "description": "A skill, capability, tool, qualification, domain knowledge area, or professional competency."
                            },
                            "weight": {
                                "type": "number",
                                "description": "Importance from 0 to 1. Essential requirements should be close to 1. Nice-to-haves should be lower."
                            },
                            "essential": {
                                "type": "boolean",
                                "description": "True if the requirement appears essential or non-negotiable. False if preferred, optional, or nice-to-have."
                            }
                        },
                        "required": [
                            "skill",
                            "weight",
                            "essential"
                        ]
                    }
                },
                "experience_years_required": {
                    "type": "number",
                    "description": "Minimum or implied years of experience required. Use 0 if not stated or not implied."
                },
                "education_required": {
                    "type": "string",
                    "description": "Minimum required education level. Use one of: high school, certificate, diploma, bachelor, master, phd, unknown.",
                    "enum": [
                        "high school",
                        "certificate",
                        "diploma",
                        "bachelor",
                        "master",
                        "phd",
                        "unknown"
                    ]
                }
            },
            "required": [
                "job_id",
                "title",
                "requirements",
                "experience_years_required",
                "education_required"
            ]
        },
        "strict": True
    }

    prompt = f"""
You are analysing a job description for an AI-powered recruitment screening system.

Your task is to extract a structured job profile.

Rules:
1. Use only the job description.
2. Do not invent requirements that are not stated or strongly implied.
3. Distinguish essential requirements from nice-to-have requirements.
4. Assign higher weights to requirements that appear essential, repeated, central to the role, or listed under mandatory criteria.
5. Assign lower weights to requirements that appear preferred, optional, desirable, or nice-to-have.
6. Use concise skill names, for example: python, sql, machine learning, stakeholder management.
7. The job_id must be exactly: {job_id}

Suggested weighting guide:
- 1.0 = absolutely essential or non-negotiable
- 0.8 to 0.9 = very important
- 0.6 to 0.7 = important but not always essential
- 0.3 to 0.5 = preferred or nice-to-have
- 0.1 to 0.2 = minor or weakly relevant

Job description:
{job_description_text}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "You are a careful job description analysis assistant."
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
    job_profile = json.loads(raw_output)

    return validate_job_profile(job_profile)


def save_job_profile(job_profile: Dict[str, Any], output_path: Path) -> None:
    """
    Saves the structured job profile to a JSON file.
    """
    output_path.write_text(
        json.dumps(job_profile, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def analyse_job_description() -> None:
    """
    Main workflow for Task 2.
    """

    load_api_key()

    client = OpenAI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jd_file = find_job_description_file()
    job_id = create_job_id(jd_file)

    print(f"Processing job description: {jd_file.name}")
    print(f"Using job_id: {job_id}")

    try:
        job_description_text = read_job_description(jd_file)
        job_profile = extract_job_profile(client, job_description_text, job_id)

        output_file = OUTPUT_DIR / f"{job_id}.json"
        save_job_profile(job_profile, output_file)

        print(f"Saved job profile: {output_file}")
        print("Task 2 job description analysis complete.")

    except Exception as error:
        print(f"Error processing job description: {error}")


if __name__ == "__main__":
    analyse_job_description()