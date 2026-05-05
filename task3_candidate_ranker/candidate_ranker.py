import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# Task 3: Candidate Ranker
# ------------------------------------------------------------
# Reads:
# - Candidate JSON files from shared_outputs/parsed_candidates/
# - Job profile JSON from shared_outputs/job_profile/
#
# Scores each candidate using:
# - Naive Bayes style probabilistic scoring
# - Shallow MLP neural network scoring
#
# Writes:
# - One scored JSON file per candidate to shared_outputs/scored_candidates/
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = BASE_DIR / "shared_outputs" / "parsed_candidates"
JOB_PROFILE_DIR = BASE_DIR / "shared_outputs" / "job_profile"
OUTPUT_DIR = BASE_DIR / "shared_outputs" / "scored_candidates"

RANDOM_SEED = 42


EDUCATION_ORDER = {
    "unknown": 0,
    "high school": 1,
    "certificate": 2,
    "diploma": 3,
    "bachelor": 4,
    "master": 5,
    "phd": 6,
}


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


def load_candidates() -> List[Dict[str, Any]]:
    """
    Loads all parsed candidate JSON files from Task 1.
    """
    candidate_files = sorted(CANDIDATE_DIR.glob("*.json"))

    if not candidate_files:
        raise FileNotFoundError(
            f"No candidate JSON files found in: {CANDIDATE_DIR}"
        )

    candidates = []

    for file_path in candidate_files:
        candidates.append(load_json(file_path))

    return candidates


def load_job_profile() -> Dict[str, Any]:
    """
    Loads the first job profile JSON file from Task 2.
    For this prototype, we assume one job description only.
    """
    job_files = sorted(JOB_PROFILE_DIR.glob("*.json"))

    if not job_files:
        raise FileNotFoundError(
            f"No job profile JSON file found in: {JOB_PROFILE_DIR}"
        )

    if len(job_files) > 1:
        print(
            "Multiple job profile files found. "
            f"Using the first one: {job_files[0].name}"
        )

    return load_json(job_files[0])


def normalise_text(value: str) -> str:
    """
    Lowercases and strips text values.
    """
    return str(value).strip().lower()


def education_to_number(level: str) -> int:
    """
    Converts education level into an ordinal number.
    """
    return EDUCATION_ORDER.get(normalise_text(level), 0)


def calculate_skill_match(candidate_skills: List[str], requirement_skill: str) -> float:
    """
    Returns a skill match score between 0 and 1.

    This uses simple matching:
    - 1.0 for exact match
    - 0.7 for partial phrase match
    - 0.0 for no match

    Example:
    candidate skill: "basic machine learning"
    requirement: "machine learning"
    returns 0.7 or 1.0 depending on wording
    """
    requirement_skill = normalise_text(requirement_skill)
    candidate_skills = [normalise_text(skill) for skill in candidate_skills]

    if requirement_skill in candidate_skills:
        return 1.0

    for skill in candidate_skills:
        if requirement_skill in skill or skill in requirement_skill:
            return 0.7

    return 0.0


def calculate_weighted_skills_score(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> float:
    """
    Calculates weighted skill match score.
    """
    requirements = job_profile["requirements"]
    candidate_skills = candidate.get("skills", [])

    total_weight = sum(float(req["weight"]) for req in requirements)

    if total_weight == 0:
        return 0.0

    weighted_score = 0.0

    for req in requirements:
        match = calculate_skill_match(candidate_skills, req["skill"])
        weight = float(req["weight"])
        weighted_score += match * weight

    return weighted_score / total_weight


def calculate_essential_skills_score(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> float:
    """
    Calculates the proportion of essential skills matched.
    """
    essential_requirements = [
        req for req in job_profile["requirements"]
        if bool(req.get("essential", False))
    ]

    if not essential_requirements:
        return 1.0

    candidate_skills = candidate.get("skills", [])

    matches = []

    for req in essential_requirements:
        match = calculate_skill_match(candidate_skills, req["skill"])
        matches.append(match)

    return sum(matches) / len(matches)


def calculate_experience_score(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> float:
    """
    Scores experience against the required years.

    Full score if candidate meets or exceeds required experience.
    Partial score if below requirement.
    """
    candidate_years = float(candidate.get("experience_years", 0))
    required_years = float(job_profile.get("experience_years_required", 0))

    if required_years <= 0:
        return 1.0

    return min(candidate_years / required_years, 1.0)


def calculate_education_score(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> float:
    """
    Scores education level against required education.

    Full score if candidate meets or exceeds the required level.
    Partial score if below required level.
    """
    candidate_level = education_to_number(candidate.get("education_level", "unknown"))
    required_level = education_to_number(job_profile.get("education_required", "unknown"))

    if required_level <= 0:
        return 1.0

    if candidate_level >= required_level:
        return 1.0

    return candidate_level / required_level


def build_feature_vector(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> List[float]:
    """
    Converts a candidate and job profile into numeric features for scoring.
    """
    skills_score = calculate_weighted_skills_score(candidate, job_profile)
    essential_score = calculate_essential_skills_score(candidate, job_profile)
    experience_score = calculate_experience_score(candidate, job_profile)
    education_score = calculate_education_score(candidate, job_profile)

    completeness_score = float(candidate.get("completeness_score", 0))
    completeness_score = max(0.0, min(1.0, completeness_score))

    return [
        skills_score,
        essential_score,
        experience_score,
        education_score,
        completeness_score,
    ]


def naive_bayes_probability_score(features: List[float]) -> float:
    """
    Naive Bayes style probabilistic scorer.

    This is a simplified Bernoulli-style approach. Each feature is treated
    as evidence for candidate suitability.

    Features:
    0. weighted skill score
    1. essential skill score
    2. experience score
    3. education score
    4. completeness score

    We estimate:
    P(suitable | evidence)

    The probabilities below are assumptions for the prototype. They can be
    adjusted or learned from data if labelled outcomes are available.
    """

    prior_suitable = 0.50
    prior_not_suitable = 0.50

    evidence_weights = [
        0.35,  # weighted skills
        0.30,  # essential skills
        0.15,  # experience
        0.10,  # education
        0.10,  # CV completeness
    ]

    likelihood_suitable = prior_suitable
    likelihood_not_suitable = prior_not_suitable

    for feature_value, feature_weight in zip(features, evidence_weights):
        feature_value = max(0.01, min(0.99, feature_value))

        # If a candidate is suitable, strong feature matches are more likely.
        p_evidence_given_suitable = 0.20 + 0.75 * feature_value

        # If a candidate is not suitable, strong feature matches are less likely.
        p_evidence_given_not_suitable = 0.95 - 0.75 * feature_value

        likelihood_suitable *= p_evidence_given_suitable ** feature_weight
        likelihood_not_suitable *= p_evidence_given_not_suitable ** feature_weight

    denominator = likelihood_suitable + likelihood_not_suitable

    if denominator == 0:
        return 0.0

    probability = likelihood_suitable / denominator

    return round(float(probability), 4)


def create_synthetic_training_data(
    job_profile: Dict[str, Any],
    n_samples: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates synthetic training data for the MLP.

    Since the assignment prototype may not have labelled recruitment data,
    this creates simulated candidates with different levels of fit.

    Label rule:
    - Suitable if combined weighted score is high enough
    - Not suitable if combined weighted score is low

    This lets the MLP learn a non-linear approximation of the scoring logic.
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X = []
    y = []

    for _ in range(n_samples):
        skills_score = random.random()
        essential_score = random.random()
        experience_score = random.random()
        education_score = random.random()
        completeness_score = random.uniform(0.5, 1.0)

        # Essential skills and weighted skills matter most.
        latent_score = (
            0.35 * skills_score
            + 0.30 * essential_score
            + 0.15 * experience_score
            + 0.10 * education_score
            + 0.10 * completeness_score
        )

        # Add small noise to make the training data less rigid.
        latent_score += random.uniform(-0.05, 0.05)

        label = 1 if latent_score >= 0.65 else 0

        X.append([
            skills_score,
            essential_score,
            experience_score,
            education_score,
            completeness_score,
        ])
        y.append(label)

    return np.array(X), np.array(y)


def train_mlp_model(job_profile: Dict[str, Any]) -> Pipeline:
    """
    Trains a shallow MLP classifier on synthetic examples.
    """
    X_train, y_train = create_synthetic_training_data(job_profile)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(8,),
                    activation="relu",
                    solver="adam",
                    max_iter=1000,
                    random_state=RANDOM_SEED
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    return model


def mlp_probability_score(model: Pipeline, features: List[float]) -> float:
    """
    Uses the trained MLP to estimate candidate suitability probability.
    """
    X = np.array(features).reshape(1, -1)

    probabilities = model.predict_proba(X)[0]

    # Class 1 means suitable.
    score = probabilities[1]

    return round(float(score), 4)


def confidence_label(final_score: float, disagreement: float) -> str:
    """
    Creates a simple confidence label based on score strength and model agreement.
    """
    if disagreement >= 0.40:
        return "low"

    if final_score >= 0.75 and disagreement < 0.20:
        return "high"

    if final_score >= 0.55 and disagreement < 0.30:
        return "medium"

    return "low"

def create_score_breakdown(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> Dict[str, float]:
    """
    Creates the score breakdown required by the project contract.
    """
    return {
        "skills": round(
            calculate_weighted_skills_score(candidate, job_profile),
            4
        ),
        "essential_skills": round(
            calculate_essential_skills_score(candidate, job_profile),
            4
        ),
        "experience": round(
            calculate_experience_score(candidate, job_profile),
            4
        ),
        "education": round(
            calculate_education_score(candidate, job_profile),
            4
        ),
        "completeness": round(
            float(candidate.get("completeness_score", 0)),
            4
        ),
    }


def create_explanation(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any],
    breakdown: Dict[str, float],
    naive_bayes_score: float,
    mlp_score: float
) -> str:
    """
    Creates a short explanation for the scored result.
    """
    candidate_name = candidate.get("name", "unknown candidate")

    strengths = []
    concerns = []

    if breakdown["skills"] >= 0.75:
        strengths.append("strong overall skill match")
    elif breakdown["skills"] < 0.45:
        concerns.append("limited overall skill match")

    if breakdown["essential_skills"] >= 0.75:
        strengths.append("good coverage of essential requirements")
    elif breakdown["essential_skills"] < 0.50:
        concerns.append("missing several essential requirements")

    if breakdown["experience"] >= 1.0:
        strengths.append("meets or exceeds required experience")
    elif breakdown["experience"] < 0.70:
        concerns.append("below the required experience level")

    if breakdown["education"] >= 1.0:
        strengths.append("meets or exceeds required education")
    elif breakdown["education"] < 1.0:
        concerns.append("education level may be below requirement")

    disagreement = abs(naive_bayes_score - mlp_score)

    if disagreement >= 0.40:
        model_comment = (
            "The Naive Bayes and MLP scores differ substantially, "
            "so this candidate should be reviewed manually."
        )
    elif disagreement >= 0.25:
        model_comment = (
            "The Naive Bayes and MLP scores show some disagreement, "
            "which reflects uncertainty in the prototype model comparison."
        )
    else:
        model_comment = (
            "The Naive Bayes and MLP scores are broadly consistent."
        )

    explanation_parts = []

    if strengths:
        explanation_parts.append(f"{candidate_name} shows " + ", ".join(strengths) + ".")

    if concerns:
        explanation_parts.append("Potential concerns include " + ", ".join(concerns) + ".")

    explanation_parts.append(model_comment)

    return " ".join(explanation_parts)


def score_candidate(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any],
    mlp_model: Pipeline
) -> Dict[str, Any]:
    """
    Scores a single candidate using Naive Bayes and MLP.
    """
    features = build_feature_vector(candidate, job_profile)
    breakdown = create_score_breakdown(candidate, job_profile)

    naive_bayes_score = naive_bayes_probability_score(features)
    mlp_score = mlp_probability_score(mlp_model, features)

    final_score = round((naive_bayes_score + mlp_score) / 2, 4)
    disagreement = round(abs(naive_bayes_score - mlp_score), 4)

    confidence = confidence_label(final_score, disagreement)

    scored_result = {
        "candidate_id": candidate["candidate_id"],
        "name": candidate.get("name", "unknown"),
        "score": final_score,
        "confidence": confidence,
        "breakdown": {
            "skills": breakdown["skills"],
            "experience": breakdown["experience"],
            "education": breakdown["education"],
        },
        "method_comparison": {
            "naive_bayes_score": naive_bayes_score,
            "mlp_score": mlp_score,
            "score_disagreement": disagreement
        },
        "extended_breakdown": breakdown,
        "explanation": create_explanation(
            candidate,
            job_profile,
            breakdown,
            naive_bayes_score,
            mlp_score
        )
    }

    return scored_result


def rank_candidates() -> None:
    """
    Main workflow for Task 3.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()
    job_profile = load_job_profile()

    print(f"Loaded {len(candidates)} candidate(s).")
    print(f"Loaded job profile: {job_profile.get('title', 'unknown job')}")

    mlp_model = train_mlp_model(job_profile)

    all_results = []

    for candidate in candidates:
        print(f"Scoring candidate: {candidate.get('candidate_id')}")

        scored_result = score_candidate(candidate, job_profile, mlp_model)

        output_file = OUTPUT_DIR / f"{candidate['candidate_id']}_score.json"
        save_json(scored_result, output_file)

        all_results.append(scored_result)

        print(f"Saved: {output_file}")

    summary = create_ranking_summary(all_results, job_profile)
    summary_file = OUTPUT_DIR / "ranking_summary.json"
    save_json(summary, summary_file)

    print(f"Saved ranking summary: {summary_file}")
    print("Task 3 candidate ranking complete.")


def create_ranking_summary(
    results: List[Dict[str, Any]],
    job_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Creates a summary file comparing all candidates.
    """
    ranked_results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )

    average_score = sum(item["score"] for item in ranked_results) / len(ranked_results)
    average_disagreement = sum(
        item["method_comparison"]["score_disagreement"]
        for item in ranked_results
    ) / len(ranked_results)

    return {
        "job_id": job_profile.get("job_id", "unknown"),
        "job_title": job_profile.get("title", "unknown"),
        "candidate_count": len(ranked_results),
        "average_score": round(average_score, 4),
        "average_model_disagreement": round(average_disagreement, 4),
        "ranking": [
            {
                "rank": index + 1,
                "candidate_id": result["candidate_id"],
                "name": result["name"],
                "score": result["score"],
                "confidence": result["confidence"],
                "naive_bayes_score": result["method_comparison"]["naive_bayes_score"],
                "mlp_score": result["method_comparison"]["mlp_score"],
                "score_disagreement": result["method_comparison"]["score_disagreement"]
            }
            for index, result in enumerate(ranked_results)
        ]
    }


if __name__ == "__main__":
    rank_candidates()