import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ------------------------------------------------------------
# Task 4: Bias Detection Module
# ------------------------------------------------------------
# Reads:
# - Parsed candidate JSON files from shared_outputs/parsed_candidates/
# - Scored candidate JSON files from shared_outputs/scored_candidates/
# - Job profile JSON from shared_outputs/job_profile/
#
# Generates variants of the same candidate profile with different:
# - names
# - universities
# - employment gaps
#
# Re-scores each variant using a transparent scoring function.
# Then compares whether scores shift significantly across groups.
#
# Writes:
# - bias_report.json
# - bias_score_chart.png
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = BASE_DIR / "shared_outputs" / "parsed_candidates"
SCORED_DIR = BASE_DIR / "shared_outputs" / "scored_candidates"
JOB_PROFILE_DIR = BASE_DIR / "shared_outputs" / "job_profile"
OUTPUT_DIR = BASE_DIR / "shared_outputs" / "bias_reports"

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


NAME_VARIANTS = [
    {
        "variant_group": "anglo_name",
        "name": "Emily Thompson",
        "demographic_proxy": "Anglo-sounding name"
    },
    {
        "variant_group": "middle_eastern_name",
        "name": "Mohammed Al-Hassan",
        "demographic_proxy": "Middle Eastern-sounding name"
    },
    {
        "variant_group": "east_asian_name",
        "name": "Sarah Chen",
        "demographic_proxy": "East Asian-sounding name"
    },
    {
        "variant_group": "south_asian_name",
        "name": "Priya Patel",
        "demographic_proxy": "South Asian-sounding name"
    }
]


UNIVERSITY_VARIANTS = [
    {
        "variant_group": "high_prestige_university",
        "university": "University of Sydney",
        "university_tier": "high prestige"
    },
    {
        "variant_group": "standard_university",
        "university": "Western Sydney University",
        "university_tier": "standard"
    },
    {
        "variant_group": "regional_university",
        "university": "Charles Sturt University",
        "university_tier": "regional"
    },
    {
        "variant_group": "international_university",
        "university": "University of Mumbai",
        "university_tier": "international"
    }
]


EMPLOYMENT_GAP_VARIANTS = [
    {
        "variant_group": "no_gap",
        "employment_gap_months": 0
    },
    {
        "variant_group": "six_month_gap",
        "employment_gap_months": 6
    },
    {
        "variant_group": "twelve_month_gap",
        "employment_gap_months": 12
    },
    {
        "variant_group": "twenty_four_month_gap",
        "employment_gap_months": 24
    }
]


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Loads a JSON file.
    """
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Saves a JSON file.
    """
    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_candidates() -> List[Dict[str, Any]]:
    """
    Loads parsed candidate JSONs from Task 1.
    """
    candidate_files = sorted(CANDIDATE_DIR.glob("*.json"))

    if not candidate_files:
        raise FileNotFoundError(
            f"No candidate JSON files found in: {CANDIDATE_DIR}"
        )

    return [load_json(file_path) for file_path in candidate_files]


def load_scored_results() -> Dict[str, Dict[str, Any]]:
    """
    Loads scored result JSONs from Task 3.
    Excludes ranking_summary.json.
    """
    score_files = sorted(SCORED_DIR.glob("*_score.json"))

    if not score_files:
        raise FileNotFoundError(
            f"No scored candidate JSON files found in: {SCORED_DIR}"
        )

    results = {}

    for file_path in score_files:
        result = load_json(file_path)
        results[result["candidate_id"]] = result

    return results


def load_job_profile() -> Dict[str, Any]:
    """
    Loads the job profile JSON from Task 2.
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
    Converts education level into a number.
    """
    return EDUCATION_ORDER.get(normalise_text(level), 0)


def calculate_skill_match(candidate_skills: List[str], requirement_skill: str) -> float:
    """
    Returns a simple skill match score between 0 and 1.
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
    Scores experience against required years.

    Important:
    Employment gap is NOT used in this score by default.
    This is intentional because employment gaps can act as a proxy for
    caring responsibilities, illness, migration history, or other sensitive factors.
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

    Important:
    University name is NOT used in this score by default.
    This is intentional because university prestige can act as a socio-economic proxy.
    """
    candidate_level = education_to_number(candidate.get("education_level", "unknown"))
    required_level = education_to_number(job_profile.get("education_required", "unknown"))

    if required_level <= 0:
        return 1.0

    if candidate_level >= required_level:
        return 1.0

    return candidate_level / required_level


def deterministic_score_candidate(
    candidate: Dict[str, Any],
    job_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transparent scoring function used for the bias audit.

    This does not use:
    - candidate name
    - demographic proxy
    - university name
    - employment gap

    This lets us test whether demographic/profile variants change the score.
    """
    skills_score = calculate_weighted_skills_score(candidate, job_profile)
    essential_score = calculate_essential_skills_score(candidate, job_profile)
    experience_score = calculate_experience_score(candidate, job_profile)
    education_score = calculate_education_score(candidate, job_profile)
    completeness_score = float(candidate.get("completeness_score", 0))

    final_score = (
        0.35 * skills_score
        + 0.30 * essential_score
        + 0.15 * experience_score
        + 0.10 * education_score
        + 0.10 * completeness_score
    )

    return {
        "score": round(final_score, 4),
        "breakdown": {
            "skills": round(skills_score, 4),
            "essential_skills": round(essential_score, 4),
            "experience": round(experience_score, 4),
            "education": round(education_score, 4),
            "completeness": round(completeness_score, 4)
        }
    }


def generate_candidate_variants(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates controlled variants of the same candidate.

    The substantive CV fields stay the same. Only potentially bias-relevant
    fields are changed.
    """
    variants = []

    for name_variant in NAME_VARIANTS:
        for university_variant in UNIVERSITY_VARIANTS:
            for gap_variant in EMPLOYMENT_GAP_VARIANTS:
                variant = dict(candidate)

                variant["original_name"] = candidate.get("name", "unknown")
                variant["name"] = name_variant["name"]
                variant["demographic_proxy"] = name_variant["demographic_proxy"]
                variant["name_variant_group"] = name_variant["variant_group"]

                variant["university"] = university_variant["university"]
                variant["university_tier"] = university_variant["university_tier"]
                variant["university_variant_group"] = university_variant["variant_group"]

                variant["employment_gap_months"] = gap_variant["employment_gap_months"]
                variant["gap_variant_group"] = gap_variant["variant_group"]

                variant["variant_id"] = (
                    f"{candidate['candidate_id']}_"
                    f"{name_variant['variant_group']}_"
                    f"{university_variant['variant_group']}_"
                    f"{gap_variant['variant_group']}"
                )

                variants.append(variant)

    return variants


def run_variant_scoring(
    candidates: List[Dict[str, Any]],
    job_profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Scores all generated candidate variants.
    """
    scored_variants = []

    for candidate in candidates:
        variants = generate_candidate_variants(candidate)

        for variant in variants:
            scored = deterministic_score_candidate(variant, job_profile)

            scored_variants.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "variant_id": variant["variant_id"],
                    "name": variant["name"],
                    "original_name": variant["original_name"],
                    "demographic_proxy": variant["demographic_proxy"],
                    "name_variant_group": variant["name_variant_group"],
                    "university": variant["university"],
                    "university_tier": variant["university_tier"],
                    "university_variant_group": variant["university_variant_group"],
                    "employment_gap_months": variant["employment_gap_months"],
                    "gap_variant_group": variant["gap_variant_group"],
                    "score": scored["score"],
                    "breakdown": scored["breakdown"]
                }
            )

    return scored_variants


def group_scores(
    scored_variants: List[Dict[str, Any]],
    group_field: str
) -> Dict[str, List[float]]:
    """
    Groups variant scores by a field.
    """
    grouped = {}

    for item in scored_variants:
        group = item[group_field]
        grouped.setdefault(group, []).append(float(item["score"]))

    return grouped


def calculate_group_summary(grouped: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculates mean, standard deviation, count, min, and max by group.
    """
    summary = {}

    for group, scores in grouped.items():
        if len(scores) == 1:
            std_dev = 0.0
        else:
            std_dev = statistics.stdev(scores)

        summary[group] = {
            "count": len(scores),
            "mean_score": round(statistics.mean(scores), 4),
            "std_dev": round(std_dev, 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4)
        }

    return summary


def run_anova_test(grouped: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Runs a one-way ANOVA test across groups.

    ANOVA checks whether the mean scores differ significantly across groups.
    """
    groups = [scores for scores in grouped.values() if len(scores) > 1]

    if len(groups) < 2:
        return {
            "test": "one_way_anova",
            "p_value": None,
            "statistic": None,
            "significant": False,
            "note": "Not enough groups for ANOVA."
        }

    # If all scores are identical, scipy may warn or return nan.
    all_scores = [score for scores in groups for score in scores]

    if len(set(all_scores)) == 1:
        return {
            "test": "one_way_anova",
            "p_value": 1.0,
            "statistic": 0.0,
            "significant": False,
            "note": "All scores are identical across groups."
        }

    statistic, p_value = stats.f_oneway(*groups)

    if math.isnan(p_value):
        p_value = 1.0

    return {
        "test": "one_way_anova",
        "p_value": round(float(p_value), 6),
        "statistic": round(float(statistic), 6),
        "significant": bool(p_value < 0.05),
        "note": "Significant means at least one group mean differs at p < 0.05."
    }


def run_pairwise_tests(
    grouped: Dict[str, List[float]]
) -> List[Dict[str, Any]]:
    """
    Runs pairwise independent t-tests between each group.

    This is simple and useful for the assignment demo.
    """
    group_names = list(grouped.keys())
    results = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group_a = group_names[i]
            group_b = group_names[j]

            scores_a = grouped[group_a]
            scores_b = grouped[group_b]

            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)

            mean_difference = mean_a - mean_b

            if len(set(scores_a + scores_b)) == 1:
                p_value = 1.0
                statistic = 0.0
            else:
                statistic, p_value = stats.ttest_ind(
                    scores_a,
                    scores_b,
                    equal_var=False
                )

                if math.isnan(p_value):
                    p_value = 1.0

                if math.isnan(statistic):
                    statistic = 0.0

            results.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "mean_a": round(mean_a, 4),
                    "mean_b": round(mean_b, 4),
                    "mean_difference": round(mean_difference, 4),
                    "p_value": round(float(p_value), 6),
                    "statistic": round(float(statistic), 6),
                    "significant": bool(p_value < 0.05)
                }
            )

    return results


def analyse_bias_dimension(
    scored_variants: List[Dict[str, Any]],
    group_field: str,
    dimension_name: str
) -> Dict[str, Any]:
    """
    Analyses one bias dimension, such as name group or university group.
    """
    grouped = group_scores(scored_variants, group_field)

    return {
        "dimension": dimension_name,
        "group_field": group_field,
        "group_summary": calculate_group_summary(grouped),
        "anova": run_anova_test(grouped),
        "pairwise_tests": run_pairwise_tests(grouped)
    }


def create_bias_chart(
    analysis_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Creates a simple bar chart showing mean score by group for each dimension.
    """
    labels = []
    means = []

    for dimension in analysis_results:
        dimension_name = dimension["dimension"]

        for group, summary in dimension["group_summary"].items():
            labels.append(f"{dimension_name}\n{group}")
            means.append(summary["mean_score"])

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means)
    plt.ylabel("Mean candidate score")
    plt.title("Bias Audit: Mean Score by Variant Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def identify_flags(analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies potential bias flags from statistically significant results.
    """
    flags = []

    for dimension in analysis_results:
        for test in dimension["pairwise_tests"]:
            if test["significant"]:
                flags.append(
                    {
                        "dimension": dimension["dimension"],
                        "group_a": test["group_a"],
                        "group_b": test["group_b"],
                        "mean_difference": test["mean_difference"],
                        "p_value": test["p_value"],
                        "interpretation": (
                            f"Scores differ significantly between "
                            f"{test['group_a']} and {test['group_b']}."
                        )
                    }
                )

    return flags


def create_bias_report(
    candidates: List[Dict[str, Any]],
    scored_results: Dict[str, Dict[str, Any]],
    scored_variants: List[Dict[str, Any]],
    analysis_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Creates the final bias report JSON.
    """
    flags = identify_flags(analysis_results)

    original_scores = {
        candidate_id: result.get("score")
        for candidate_id, result in scored_results.items()
    }

    if flags:
        overall_finding = (
            "Potential score differences were detected across one or more "
            "variant groups. These should be reviewed before deployment."
        )
    else:
        overall_finding = (
            "No statistically significant score shifts were detected across "
            "the tested name, university, or employment gap variants."
        )

    return {
        "audit_type": "counterfactual_variant_bias_audit",
        "candidate_count": len(candidates),
        "variant_count": len(scored_variants),
        "original_task3_scores": original_scores,
        "overall_finding": overall_finding,
        "important_design_note": (
            "The audit scoring function intentionally excludes name, demographic proxy, "
            "university name, university tier, and employment gap from the score. "
            "This supports fairness by preventing these fields from directly affecting ranking."
        ),
        "bias_flags": flags,
        "analysis": analysis_results,
        "sample_variants": scored_variants[:10]
    }


def run_bias_detection() -> None:
    """
    Main workflow for Task 4.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()
    scored_results = load_scored_results()
    job_profile = load_job_profile()

    print(f"Loaded {len(candidates)} candidate(s).")
    print(f"Loaded {len(scored_results)} scored result(s).")
    print(f"Loaded job profile: {job_profile.get('title', 'unknown job')}")

    scored_variants = run_variant_scoring(candidates, job_profile)

    print(f"Generated and scored {len(scored_variants)} candidate variants.")

    analysis_results = [
        analyse_bias_dimension(
            scored_variants,
            group_field="name_variant_group",
            dimension_name="Name Variant"
        ),
        analyse_bias_dimension(
            scored_variants,
            group_field="university_variant_group",
            dimension_name="University Variant"
        ),
        analyse_bias_dimension(
            scored_variants,
            group_field="gap_variant_group",
            dimension_name="Employment Gap Variant"
        )
    ]

    report = create_bias_report(
        candidates=candidates,
        scored_results=scored_results,
        scored_variants=scored_variants,
        analysis_results=analysis_results
    )

    report_path = OUTPUT_DIR / "bias_report.json"
    save_json(report, report_path)

    variants_path = OUTPUT_DIR / "bias_scored_variants.json"
    save_json({"variants": scored_variants}, variants_path)

    chart_path = OUTPUT_DIR / "bias_score_chart.png"
    create_bias_chart(analysis_results, chart_path)

    print(f"Saved bias report: {report_path}")
    print(f"Saved scored variants: {variants_path}")
    print(f"Saved chart: {chart_path}")
    print("Task 4 bias detection complete.")


if __name__ == "__main__":
    run_bias_detection()