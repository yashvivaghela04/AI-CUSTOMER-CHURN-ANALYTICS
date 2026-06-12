"""Generate a research paper document from live churn analytics data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from data_loader import load_data
from kpi import (
    calculate_engagement_churn,
    calculate_high_value_churn,
    calculate_overall_churn,
    calculate_segment_churn,
)
from preprocessing import preprocess_data

OUTPUT_FILE = Path(__file__).resolve().parent / "Customer_Churn_Research_Paper.txt"


def _pct(value: float) -> str:
    if value != value:  # NaN check
        return "N/A"
    return f"{value:.2f}%"


def _compute_stats(df):
    """Compute all statistics used throughout the research paper."""
    overall = calculate_overall_churn(df)
    geo = calculate_segment_churn(df, "Geography")
    age = calculate_segment_churn(df, "AgeGroup")
    gender = calculate_segment_churn(df, "Gender")
    high_value = calculate_high_value_churn(df)
    engagement = calculate_engagement_churn(df).set_index("EngagementStatus")
    risk_pct = float((df["Risk"] == "High Risk").mean() * 100) if "Risk" in df.columns else float("nan")
    high_value_count = int((df["Balance"] >= 50000).sum())

    active_rate = float(engagement.loc["Active", "ChurnRate"]) if "Active" in engagement.index else float("nan")
    inactive_rate = float(engagement.loc["Inactive", "ChurnRate"]) if "Inactive" in engagement.index else float("nan")

    top_geo = geo.iloc[0]
    top_age = age.iloc[0]
    top_gender = gender.iloc[0]
    lowest_geo = geo.iloc[-1]

    return {
        "overall": overall,
        "geo": geo,
        "age": age,
        "gender": gender,
        "high_value": high_value,
        "high_value_count": high_value_count,
        "engagement": engagement,
        "active_rate": active_rate,
        "inactive_rate": inactive_rate,
        "risk_pct": risk_pct,
        "top_geo": top_geo,
        "top_age": top_age,
        "top_gender": top_gender,
        "lowest_geo": lowest_geo,
    }


def _format_segment_table(segment_df, label_col: str) -> str:
    lines = []
    for _, row in segment_df.iterrows():
        lines.append(
            f"  - {row[label_col]}: {_pct(float(row['ChurnRate']))} "
            f"({int(row['CustomerCount']):,} customers)"
        )
    return "\n".join(lines)


def build_research_paper(df, stats: dict) -> str:
    today = date.today().strftime("%B %d, %Y")
    s = stats

    abstract = (
        "Customer churn remains a critical challenge for European retail banks, directly affecting "
        "revenue stability and long-term profitability. This study applies an AI-assisted analytics "
        f"framework to a dataset of 10,000 bank customers to quantify churn patterns across demographic, "
        f"geographic, and financial dimensions. Using data cleaning, feature engineering, and segmentation "
        f"analysis, we identify high-risk cohorts and engagement-driven churn differentials. The overall "
        f"churn rate is {_pct(s['overall'])}, with notable variation across regions and customer segments. "
        "Findings support targeted retention strategies that prioritize inactive, high-balance customers "
        "in the highest-churn geographies."
    )

    intro_p1 = (
        "Customer attrition in banking erodes deposit bases, increases acquisition costs, and weakens "
        "cross-selling opportunities. Banks operating across European markets face heterogeneous customer "
        "behaviors shaped by geography, product usage, credit profiles, and engagement levels. Predicting "
        "and preventing churn therefore requires structured segmentation rather than one-size-fits-all "
        "retention campaigns."
    )
    intro_p2 = (
        "The primary objectives of this research are to (1) characterize the bank customer dataset "
        "and engineered segmentation features, (2) measure churn rates across key customer dimensions, "
        "(3) identify high-risk segments including high-value and inactive accounts, and (4) translate "
        "analytical findings into actionable retention recommendations. An interactive dashboard "
        "companion tool enables stakeholders to explore these insights dynamically."
    )

    column_descriptions = [
        ("Year", "Calendar year associated with the customer record."),
        ("CustomerId", "Unique identifier for each customer."),
        ("Surname", "Customer surname (removed during preprocessing as non-analytical)."),
        ("CreditScore", "Numeric creditworthiness score assigned to the customer."),
        ("Geography", "Country of residence (France, Germany, or Spain)."),
        ("Gender", "Customer gender (Male or Female)."),
        ("Age", "Customer age in years."),
        ("Tenure", "Number of years the customer has been with the bank."),
        ("Balance", "Account balance held at the bank."),
        ("NumOfProducts", "Count of bank products used by the customer."),
        ("HasCrCard", "Binary indicator (1/0) of credit card ownership."),
        ("IsActiveMember", "Binary indicator (1/0) of active membership status."),
        ("EstimatedSalary", "Estimated annual salary of the customer."),
        ("Exited", "Binary target variable (1 = churned, 0 = retained)."),
    ]
    dataset_lines = "\n".join(f"  - {name}: {desc}" for name, desc in column_descriptions)

    methodology = (
        "Data cleaning removed non-analytical identifiers (Surname), imputed missing numeric values "
        "with column medians and categorical values with modes, and validated binary target fields. "
        "Feature engineering created five segmentation attributes: AgeGroup (Young, Mid-Age, Senior, Elder), "
        "CreditScoreBand (Low, Medium, High), TenureGroup (New, Mid, Long), BalanceSegment (Zero, Low, High), "
        "and Risk (High Risk when Balance > 50,000 and IsActiveMember = 0; otherwise Normal). "
        "Churn KPIs were computed as the percentage of customers with Exited = 1 within each segment."
    )

    eda = f"""Overall Churn Rate: {_pct(s['overall'])}

Churn by Geography:
{_format_segment_table(s['geo'], 'Geography')}

Churn by Age Group:
{_format_segment_table(s['age'], 'AgeGroup')}

Churn by Gender:
{_format_segment_table(s['gender'], 'Gender')}

High-Value Customer Churn (Balance >= 50,000):
  - Churn Rate: {_pct(s['high_value'])}
  - Customer Count: {s['high_value_count']:,}

Active vs Inactive Member Churn:
  - Active: {_pct(s['active_rate'])}
  - Inactive: {_pct(s['inactive_rate'])}
  - Gap (Inactive minus Active): {_pct(s['inactive_rate'] - s['active_rate'])}"""

    findings = [
        f"The overall customer churn rate across the portfolio is {_pct(s['overall'])}, establishing the baseline retention challenge.",
        f"{s['top_geo']['Geography']} exhibits the highest geographic churn at {_pct(float(s['top_geo']['ChurnRate']))}, "
        f"compared with {s['lowest_geo']['Geography']} at {_pct(float(s['lowest_geo']['ChurnRate']))}.",
        f"The {s['top_age']['AgeGroup']} age cohort shows the highest churn ({_pct(float(s['top_age']['ChurnRate']))}), "
        "indicating demographic concentration of attrition risk.",
        f"Inactive members churn at {_pct(s['inactive_rate'])}, which is {_pct(s['inactive_rate'] - s['active_rate'])} "
        f"higher than active members ({_pct(s['active_rate'])}).",
        f"High-value customers (balance >= EUR 50,000) represent {s['high_value_count']:,} accounts with a churn rate of "
        f"{_pct(s['high_value'])}, and {_pct(s['risk_pct'])} of all customers are classified as High Risk.",
    ]

    recommendations = [
        "Deploy geography-specific retention programs in the highest-churn region, combining localized offers and relationship-manager outreach.",
        "Launch re-engagement campaigns targeting inactive members with personalized product bundles and digital banking incentives.",
        "Implement proactive monitoring and white-glove service for high-balance accounts to protect revenue concentration.",
        "Design age-tailored communication for the highest-churn age cohort, emphasizing loyalty benefits and financial advisory support.",
        "Integrate the Risk segmentation flag into CRM workflows so relationship teams prioritize High Risk customers before exit events.",
    ]

    conclusion_p1 = (
        f"This analysis of 10,000 European bank customers demonstrates that churn is not uniformly distributed: "
        f"overall attrition stands at {_pct(s['overall'])}, with meaningful spread across geography, age, gender, "
        "and engagement status. Inactive customers and high-balance segments warrant disproportionate retention "
        "investment because disengagement amplifies financial exposure."
    )
    conclusion_p2 = (
        "The combination of engineered segmentation features and KPI-driven reporting provides a reproducible "
        "foundation for both strategic planning and operational intervention. Future work may extend this framework "
        "with predictive machine learning models and A/B testing of retention campaigns to quantify uplift."
    )

    references = [
        "Kaggle. Bank Customer Churn Modelling Dataset.",
        "Verbeke, W., Martens, D., & Baesens, B. (2014). Social network analysis for customer churn prediction.",
        "Neslin, S. A., & Van Heerde, H. J. (2022). Customer Retention Models and Strategies.",
        "Streamlit Inc. Streamlit Documentation — Interactive Data Applications.",
    ]

    sections = [
        "TITLE: AI-Driven Customer Churn Analysis and Retention Strategy",
        "AUTHOR: Yashvi Chunilal Vaghela",
        f"DATE: {today}",
        "",
        "=" * 72,
        "SECTION 1 - ABSTRACT",
        "=" * 72,
        abstract,
        "",
        "=" * 72,
        "SECTION 2 - INTRODUCTION",
        "=" * 72,
        intro_p1,
        "",
        intro_p2,
        "",
        "=" * 72,
        "SECTION 3 - DATASET DESCRIPTION",
        "=" * 72,
        "The dataset contains 10,000 customer records and 14 columns:",
        "",
        dataset_lines,
        "",
        "=" * 72,
        "SECTION 4 - METHODOLOGY",
        "=" * 72,
        methodology,
        "",
        "=" * 72,
        "SECTION 5 - EDA",
        "=" * 72,
        eda,
        "",
        "=" * 72,
        "SECTION 6 - KEY FINDINGS",
        "=" * 72,
        *[f"  * {item}" for item in findings],
        "",
        "=" * 72,
        "SECTION 7 - RECOMMENDATIONS",
        "=" * 72,
        *[f"  {i + 1}. {item}" for i, item in enumerate(recommendations)],
        "",
        "=" * 72,
        "SECTION 8 - CONCLUSION",
        "=" * 72,
        conclusion_p1,
        "",
        conclusion_p2,
        "",
        "=" * 72,
        "SECTION 9 - REFERENCES",
        "=" * 72,
        *[f"  [{i + 1}] {ref}" for i, ref in enumerate(references)],
    ]

    return "\n".join(sections) + "\n"


def main() -> None:
    raw_df = load_data()
    df = preprocess_data(raw_df)
    stats = _compute_stats(df)
    content = build_research_paper(df, stats)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print("Research paper saved successfully!")


if __name__ == "__main__":
    main()
