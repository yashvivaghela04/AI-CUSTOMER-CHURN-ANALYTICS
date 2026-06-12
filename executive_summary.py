"""Generate an executive summary document from live churn analytics data."""

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

OUTPUT_FILE = Path(__file__).resolve().parent / "Customer_Churn_Executive_Summary.txt"


def _pct(value: float) -> str:
    if value != value:
        return "N/A"
    return f"{value:.2f}%"


def _compute_stats(df):
    """Compute all statistics used throughout the executive summary."""
    overall = calculate_overall_churn(df)
    geo = calculate_segment_churn(df, "Geography")
    age = calculate_segment_churn(df, "AgeGroup")
    balance = calculate_segment_churn(df, "BalanceSegment")
    credit = calculate_segment_churn(df, "CreditScoreBand")
    high_value = calculate_high_value_churn(df)
    engagement = calculate_engagement_churn(df).set_index("EngagementStatus")

    active_rate = float(engagement.loc["Active", "ChurnRate"]) if "Active" in engagement.index else float("nan")
    inactive_rate = float(engagement.loc["Inactive", "ChurnRate"]) if "Inactive" in engagement.index else float("nan")
    risk_pct = float((df["Risk"] == "High Risk").mean() * 100) if "Risk" in df.columns else float("nan")
    risk_count = int((df["Risk"] == "High Risk").sum()) if "Risk" in df.columns else 0
    high_value_count = int((df["Balance"] >= 50000).sum())
    retained_pct = 100.0 - overall if overall == overall else float("nan")

    top_geo = geo.iloc[0]
    top_age = age.iloc[0]
    top_balance = balance.iloc[0]
    top_credit = credit.iloc[0]

    high_risk_segments = []
    if "Risk" in df.columns:
        risk_churn = calculate_segment_churn(df, "Risk")
        high_risk_row = risk_churn[risk_churn["Risk"] == "High Risk"]
        if not high_risk_row.empty:
            high_risk_segments.append(
                f"High Risk flag (Balance > 50,000 & Inactive): {_pct(float(high_risk_row.iloc[0]['ChurnRate']))} "
                f"churn across {risk_count:,} customers ({_pct(risk_pct)} of portfolio)"
            )
    high_risk_segments.append(
        f"Highest geography — {top_geo['Geography']}: {_pct(float(top_geo['ChurnRate']))} "
        f"({int(top_geo['CustomerCount']):,} customers)"
    )
    high_risk_segments.append(
        f"Highest age group — {top_age['AgeGroup']}: {_pct(float(top_age['ChurnRate']))} "
        f"({int(top_age['CustomerCount']):,} customers)"
    )
    high_risk_segments.append(
        f"Highest balance segment — {top_balance['BalanceSegment']}: {_pct(float(top_balance['ChurnRate']))} "
        f"({int(top_balance['CustomerCount']):,} customers)"
    )
    high_risk_segments.append(
        f"Inactive members: {_pct(inactive_rate)} churn vs {_pct(active_rate)} for active members"
    )

    return {
        "overall": overall,
        "retained_pct": retained_pct,
        "high_value": high_value,
        "high_value_count": high_value_count,
        "active_rate": active_rate,
        "inactive_rate": inactive_rate,
        "risk_pct": risk_pct,
        "risk_count": risk_count,
        "top_geo": top_geo,
        "top_age": top_age,
        "top_balance": top_balance,
        "top_credit": top_credit,
        "geo": geo,
        "high_risk_segments": high_risk_segments,
        "total_customers": len(df),
    }


def build_executive_summary(stats: dict) -> str:
    today = date.today().strftime("%B %d, %Y")
    s = stats

    business_context = (
        "European retail banks face sustained pressure to retain customers amid competitive product offerings "
        "and digital-first alternatives. Customer churn erodes deposit balances, reduces fee income, and "
        "increases the cost of replacement acquisition. This executive summary distills data-driven insights "
        "from 10,000 customer records to guide immediate and strategic retention decisions."
    )

    key_metrics = f"""  - Total Customers Analyzed: {s['total_customers']:,}
  - Overall Churn Rate: {_pct(s['overall'])}
  - Retention Rate: {_pct(s['retained_pct'])}
  - High-Value Customer Churn (Balance >= 50,000): {_pct(s['high_value'])} ({s['high_value_count']:,} customers)
  - Active Member Churn: {_pct(s['active_rate'])}
  - Inactive Member Churn: {_pct(s['inactive_rate'])}
  - High Risk Portfolio Share: {_pct(s['risk_pct'])} ({s['risk_count']:,} customers)"""

    critical_findings = [
        f"Portfolio-wide churn stands at {_pct(s['overall'])}, meaning roughly one in five customers has exited.",
        f"{s['top_geo']['Geography']} is the highest-churn geography at {_pct(float(s['top_geo']['ChurnRate']))}.",
        f"The {s['top_age']['AgeGroup']} segment leads age-based attrition at {_pct(float(s['top_age']['ChurnRate']))}.",
        f"Inactive customers churn at {_pct(s['inactive_rate'])}, exceeding active customers by "
        f"{_pct(s['inactive_rate'] - s['active_rate'])}.",
        f"High-value accounts (>= EUR 50,000 balance) churn at {_pct(s['high_value'])}, representing material revenue at risk.",
    ]

    immediate_actions = [
        f"Assign relationship managers to the {s['risk_count']:,} High Risk customers flagged by balance and inactivity.",
        f"Launch a 30-day win-back campaign in {s['top_geo']['Geography']} targeting the top churn geography.",
        "Activate dormant accounts through personalized digital engagement and fee-relief incentives for inactive members.",
    ]

    strategic_recommendations = [
        "Embed churn KPIs and Risk flags into CRM dashboards for weekly executive review and branch-level accountability.",
        "Develop segment-specific loyalty programs aligned to age cohorts and credit score bands with measurable retention targets.",
        "Establish a high-value customer protection program combining proactive outreach, advisory services, and escalation protocols.",
    ]

    expected_impact = (
        f"Targeting the highest-churn geography ({s['top_geo']['Geography']}) and inactive member gap "
        f"({_pct(s['inactive_rate'] - s['active_rate'])} above active baseline) can materially reduce portfolio churn "
        f"from the current {_pct(s['overall'])} baseline. Protecting {s['high_value_count']:,} high-value accounts "
        "should stabilize core deposit revenue while re-engagement programs convert disengaged customers into active, lower-risk relationships."
    )

    sections = [
        "TITLE: Executive Summary — AI-Driven Customer Churn Analytics",
        "AUTHOR: Yashvi Chunilal Vaghela",
        f"DATE: {today}",
        "",
        "=" * 72,
        "SECTION 1 - BUSINESS CONTEXT",
        "=" * 72,
        business_context,
        "",
        "=" * 72,
        "SECTION 2 - KEY METRICS",
        "=" * 72,
        key_metrics,
        "",
        "=" * 72,
        "SECTION 3 - CRITICAL FINDINGS",
        "=" * 72,
        *[f"  * {item}" for item in critical_findings],
        "",
        "=" * 72,
        "SECTION 4 - HIGH RISK SEGMENTS",
        "=" * 72,
        *[f"  * {item}" for item in s["high_risk_segments"]],
        "",
        "=" * 72,
        "SECTION 5 - IMMEDIATE ACTION ITEMS",
        "=" * 72,
        *[f"  {i + 1}. {item}" for i, item in enumerate(immediate_actions)],
        "",
        "=" * 72,
        "SECTION 6 - STRATEGIC RECOMMENDATIONS",
        "=" * 72,
        *[f"  {i + 1}. {item}" for i, item in enumerate(strategic_recommendations)],
        "",
        "=" * 72,
        "SECTION 7 - EXPECTED IMPACT",
        "=" * 72,
        expected_impact,
    ]

    return "\n".join(sections) + "\n"


def main() -> None:
    raw_df = load_data()
    df = preprocess_data(raw_df)
    stats = _compute_stats(df)
    content = build_executive_summary(stats)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print("Executive summary saved successfully!")


if __name__ == "__main__":
    main()
