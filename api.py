"""
Rent Default Risk Prediction — FastAPI Microservice
====================================================
Serves two XGBoost models trained on the Prosper Loan Dataset:
  Tier 1  : Self-reported only (7 features)       — AUC 0.6361
  Tier 2  : + Credit bureau / platform data (31)  — AUC 0.7467

The tier is selected automatically based on which fields the caller
provides. If bureau / open-banking fields are present, Tier 2 is used;
otherwise the service falls back to Tier 1.

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent / "models"

_tier1_model = None
_tier2_model = None


def _load_models() -> None:
    global _tier1_model, _tier2_model
    t1_path = BASE_DIR / "tier1_model_xgboost.pkl"
    t2_path = BASE_DIR / "platform_model_xgboost.pkl"
    if not t1_path.exists():
        raise FileNotFoundError(f"Tier 1 model not found: {t1_path}")
    if not t2_path.exists():
        raise FileNotFoundError(f"Tier 2 model not found: {t2_path}")
    _tier1_model = joblib.load(t1_path)
    _tier2_model = joblib.load(t2_path)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Rent Default Risk API",
    description="Predicts the probability of rent default for a tenant applicant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production to your Next.js domain
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    _load_models()


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

EMPLOYMENT_STATUS_VALUES = [
    "Employed", "Self-employed", "Part-time", "Not employed",
    "Retired", "Student", "Other",
]

INCOME_RANGE_VALUES = [
    "Not employed",
    "£0–£24,999",
    "£25,000–£49,999",
    "£50,000–£74,999",
    "£75,000–£99,999",
    "£100,000+",
    "Not displayed",
]

# Ordinal encoding for IncomeRange (must match training label encoding order)
INCOME_RANGE_ORDINAL: dict[str, int] = {
    "Not employed":       0,
    "£0–£24,999":         1,
    "£25,000–£49,999":    2,
    "£50,000–£74,999":    3,
    "£75,000–£99,999":    4,
    "£100,000+":          5,
    "Not displayed":      6,
}


class ApplicantInput(BaseModel):
    # ------------------------------------------------------------------
    # TIER 1 — self-reported (always required)
    # ------------------------------------------------------------------
    employment_status: Literal[
        "Employed", "Self-employed", "Part-time",
        "Not employed", "Retired", "Student", "Other"
    ] = Field(
        ...,
        description="Current employment status of the applicant.",
        examples=["Employed"],
    )

    employment_duration_months: float = Field(
        ...,
        ge=0,
        description=(
            "How long the applicant has been in their current employment, "
            "in whole months. E.g. 24 = 2 years."
        ),
        examples=[24],
    )

    monthly_income_gbp: float = Field(
        ...,
        gt=0,
        description=(
            "Applicant's gross stated monthly income in GBP. "
            "Convert annual salary: annual / 12."
        ),
        examples=[3500.0],
    )

    monthly_rent_gbp: float = Field(
        ...,
        gt=0,
        description="Monthly rent for the property being applied for, in GBP.",
        examples=[1500.0],
    )

    income_range: Literal[
        "Not employed",
        "£0–£24,999",
        "£25,000–£49,999",
        "£50,000–£74,999",
        "£75,000–£99,999",
        "£100,000+",
        "Not displayed",
    ] = Field(
        ...,
        description=(
            "Annual gross income bracket (self-reported). "
            "Select the band that matches monthly_income_gbp * 12."
        ),
        examples=["£25,000–£49,999"],
    )

    income_verified: bool = Field(
        ...,
        description=(
            "Whether the applicant's income has been independently verified "
            "(e.g. payslip, bank statement upload)."
        ),
        examples=[True],
    )

    # ------------------------------------------------------------------
    # TIER 2 — credit bureau / open-banking (all optional)
    # Providing any of these fields activates Tier 2 scoring.
    # ------------------------------------------------------------------
    credit_score: Optional[float] = Field(
        None,
        ge=300,
        le=850,
        description=(
            "Midpoint credit score from bureau (e.g. Experian/Equifax/TransUnion). "
            "Typically 300–850. If the bureau returns a range, pass (lower+upper)/2."
        ),
        examples=[680.0],
    )

    debt_to_income_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description=(
            "Total monthly debt obligations divided by gross monthly income. "
            "E.g. 0.25 means 25% of income goes to existing debt repayments."
        ),
        examples=[0.25],
    )

    bankcard_utilisation: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Revolving credit utilisation ratio (0.0–1.0). "
            "E.g. 0.30 means 30% of credit limit is being used."
        ),
        examples=[0.30],
    )

    open_revolving_monthly_payment_gbp: Optional[float] = Field(
        None,
        ge=0,
        description="Total existing monthly revolving debt payments (credit cards etc.) in GBP.",
        examples=[200.0],
    )

    revolving_credit_balance_gbp: Optional[float] = Field(
        None,
        ge=0,
        description="Total outstanding revolving credit balance in GBP.",
        examples=[5000.0],
    )

    available_bankcard_credit_gbp: Optional[float] = Field(
        None,
        ge=0,
        description="Total unused revolving credit available in GBP.",
        examples=[8000.0],
    )

    open_credit_lines: Optional[int] = Field(
        None,
        ge=0,
        description="Number of currently open credit lines.",
        examples=[4],
    )

    current_credit_lines: Optional[int] = Field(
        None,
        ge=0,
        description="Number of active (non-closed) credit lines.",
        examples=[4],
    )

    total_credit_lines_past_7_years: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of credit lines opened in the past 7 years.",
        examples=[12],
    )

    total_trades: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of credit trades / accounts ever opened.",
        examples=[15],
    )

    trades_never_delinquent_pct: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Proportion of trades that were never delinquent (0.0–1.0).",
        examples=[0.92],
    )

    delinquencies_last_7_years: Optional[int] = Field(
        None,
        ge=0,
        description="Number of delinquencies on record in the past 7 years.",
        examples=[0],
    )

    current_delinquencies: Optional[int] = Field(
        None,
        ge=0,
        description="Number of currently open delinquent accounts.",
        examples=[0],
    )

    amount_delinquent_gbp: Optional[float] = Field(
        None,
        ge=0,
        description="Total GBP amount currently delinquent.",
        examples=[0.0],
    )

    public_records_last_10_years: Optional[int] = Field(
        None,
        ge=0,
        description="Number of public records (CCJs, bankruptcies etc.) in the last 10 years.",
        examples=[0],
    )

    public_records_last_12_months: Optional[int] = Field(
        None,
        ge=0,
        description="Number of public records filed in the last 12 months.",
        examples=[0],
    )

    inquiries_last_6_months: Optional[int] = Field(
        None,
        ge=0,
        description="Number of hard credit searches in the past 6 months.",
        examples=[1],
    )

    total_inquiries: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of hard credit searches ever on record.",
        examples=[5],
    )

    trades_opened_last_6_months: Optional[int] = Field(
        None,
        ge=0,
        description="Number of new credit accounts opened in the past 6 months.",
        examples=[0],
    )

    credit_history_months: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Length of credit history in months, from the date of the first "
            "recorded credit line to today."
        ),
        examples=[96.0],
    )

    @field_validator("monthly_rent_gbp")
    @classmethod
    def rent_sanity(cls, v: float) -> float:
        if v > 50_000:
            raise ValueError("monthly_rent_gbp looks unreasonably high (> £50,000).")
        return v

    @field_validator("monthly_income_gbp")
    @classmethod
    def income_sanity(cls, v: float) -> float:
        if v > 500_000:
            raise ValueError("monthly_income_gbp looks unreasonably high (> £500,000).")
        return v


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

RiskLabel = Literal["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk", "Insufficient Data"]


class PredictionOutput(BaseModel):
    risk_label: RiskLabel = Field(
        description="Risk indicator label: Low Risk / Moderate Risk / Elevated Risk / High Risk."
    )
    default_probability: float = Field(
        description="Model's estimated probability of default (0.0–1.0)."
    )
    tier_used: Literal[1, 2]
    features_used: int
    explanation: str


# ---------------------------------------------------------------------------
# Feature derivation helpers
# ---------------------------------------------------------------------------

TIER1_COLS = [
    "StatedMonthlyIncome",
    "EmploymentStatusDuration",
    "MonthlyLoanPayment",
    "RentToIncomeRatio",
    "IncomeVerifiable",
    "EmploymentStatus",
    "IncomeRange",
]

# Must match training order in train_paper_xgboost_platform.py
TIER2_COLS = [
    # LOG features
    "StatedMonthlyIncome", "EmploymentStatusDuration", "CreditHistoryMonths",
    "MonthlyLoanPayment",
    "RentToIncomeRatio", "MonthlyDebtServiceRatio",
    "DelinquenciesLast7Years", "CurrentDelinquencies", "AmountDelinquent",
    "PublicRecordsLast10Years", "PublicRecordsLast12Months",
    "InquiriesLast6Months", "TotalInquiries", "TradesOpenedLast6Months",
    "OpenRevolvingMonthlyPayment", "RevolvingCreditBalance",
    "AvailableBankcardCredit", "OpenRevolvingAccounts",
    "DelinquencyRatio", "RecentInquiryRatio",
    # SCALE features
    "CreditScore",
    "DebtToIncomeRatio",
    "BankcardUtilization",
    "TradesNeverDelinquent (percentage)",
    "OpenCreditLines", "CurrentCreditLines",
    "TotalCreditLinespast7years", "TotalTrades",
    "AvailableCreditBuffer", "UtilisationXInquiries",
    # BIN
    "IncomeVerifiable",
    # CAT
    "EmploymentStatus",
    "IncomeRange",
]

EMPLOYMENT_STATUS_MAP = {
    "Employed":      "Employed",
    "Self-employed": "Self-employed",
    "Part-time":     "Part-time",
    "Not employed":  "Not employed",
    "Retired":       "Retired",
    "Student":       "Other",
    "Other":         "Other",
}


def _build_tier1_row(inp: ApplicantInput) -> pd.DataFrame:
    rent_to_income = inp.monthly_rent_gbp / max(inp.monthly_income_gbp, 1)
    income_range_label = inp.income_range

    row = {
        "StatedMonthlyIncome":      inp.monthly_income_gbp,
        "EmploymentStatusDuration": inp.employment_duration_months,
        "MonthlyLoanPayment":       inp.monthly_rent_gbp,
        "RentToIncomeRatio":        rent_to_income,
        "IncomeVerifiable":         int(inp.income_verified),
        "EmploymentStatus":         EMPLOYMENT_STATUS_MAP.get(inp.employment_status, "Other"),
        "IncomeRange":              income_range_label,
    }
    return pd.DataFrame([row])[TIER1_COLS]


def _build_tier2_row(inp: ApplicantInput) -> pd.DataFrame:
    rent_to_income      = inp.monthly_rent_gbp / max(inp.monthly_income_gbp, 1)
    revolving_payment   = inp.open_revolving_monthly_payment_gbp or 0.0
    monthly_debt_service = revolving_payment / max(inp.monthly_income_gbp, 1)

    total_inquiries = inp.total_inquiries or 0
    inquiries_6m    = inp.inquiries_last_6_months or 0
    delinquencies_7y = inp.delinquencies_last_7_years or 0
    total_credit_lines = inp.total_credit_lines_past_7_years or 1  # avoid /0
    delinquency_ratio = delinquencies_7y / max(total_credit_lines, 1)
    recent_inquiry_ratio = inquiries_6m / (total_inquiries + 1)

    rev_balance = inp.revolving_credit_balance_gbp or 0.0
    avail_credit = inp.available_bankcard_credit_gbp or 0.0
    available_credit_buffer = avail_credit / max(rev_balance + avail_credit + 1, 1)

    bankcard_util = inp.bankcard_utilisation or 0.0
    utilisation_x_inquiries = bankcard_util * inquiries_6m

    row = {
        # LOG
        "StatedMonthlyIncome":          inp.monthly_income_gbp,
        "EmploymentStatusDuration":     inp.employment_duration_months,
        "CreditHistoryMonths":          inp.credit_history_months or 0.0,
        "MonthlyLoanPayment":           inp.monthly_rent_gbp,
        "RentToIncomeRatio":            rent_to_income,
        "MonthlyDebtServiceRatio":      monthly_debt_service,
        "DelinquenciesLast7Years":      delinquencies_7y,
        "CurrentDelinquencies":         inp.current_delinquencies or 0,
        "AmountDelinquent":             inp.amount_delinquent_gbp or 0.0,
        "PublicRecordsLast10Years":     inp.public_records_last_10_years or 0,
        "PublicRecordsLast12Months":    inp.public_records_last_12_months or 0,
        "InquiriesLast6Months":         inquiries_6m,
        "TotalInquiries":               total_inquiries,
        "TradesOpenedLast6Months":      inp.trades_opened_last_6_months or 0,
        "OpenRevolvingMonthlyPayment":  revolving_payment,
        "RevolvingCreditBalance":       rev_balance,
        "AvailableBankcardCredit":      avail_credit,
        "OpenRevolvingAccounts":        inp.open_credit_lines or 0,
        "DelinquencyRatio":             delinquency_ratio,
        "RecentInquiryRatio":           recent_inquiry_ratio,
        # SCALE
        "CreditScore":                  inp.credit_score or 650.0,
        "DebtToIncomeRatio":            inp.debt_to_income_ratio or 0.0,
        "BankcardUtilization":          bankcard_util,
        "TradesNeverDelinquent (percentage)": inp.trades_never_delinquent_pct or 0.9,
        "OpenCreditLines":              inp.open_credit_lines or 0,
        "CurrentCreditLines":           inp.current_credit_lines or 0,
        "TotalCreditLinespast7years":   inp.total_credit_lines_past_7_years or 0,
        "TotalTrades":                  inp.total_trades or 0,
        "AvailableCreditBuffer":        available_credit_buffer,
        "UtilisationXInquiries":        utilisation_x_inquiries,
        # BIN
        "IncomeVerifiable":             int(inp.income_verified),
        # CAT
        "EmploymentStatus":             EMPLOYMENT_STATUS_MAP.get(inp.employment_status, "Other"),
        "IncomeRange":                  inp.income_range,
    }
    return pd.DataFrame([row])[TIER2_COLS]


# ---------------------------------------------------------------------------
# Risk label mapping
# ---------------------------------------------------------------------------

def _risk_label(prob: float, tier: int) -> tuple[RiskLabel, str]:
    """Map default probability to a human-readable risk label + explanation.

    Tier 1 thresholds (self-reported data only — wider bands reflect lower precision):
      < 46%  → Low Risk
      46–60% → Moderate Risk
      ≥ 60%  → High Risk

    Tier 2 thresholds (bureau data — tighter bands reflect higher model confidence):
      < 15%  → Low Risk
      15–25% → Moderate Risk
      25–40% → Elevated Risk
      ≥ 40%  → High Risk
    """
    if tier == 1:
        if prob < 0.46:
            return (
                "Low Risk",
                "The applicant's profile is consistent with reliable rent payment. "
                "Proceed with standard referencing.",
            )
        elif prob < 0.60:
            return (
                "Moderate Risk",
                "Some risk factors are present. Consider additional referencing "
                "before proceeding.",
            )
        else:
            return (
                "High Risk",
                "Multiple risk factors identified. Detailed manual due diligence "
                "is strongly recommended before accepting this applicant.",
            )
    else:
        # Tier 2 — bureau data, tighter thresholds
        if prob < 0.15:
            return (
                "Low Risk",
                "The applicant's profile is consistent with reliable rent payment. "
                "Proceed with standard referencing.",
            )
        elif prob < 0.25:
            return (
                "Moderate Risk",
                "Below-average predicted default risk. Minor risk factors are present "
                "but the overall profile is positive.",
            )
        elif prob < 0.40:
            return (
                "Elevated Risk",
                "Several risk factors are present. Consider requesting a guarantor "
                "or additional references before proceeding.",
            )
        else:
            return (
                "High Risk",
                "Multiple risk factors identified. Detailed manual due diligence "
                "is strongly recommended before accepting this applicant.",
            )


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

def _has_bureau_data(inp: ApplicantInput) -> bool:
    """Return True if enough bureau fields are provided to use Tier 2."""
    tier2_fields = [
        inp.credit_score,
        inp.debt_to_income_ratio,
        inp.bankcard_utilisation,
        inp.inquiries_last_6_months,
        inp.delinquencies_last_7_years,
    ]
    return sum(f is not None for f in tier2_fields) >= 2


@app.post("/predict", response_model=PredictionOutput, summary="Predict rent default risk")
def predict(inp: ApplicantInput) -> PredictionOutput:
    """
    Returns a risk label, raw default probability, and which tier model was used.

    - **Tier 1** is used when only self-reported data is provided (AUC ~0.64).
    - **Tier 2** is used when at least 2 credit bureau / open-banking fields are
      present alongside the Tier 1 fields (AUC ~0.75).
    """
    if _tier1_model is None or _tier2_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    use_tier2 = _has_bureau_data(inp)

    try:
        if use_tier2:
            row = _build_tier2_row(inp)
            prob = float(_tier2_model.predict_proba(row)[0, 1])
            tier = 2
            n_features = len(TIER2_COLS)
        else:
            row = _build_tier1_row(inp)
            prob = float(_tier1_model.predict_proba(row)[0, 1])
            tier = 1
            n_features = len(TIER1_COLS)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    label, explanation = _risk_label(prob, tier)

    return PredictionOutput(
        risk_label=label,
        default_probability=round(prob, 4),
        tier_used=tier,
        features_used=n_features,
        explanation=explanation,
    )


@app.get("/health", summary="Health check")
def health() -> dict:
    return {
        "status": "ok",
        "tier1_loaded": _tier1_model is not None,
        "tier2_loaded": _tier2_model is not None,
    }


@app.get("/", summary="Service info")
def root() -> dict:
    return {
        "service": "Rent Default Risk API",
        "version": "1.0.0",
        "docs": "/docs",
    }
