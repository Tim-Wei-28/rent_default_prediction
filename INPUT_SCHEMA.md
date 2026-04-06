# Rent Default Risk API — Input Field Reference

Send a `POST /predict` request with a JSON body. All monetary values are in **GBP**.

---

## Tier 1 Fields — Always Required (Self-Reported)

These seven fields are collected directly from the applicant during sign-up or
application. No third-party data source is needed.

| Field | Type | Unit / Format | Valid Values | Notes |
|---|---|---|---|---|
| `employment_status` | string | — | `"Employed"`, `"Self-employed"`, `"Part-time"`, `"Not employed"`, `"Retired"`, `"Student"`, `"Other"` | Ask applicant to select from dropdown. |
| `employment_duration_months` | number | Whole months | `0` – no upper limit | Duration in current job/status. Convert years × 12. E.g. 2 years → `24`. |
| `monthly_income_gbp` | number | GBP per month | `> 0` | **Gross** (pre-tax) monthly income. If applicant provides annual salary, divide by 12. E.g. £42,000/year → `3500.0`. |
| `monthly_rent_gbp` | number | GBP per month | `> 0`, `≤ 50,000` | Monthly rent for the specific property being applied for. Taken from the listing — applicant does not enter this. |
| `income_range` | string | Annual gross bracket | See below | Select the band that matches `monthly_income_gbp × 12`. |
| `income_verified` | boolean | — | `true` / `false` | Set to `true` only if a payslip, bank statement, or HMRC letter has been uploaded and checked. |

### `income_range` valid values

| Value | Annual Gross Equivalent |
|---|---|
| `"Not employed"` | £0 (no income) |
| `"£0–£24,999"` | Up to £24,999/year |
| `"£25,000–£49,999"` | £25,000–£49,999/year |
| `"£50,000–£74,999"` | £50,000–£74,999/year |
| `"£75,000–£99,999"` | £75,000–£99,999/year |
| `"£100,000+"` | £100,000+/year |
| `"Not displayed"` | Applicant declined to state |

---

## Tier 2 Fields — Optional (Credit Bureau / Open Banking)

Providing **at least 2** of these fields automatically upgrades scoring to
Tier 2 (AUC ~0.75 vs 0.64 for Tier 1). All are sourced from a credit reference
agency (e.g. Experian, Equifax, TransUnion) or an Open Banking data feed.

> **Legal note:** Obtaining bureau data requires the applicant's explicit
> consent under UK GDPR (Article 6(1)(a)) and, if used to make automated
> credit-like decisions, compliance with Article 22. See `legal_summary.txt`.

| Field | Type | Unit / Format | Valid Range | Source | Notes |
|---|---|---|---|---|---|
| `credit_score` | number | Points | `300` – `850` | Credit bureau | Midpoint of bureau score range. If bureau returns a band (e.g. 680–720), send `700.0`. |
| `debt_to_income_ratio` | number | Ratio (0–10) | `0.0` – `10.0` | Bureau / Open Banking | Total monthly committed debt payments ÷ gross monthly income. E.g. £800 debt / £3,200 income = `0.25`. |
| `bankcard_utilisation` | number | Ratio (0–1) | `0.0` – `1.0` | Bureau | Revolving credit used ÷ total revolving limit. E.g. £1,500 used / £5,000 limit = `0.30`. |
| `open_revolving_monthly_payment_gbp` | number | GBP/month | `≥ 0` | Bureau / Open Banking | Sum of current monthly minimum payments on revolving accounts (credit cards, overdrafts). |
| `revolving_credit_balance_gbp` | number | GBP | `≥ 0` | Bureau | Total outstanding revolving credit balance. |
| `available_bankcard_credit_gbp` | number | GBP | `≥ 0` | Bureau | Total unused revolving credit limit. |
| `open_credit_lines` | integer | Count | `≥ 0` | Bureau | Number of currently open credit lines. |
| `current_credit_lines` | integer | Count | `≥ 0` | Bureau | Number of active (non-closed) credit accounts. |
| `total_credit_lines_past_7_years` | integer | Count | `≥ 0` | Bureau | All credit lines opened in the past 7 years. |
| `total_trades` | integer | Count | `≥ 0` | Bureau | Total credit accounts ever opened. |
| `trades_never_delinquent_pct` | number | Ratio (0–1) | `0.0` – `1.0` | Bureau | Proportion of all trades that were never delinquent. E.g. `0.92` = 92% clean. |
| `delinquencies_last_7_years` | integer | Count | `≥ 0` | Bureau | Total delinquencies in the past 7 years. |
| `current_delinquencies` | integer | Count | `≥ 0` | Bureau | Currently open delinquent accounts. |
| `amount_delinquent_gbp` | number | GBP | `≥ 0` | Bureau | Total amount currently past due. |
| `public_records_last_10_years` | integer | Count | `≥ 0` | Bureau | CCJs, IVAs, bankruptcies in the past 10 years. |
| `public_records_last_12_months` | integer | Count | `≥ 0` | Bureau | Public records in the past 12 months. |
| `inquiries_last_6_months` | integer | Count | `≥ 0` | Bureau | Hard credit searches in the past 6 months. |
| `total_inquiries` | integer | Count | `≥ 0` | Bureau | Total hard searches ever on record. |
| `trades_opened_last_6_months` | integer | Count | `≥ 0` | Bureau | New credit accounts opened in the past 6 months. |
| `credit_history_months` | number | Months | `≥ 0` | Bureau | Age of credit file: months from first recorded credit line to today. |

---

## Example Request Bodies

### Tier 1 Only (self-reported)

```json
{
  "employment_status": "Employed",
  "employment_duration_months": 36,
  "monthly_income_gbp": 3500.0,
  "monthly_rent_gbp": 1400.0,
  "income_range": "£25,000–£49,999",
  "income_verified": true
}
```

### Tier 2 (with bureau data)

```json
{
  "employment_status": "Employed",
  "employment_duration_months": 36,
  "monthly_income_gbp": 3500.0,
  "monthly_rent_gbp": 1400.0,
  "income_range": "£25,000–£49,999",
  "income_verified": true,
  "credit_score": 710.0,
  "debt_to_income_ratio": 0.22,
  "bankcard_utilisation": 0.28,
  "inquiries_last_6_months": 1,
  "delinquencies_last_7_years": 0,
  "public_records_last_10_years": 0,
  "open_credit_lines": 4,
  "credit_history_months": 84.0
}
```

---

## Example Response

```json
{
  "risk_label": "Good",
  "default_probability": 0.1823,
  "tier_used": 2,
  "features_used": 33,
  "explanation": "Below-average predicted default risk. Minor risk factors are present but the overall profile is positive."
}
```

### Risk Labels

| Label | Default Probability | Recommended Action |
|---|---|---|
| `Strong` | < 15% | Proceed with confidence |
| `Good` | 15%–25% | Standard referencing |
| `Limited` | 25%–40% | Request guarantor or extra references |
| `Concerns` | ≥ 40% | Detailed manual review required |

---

## Notes for Developers

- The API automatically selects Tier 1 or Tier 2 based on how many optional
  bureau fields are provided (≥ 2 bureau fields → Tier 2).
- All monetary fields should use the **monthly** GBP value unless stated otherwise.
- The `monthly_rent_gbp` field comes from the **property listing**, not from
  the applicant — the frontend should inject it automatically.
- Probabilities are calibrated to the Prosper Loan Dataset (US consumer loans,
  30.77% base default rate). Absolute values should be interpreted relatively,
  not as precise UK rental-market probabilities.
- Models: XGBoost (tree_method='hist'), trained with RepeatedStratifiedKFold(5,3)
  + RandomizedSearchCV(50). Tier 1 AUC 0.6361 | Tier 2 AUC 0.7467.
