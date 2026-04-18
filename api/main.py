import os
import numpy as np
import joblib
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Load model bundle once at startup ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models.pkl")

try:
    bundle = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"models.pkl not found at {MODEL_PATH}")

scaler   = bundle["scaler"]
gb_ctrl  = bundle["gb_ctrl"]
gb_mens  = bundle["gb_mens"]
gb_wom   = bundle["gb_wom"]
base     = bundle["baselines"]
FEATURES = bundle["FEATURES"]
hist_map = bundle["hist_map"]
zip_map  = bundle["zip_map"]
ch_map   = bundle["ch_map"]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Email Campaign Optimization using Uplift Modeling & A/B Testing",
    description="""
## Hillstrom MineThatData T-Learner Uplift Model

This API predicts the **incremental lift** an email campaign will have on a specific customer visiting the website.

### How it works
- **T-Learner** (Two-model learner): three separate Gradient Boosting models trained on each experimental arm (Control, Mens Email, Womens Email).
- **Uplift** = P(visit | email) − P(visit | no email)
- Customers with high uplift are **persuadables** ,they respond to email.
- Customers with low uplift would visit anyway, or won't respond regardless.

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Score a single customer |
| `POST /predict/batch` | Score multiple customers at once |
| `GET /baselines` | Population-level A/B test results |
| `GET /health` | Health check |
""",
    version="1.0.0",
    contact={
        "name": "Winkle",
        "url": "https://www.linkedin.com/in/winkle-data-scientist/",
    },
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

HistorySegment = Literal[
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]
ZipCode  = Literal["Rural", "Surburban", "Urban"]
Channel  = Literal["Phone", "Web", "Multichannel"]


class CustomerInput(BaseModel):
    """Features for a single customer."""

    recency: int = Field(
        ..., ge=1, le=12,
        description="Months since last purchase (1 = very recent, 12 = lapsed)",
        example=6,
    )
    history: float = Field(
        ..., ge=0,
        description="Total historical spend in dollars",
        example=242.0,
    )
    mens: int = Field(
        ..., ge=0, le=1,
        description="Has previously bought mens items (1 = yes, 0 = no)",
        example=1,
    )
    womens: int = Field(
        ..., ge=0, le=1,
        description="Has previously bought womens items (1 = yes, 0 = no)",
        example=1,
    )
    zip_code: ZipCode = Field(
        ...,
        description="Zip code type: Rural, Surburban, or Urban",
        example="Surburban",
    )
    newbie: int = Field(
        ..., ge=0, le=1,
        description="New customer acquired in past 12 months (1 = yes, 0 = no)",
        example=0,
    )
    channel: Channel = Field(
        ...,
        description="Acquisition channel: Phone, Web, or Multichannel",
        example="Web",
    )
    history_segment: HistorySegment = Field(
        ...,
        description="Spend history band this customer falls into",
        example="3) $200 - $350",
    )

    @field_validator("mens", "womens", "newbie")
    @classmethod
    def must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("Must be 0 or 1")
        return v


class BatchInput(BaseModel):
    """A list of customers for batch scoring."""
    customers: list[CustomerInput] = Field(
        ..., min_length=1, max_length=10000,
        description="List of customer records to score (max 10,000 per request)",
    )


class UpliftPrediction(BaseModel):
    """Uplift scores and recommendation for one customer."""
    p_visit_no_email:     float = Field(..., description="P(visit | no email sent)")
    p_visit_mens_email:   float = Field(..., description="P(visit | mens email sent)")
    p_visit_womens_email: float = Field(..., description="P(visit | womens email sent)")
    uplift_mens:          float = Field(..., description="Incremental lift from mens email (pp)")
    uplift_womens:        float = Field(..., description="Incremental lift from womens email (pp)")
    best_campaign:        str   = Field(..., description="Campaign with highest predicted uplift")
    best_uplift:          float = Field(..., description="Maximum predicted uplift (pp)")
    recommendation:       str   = Field(..., description="Send | Borderline | Skip")


class BatchPrediction(BaseModel):
    total_customers:  int                  = Field(..., description="Number of customers scored")
    send_count:       int                  = Field(..., description="Customers recommended to receive email")
    borderline_count: int                  = Field(..., description="Borderline customers")
    skip_count:       int                  = Field(..., description="Customers to skip")
    predictions:      list[UpliftPrediction]


class BaselinesResponse(BaseModel):
    """Population-level A/B test results from the original experiment."""
    control_visit_rate:       float
    mens_email_visit_rate:    float
    womens_email_visit_rate:  float
    mens_email_uplift_pp:     float
    womens_email_uplift_pp:   float
    control_conversion_rate:  float
    mens_conversion_rate:     float
    womens_conversion_rate:   float
    sample_size_total:        int
    note: str


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(customer: CustomerInput) -> np.ndarray:
    h = hist_map[customer.history_segment]
    z = zip_map[customer.zip_code]
    c = ch_map[customer.channel]
    both     = int(customer.mens == 1 and customer.womens == 1)
    rec_hist = customer.recency * np.log1p(customer.history)
    high_val = int(h >= 4)
    raw = np.array([[
        customer.recency, customer.history,
        customer.mens, customer.womens,
        z, customer.newbie, c, h,
        both, rec_hist, high_val,
    ]])
    return scaler.transform(raw)


def score_features(xs: np.ndarray) -> UpliftPrediction:
    p_ctrl = float(gb_ctrl.predict_proba(xs)[0, 1])
    p_mens = float(gb_mens.predict_proba(xs)[0, 1])
    p_wom  = float(gb_wom.predict_proba(xs)[0, 1])

    uplift_mens = round(p_mens - p_ctrl, 6)
    uplift_wom  = round(p_wom  - p_ctrl, 6)
    best_uplift = max(uplift_mens, uplift_wom)
    best_campaign = "Mens email" if uplift_mens >= uplift_wom else "Womens email"

    if best_uplift >= 0.08:
        recommendation = "Send"
    elif best_uplift >= 0.03:
        recommendation = "Borderline"
    else:
        recommendation = "Skip"

    return UpliftPrediction(
        p_visit_no_email=round(p_ctrl, 6),
        p_visit_mens_email=round(p_mens, 6),
        p_visit_womens_email=round(p_wom, 6),
        uplift_mens=uplift_mens,
        uplift_womens=uplift_wom,
        best_campaign=best_campaign,
        best_uplift=round(best_uplift, 6),
        recommendation=recommendation,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
def health():
    """Returns 200 OK if the API is running and models are loaded."""
    return {"status": "ok", "model": "T-Learner GBM", "version": "1.0.0"}


@app.get(
    "/baselines",
    response_model=BaselinesResponse,
    tags=["Experiment Results"],
    summary="Population-level A/B test results",
)
def baselines():
    """
    Returns the population-level visit and conversion rates from the original
    Hillstrom email experiment across all three arms.
    """
    return BaselinesResponse(
        control_visit_rate=round(base["ctrl_visit"], 4),
        mens_email_visit_rate=round(base["mens_visit"], 4),
        womens_email_visit_rate=round(base["wom_visit"], 4),
        mens_email_uplift_pp=round(base["mens_visit"] - base["ctrl_visit"], 4),
        womens_email_uplift_pp=round(base["wom_visit"] - base["ctrl_visit"], 4),
        control_conversion_rate=round(base["ctrl_conv"], 4),
        mens_conversion_rate=round(base["mens_conv"], 4),
        womens_conversion_rate=round(base["wom_conv"], 4),
        sample_size_total=64000,
        note=(
            "Mens email drives +72% relative uplift on visit rate. "
            "Both campaigns are statistically significant at p < 0.0001."
        ),
    )


@app.post(
    "/predict",
    response_model=UpliftPrediction,
    tags=["Predictions"],
    summary="Score a single customer",
)
def predict_single(customer: CustomerInput):
    """
    Predict the visit probability and email uplift for a **single customer**.

    Returns:
    - Visit probability under each campaign (no email / mens / womens)
    - Incremental uplift for each email vs no email baseline
    - Best campaign to send
    - Recommendation: **Send** (uplift ≥ 8pp) | **Borderline** (3–8pp) | **Skip** (< 3pp)
    """
    try:
        xs = engineer_features(customer)
        return score_features(xs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPrediction,
    tags=["Predictions"],
    summary="Score multiple customers at once",
)
def predict_batch(payload: BatchInput):
    """
    Score a **batch of customers** in a single request (max 10,000).

    Returns individual predictions for each customer plus aggregate summary counts
    (how many to Send, Borderline, or Skip).

    Useful for scoring a full customer list before an email campaign launch.
    """
    try:
        predictions = []
        for customer in payload.customers:
            xs   = engineer_features(customer)
            pred = score_features(xs)
            predictions.append(pred)

        send_count       = sum(1 for p in predictions if p.recommendation == "Send")
        borderline_count = sum(1 for p in predictions if p.recommendation == "Borderline")
        skip_count       = sum(1 for p in predictions if p.recommendation == "Skip")

        return BatchPrediction(
            total_customers=len(predictions),
            send_count=send_count,
            borderline_count=borderline_count,
            skip_count=skip_count,
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
