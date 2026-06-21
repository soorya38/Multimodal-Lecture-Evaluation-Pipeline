from pydantic import BaseModel, Field


class EvaluateResponse(BaseModel):
    """
    Final evaluation response for a lecture video.

    Contains all four scores produced by the end-to-end evaluation pipeline:
    technical accuracy, grammar quality, and language distribution percentages.
    """

    technical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Technical accuracy score (0–100) based on subject correctness.",
    )
    grammatical_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Grammar quality score (0–100) based on transcript analysis.",
    )
    english_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of the lecture delivered in English (0–100).",
    )
    tamil_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of the lecture delivered in Tamil (0–100).",
    )
