from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    type: str = Field(..., pattern="^[HML]$")
    air_temperature: float = Field(..., gt=0)
    process_temperature: float = Field(..., gt=0)
    rotational_speed: float = Field(..., gt=0)
    torque: float = Field(..., gt=0)
    tool_wear: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    failure_probability: float
    confidence: float


class ExplanationItem(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ExplanationResponse(BaseModel):
    prediction: int
    result: str
    failure_probability: float
    explanations: list[ExplanationItem]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class SweepRequest(BaseModel):
    input: PredictionRequest
    feature: str
    steps: int = Field(40, ge=5, le=200)


class SweepPoint(BaseModel):
    value: float
    failure_probability: float


class SweepResponse(BaseModel):
    feature: str
    points: list[SweepPoint]
