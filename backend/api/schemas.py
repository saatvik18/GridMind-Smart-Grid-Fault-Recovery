"""Pydantic request / response models."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FaultType = Literal["Normal", "Voltage Sag", "Overload", "Fault"]


class SimulateRequest(BaseModel):
    fault_type: FaultType = "Fault"
    section: Optional[int] = Field(None, ge=2, le=5, description="Affected section 2-5 (random if omitted)")
    seed: Optional[int] = None


class ClassifyRequest(BaseModel):
    fault_type: Optional[FaultType] = Field(None, description="Generate a new waveform; omit to classify the current one")
    seed: Optional[int] = None


class TrainCNNRequest(BaseModel):
    epochs: int = Field(30, ge=1, le=200)
    regenerate_dataset: bool = False


class TrainDQNRequest(BaseModel):
    episodes: int = Field(300, ge=10, le=3000)
    evaluate: bool = True


class EvaluateRequest(BaseModel):
    episodes: int = Field(100, ge=10, le=1000)


class LLMRequest(BaseModel):
    include_baseline: bool = False


class Classification(BaseModel):
    class_index: int
    fault_class: str
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    backend: str
    cnn_model: bool
    dqn_model: bool
    ollama: Dict[str, Any]


class SimulateResponse(BaseModel):
    fault_type: str
    section: Optional[int]
    waveform: Dict[str, List[float]]
    stats: Dict[str, Any]
    spectrogram_image: str
    classification: Classification
    feature_vector: List[float]
    network: Dict[str, Any]


class ClassifyResponse(BaseModel):
    true_class: str
    classification: Classification
    feature_vector: List[float]
    correct: bool


class RestoreResponse(BaseModel):
    agent: str
    scenario: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    summary: Dict[str, Any]
    initial_network: Dict[str, Any]
    log_path: Optional[str] = None


class LLMResponse(BaseModel):
    model: str
    report: Dict[str, Any]
    json_parsed: bool
    latency_s: float
    context: Dict[str, Any]
    report_path: Optional[str] = None


class GenericResponse(BaseModel):
    data: Dict[str, Any]
