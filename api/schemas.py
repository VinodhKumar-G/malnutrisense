"""
api/schemas.py — Pydantic request/response schemas for the MalnutriSense API.

ChildFeatures: input schema — all features needed for prediction.
PredictionResponse: output schema — risk scores + SHAP + equity flag.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChildFeatures(BaseModel):
	"""Input features for a single child. All fields match NFHS-5 column encodings."""

	age_months: int = Field(..., ge=0, le=59, description="Child age in months")
	sex: str = Field(..., description="male or female")
	wealth_quintile: int = Field(..., ge=1, le=5, description="1=Poorest, 5=Richest")
	mother_education: str = Field(..., description="no_education/primary/secondary/higher")
	water_source: str = Field(..., description="piped_on_premises/tube_well/etc")
	toilet_type: str = Field(..., description="flush_piped/pit_with_slab/etc")
	diarrhoea_2weeks: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
	birth_weight_g: Optional[float] = Field(None, description="Birth weight in grams")
	breastfeed_months: Optional[float] = Field(None, description="Duration of breastfeeding")
	residence: str = Field("rural", description="urban or rural")
	state_code: Optional[int] = Field(None, description="NFHS V024 state code")


class PhenotypeRisk(BaseModel):
	probability: float
	prediction: int  # 0 = healthy, 1 = at risk
	threshold: float  # decision threshold applied


class SHAPFeature(BaseModel):
	feature: str
	shap_value: float  # positive = pushes toward malnourished
	value: str  # human-readable feature value


class PredictionResponse(BaseModel):
	model_config = ConfigDict(protected_namespaces=())

	stunted: PhenotypeRisk
	underweight: PhenotypeRisk
	wasted: PhenotypeRisk
	overall_risk: str  # 'high', 'medium', 'low'
	top_shap_features: list[SHAPFeature]  # top 3 features driving highest risk
	equity_flag: bool  # True if child is in high-FNR demographic group
	equity_reason: str  # why equity flag was raised
	model_version: str
