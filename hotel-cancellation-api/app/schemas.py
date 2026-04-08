from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BookingFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hotel": "City Hotel",
                "lead_time": 45,
                "arrival_date_month": "August",
                "meal": "BB",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "reserved_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 120.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 1,
                "previous_cancellations": 0,
            }
        }
    )

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0, le=1000)
    arrival_date_month: Literal[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    meal: Literal["BB", "FB", "HB", "SC", "Undefined"]
    market_segment: str = Field(min_length=1, max_length=50)
    distribution_channel: str = Field(min_length=1, max_length=50)
    reserved_room_type: str = Field(min_length=1, max_length=5)
    booking_changes: int = Field(ge=0, le=50)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    days_in_waiting_list: int = Field(ge=0, le=500)
    customer_type: Literal["Transient", "Contract", "Transient-Party", "Group"]
    adr: float = Field(ge=0.0, le=5000.0)
    required_car_parking_spaces: int = Field(ge=0, le=8)
    total_of_special_requests: int = Field(ge=0, le=5)
    previous_cancellations: int = Field(ge=0, le=10)


class PredictionResponse(BaseModel):
    is_cancelled: bool
    cancellation_probability: float = Field(ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    total_predictions: int
    cancellations_predicted: int
    non_cancellations_predicted: int
    uptime_seconds: float