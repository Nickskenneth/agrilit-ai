from pydantic import BaseModel
from typing import Optional


class PredictionResult(BaseModel):
    """Response sukses dari endpoint /predict"""
    success: bool
    filename: str
    prediction: str                # Label kelas: "Cabai_BacterialSpot" | "Tidak_Teridentifikasi"
    prediction_id: str             # Label Indonesia: "Bercak Bakteri pada Cabai"
    commodity: str                 # Komoditas: "cabai" | "kentang" | "jagung"
    confidence: float              # 0-100, contoh: 92.34
    confidence_normalized: float   # 0-1, contoh: 0.9234 (untuk disimpan di Laravel)
    low_confidence: bool = False   # True jika raw confidence < RAW_CONFIDENCE_THRESHOLD


class ErrorResponse(BaseModel):
    """Response error"""
    success: bool = False
    message: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Response health check"""
    status: str
    model_loaded: bool
    model_path: str
    classes: int