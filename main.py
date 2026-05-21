import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.classifier import classifier, validate_upload, COMMODITY_CLASS_INDICES
from app.config import HOST, PORT
from app.schemas import ErrorResponse, HealthResponse, PredictionResult

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# LIFESPAN — Load model sekali saat startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgriLit AI Service sedang startup...")
    classifier.load_model()
    logger.info("AgriLit AI Service siap menerima request.")
    yield
    logger.info("AgriLit AI Service shutdown.")


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="AgriLit AI Service",
    description="Microservice identifikasi penyakit tanaman (Cabai, Kentang, Jagung)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok" if classifier.is_loaded else "model_not_loaded",
        model_loaded=classifier.is_loaded,
        model_path="model/model_terbaik.keras",
        classes=9,
    )


# ============================================================
# PREDICT — Dengan commodity filter
# POST /predict/{jenis_tanaman}
# Contoh: POST /predict/cabai, POST /predict/kentang
# ============================================================
@app.post(
    "/predict/{jenis_tanaman}",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
    summary="Prediksi penyakit dengan filter komoditas",
)
async def predict_with_commodity(
    jenis_tanaman: str = Path(
        ...,
        description="Jenis tanaman: cabai, kentang, atau jagung",
        examples=["cabai", "kentang", "jagung"],
    ),
    file: UploadFile = File(
        ...,
        description="Gambar daun tanaman (JPG/PNG, maks 5MB)",
    ),
):
    """
    Endpoint utama inference DENGAN filter komoditas.

    Model memprediksi semua 9 kelas, lalu post-processing
    hanya mempertimbangkan kelas milik komoditas yang dipilih.
    Ini mencegah misklasifikasi lintas komoditas.
    """
    # Validasi jenis tanaman
    jenis = jenis_tanaman.lower().strip()
    if jenis not in COMMODITY_CLASS_INDICES:
        raise HTTPException(
            status_code=400,
            detail=f"Jenis tanaman '{jenis_tanaman}' tidak didukung. "
                   f"Gunakan: cabai, kentang, atau jagung.",
        )

    image_bytes = await file.read()

    try:
        validate_upload(
            content_type=file.content_type or "",
            file_size=len(image_bytes),
            filename=file.filename or "unknown",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = classifier.predict(
            image_bytes=image_bytes,
            filename=file.filename or "unknown",
            jenis_tanaman=jenis,  # ← Parameter baru
        )
        return PredictionResult(**result)

    except ValueError as e:
        logger.warning(f"Gambar tidak valid: {file.filename} — {e}")
        raise HTTPException(status_code=400, detail=f"Gambar tidak bisa diproses: {e}")

    except Exception as e:
        logger.error(f"Error inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal.")


# ============================================================
# PREDICT — Tanpa commodity filter (fallback/global)
# POST /predict
# Memprediksi dari semua 9 kelas tanpa filtering
# ============================================================
@app.post(
    "/predict",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
    summary="Prediksi penyakit tanpa filter komoditas (global)",
)
async def predict_global(
    file: UploadFile = File(
        ...,
        description="Gambar daun tanaman (JPG/PNG, maks 5MB)",
    ),
):
    """
    Endpoint inference TANPA filter komoditas.
    Prediksi dari semua 9 kelas secara global.
    Gunakan endpoint /predict/{jenis_tanaman} untuk hasil lebih akurat.
    """
    image_bytes = await file.read()

    try:
        validate_upload(
            content_type=file.content_type or "",
            file_size=len(image_bytes),
            filename=file.filename or "unknown",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = classifier.predict(
            image_bytes=image_bytes,
            filename=file.filename or "unknown",
            jenis_tanaman=None,  # ← Tanpa filter
        )
        return PredictionResult(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Gambar tidak bisa diproses: {e}")

    except Exception as e:
        logger.error(f"Error inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal.")


# ============================================================
# EXCEPTION HANDLERS
# ============================================================
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "message": "Endpoint tidak ditemukan."},
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request, exc):
    return JSONResponse(
        status_code=405,
        content={"success": False, "message": "Method tidak diizinkan."},
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)