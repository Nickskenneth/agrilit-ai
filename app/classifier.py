import io
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.config import MODEL_PATH, ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)

# ============================================================
# CLASS MAPPING — Urutan absolut sesuai folder training (A-Z)
# ============================================================
CLASS_MAPPING: dict[int, str] = {
    0: "Cabai_BacterialSpot",
    1: "Cabai_Healthy",
    2: "Jagung_Blight",
    3: "Jagung_CommonRust",
    4: "Jagung_GrayLeafSpot",
    5: "Jagung_Healthy",
    6: "Kentang_EarlyBlight",
    7: "Kentang_Healthy",
    8: "Kentang_LateBlight",
}

LABEL_ID_MAPPING: dict[str, str] = {
    "Cabai_BacterialSpot": "Bercak Bakteri pada Cabai",
    "Cabai_Healthy":       "Cabai Sehat",
    "Jagung_Blight":       "Hawar Daun Jagung",
    "Jagung_CommonRust":   "Karat Biasa pada Jagung",
    "Jagung_GrayLeafSpot": "Bercak Daun Abu-abu Jagung",
    "Jagung_Healthy":      "Jagung Sehat",
    "Kentang_EarlyBlight": "Bercak Awal pada Kentang",
    "Kentang_Healthy":     "Kentang Sehat",
    "Kentang_LateBlight":  "Busuk Daun (Late Blight) Kentang",
}

COMMODITY_MAPPING: dict[str, str] = {
    "Cabai_BacterialSpot": "cabai",
    "Cabai_Healthy":       "cabai",
    "Jagung_Blight":       "jagung",
    "Jagung_CommonRust":   "jagung",
    "Jagung_GrayLeafSpot": "jagung",
    "Jagung_Healthy":      "jagung",
    "Kentang_EarlyBlight": "kentang",
    "Kentang_Healthy":     "kentang",
    "Kentang_LateBlight":  "kentang",
}

# ============================================================
# COMMODITY FILTER — Index kelas yang termasuk tiap komoditas
# Ini adalah inti perbaikan bug misklasifikasi lintas komoditas
# ============================================================
COMMODITY_CLASS_INDICES: dict[str, list[int]] = {
    "cabai":   [0, 1],        # Cabai_BacterialSpot, Cabai_Healthy
    "jagung":  [2, 3, 4, 5],  # Jagung_Blight, CommonRust, GrayLeafSpot, Healthy
    "kentang": [6, 7, 8],     # Kentang_EarlyBlight, Healthy, LateBlight
}


class DiseaseClassifier:
    """
    Singleton class untuk model inference.
    Model dimuat SEKALI saat startup aplikasi.
    """

    _instance: "DiseaseClassifier | None" = None
    _model: tf.keras.Model | None = None

    def __new__(cls) -> "DiseaseClassifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        if self._model is not None:
            logger.info("Model sudah dimuat, skip reload.")
            return

        logger.info(f"Memuat model dari: {MODEL_PATH}")
        try:
            self._model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(
                f"Model berhasil dimuat. "
                f"Input shape: {self._model.input_shape}, "
                f"Output classes: {self._model.output_shape[-1]}"
            )
        except Exception as e:
            logger.error(f"FATAL: Gagal memuat model — {e}")
            raise RuntimeError(f"Gagal memuat model: {e}") from e

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Pipeline preprocessing sesuai spesifikasi handover:
        1. Bytes → PIL Image
        2. Convert ke RGB
        3. Resize ke 224x224
        4. Numpy array
        5. Expand dims → (1, 224, 224, 3)
        6. preprocess_input (BUKAN /255.0)
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            raise ValueError(f"Gagal memproses gambar: {e}") from e

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        jenis_tanaman: str | None = None,
    ) -> dict:
        """
        Jalankan inference dengan commodity-scoped filtering.

        Jika jenis_tanaman diberikan (cabai/kentang/jagung):
          → Model tetap memprediksi semua 9 kelas
          → Post-processing: hanya kelas milik komoditas tersebut yang dipertimbangkan
          → Argmax diambil dari subset kelas komoditas saja

        Jika jenis_tanaman tidak diberikan atau tidak valid:
          → Prediksi global dari semua 9 kelas (behavior lama)
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load_model() dulu.")

        img_array = self._preprocess_image(image_bytes)

        # Inference — model selalu memprediksi semua 9 kelas
        raw_predictions = self._model.predict(img_array, verbose=0)
        all_probs = raw_predictions[0]  # shape: (9,)

        # ============================================================
        # POST-PROCESSING: Commodity-scoped filtering
        # ============================================================
        valid_indices = None
        if jenis_tanaman and jenis_tanaman.lower() in COMMODITY_CLASS_INDICES:
            valid_indices = COMMODITY_CLASS_INDICES[jenis_tanaman.lower()]

        if valid_indices is not None:
            # Buat masked array — set probabilitas kelas di luar komoditas ke -1
            # sehingga argmax hanya bisa memilih kelas dalam komoditas yang benar
            masked_probs = np.full_like(all_probs, -1.0)
            for idx in valid_indices:
                masked_probs[idx] = all_probs[idx]

            predicted_index = int(np.argmax(masked_probs))
            # Confidence dihitung dari probabilitas asli (bukan masked)
            # tapi hanya di-normalize terhadap total probabilitas komoditas ini
            commodity_total = sum(all_probs[idx] for idx in valid_indices)
            if commodity_total > 0:
                confidence_raw = float(all_probs[predicted_index] / commodity_total * 100)
            else:
                confidence_raw = float(all_probs[predicted_index] * 100)

            logger.info(
                f"Commodity filter: {jenis_tanaman} → "
                f"valid indices: {valid_indices}, "
                f"raw probs: {[f'{all_probs[i]:.4f}' for i in valid_indices]}, "
                f"selected: {predicted_index} ({CLASS_MAPPING[predicted_index]})"
            )
        else:
            # Tanpa filter — prediksi global (semua 9 kelas)
            predicted_index = int(np.argmax(all_probs))
            confidence_raw  = float(np.max(all_probs) * 100)

            logger.info(
                f"Global prediction (no commodity filter) → "
                f"selected: {predicted_index} ({CLASS_MAPPING[predicted_index]})"
            )

        prediction_label    = CLASS_MAPPING[predicted_index]
        prediction_label_id = LABEL_ID_MAPPING[prediction_label]
        commodity           = COMMODITY_MAPPING[prediction_label]

        # Log semua probabilitas untuk debugging
        logger.info("Full prediction breakdown:")
        for i, prob in enumerate(all_probs):
            marker = " ← SELECTED" if i == predicted_index else ""
            logger.info(
                f"  [{i}] {CLASS_MAPPING[i]:30s} → {prob*100:6.2f}%{marker}"
            )

        return {
            "success":               True,
            "filename":              filename,
            "prediction":            prediction_label,
            "prediction_id":         prediction_label_id,
            "commodity":             commodity,
            "confidence":            round(confidence_raw, 2),
            "confidence_normalized": round(confidence_raw / 100, 4),
        }


def validate_upload(
    content_type: str,
    file_size: int,
    filename: str,
) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Tipe file tidak didukung: '{content_type}'. "
            f"Hanya menerima: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Ukuran file terlalu besar: {file_size / 1024 / 1024:.1f}MB. "
            f"Maksimal 5MB."
        )


# Instance global
classifier = DiseaseClassifier()