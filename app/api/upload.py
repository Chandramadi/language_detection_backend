from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.extractor import extract_and_chunk
from app.services.detector import predict_batch
from app.models.schemas import BatchInput
import shutil, os
from collections import Counter

router = APIRouter()

@router.post("/upload_and_detect", summary="Upload a PDF and detect languages", status_code=status.HTTP_200_OK)
async def upload_and_detect(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = extract_and_chunk(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        response = predict_batch(chunks)
        languages = [entry["language"] for entry in response]
        counts = Counter(languages)
        total = len(languages)
        percentages = {lang: round((count / total) * 100, 2) for lang, count in counts.items()}

        return percentages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
