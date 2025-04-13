from fastapi import APIRouter, HTTPException, status
from app.models.schemas import TextInput, BatchInput
from app.services import detector

router = APIRouter()

@router.post("/predict", summary="Predict language of a single sentence", status_code=status.HTTP_200_OK)
async def predict(input: TextInput):
    try:
        return detector.predict_single(input.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_batch", summary="Predict languages for a list of sentences", status_code=status.HTTP_200_OK)
async def predict_batch(batch: BatchInput):
    try:
        return {"results": detector.predict_batch(batch.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
