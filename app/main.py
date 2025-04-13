# Entry point
from fastapi import FastAPI
from app.api import predict, upload

app = FastAPI(title="Indian Language Detection API")
app.include_router(predict.router)
app.include_router(upload.router)

@app.get("/", summary="Health check")
def root():
    return {"status": "OK", "message": "Language Detection API is running"}
