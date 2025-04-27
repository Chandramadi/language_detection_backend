# Entry point
from fastapi import FastAPI
from app.api import predict, upload
from fastapi.middleware.cors import CORSMiddleware # Add CORS middleware

app = FastAPI(title="Indian Language Detection API")

# Add CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your React frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

app.include_router(predict.router)
app.include_router(upload.router)

@app.get("/", summary="Health check")
def root():
    return {"status": "OK", "message": "Language Detection API is running"}
