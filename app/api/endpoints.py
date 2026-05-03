from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import time
import logging
from app.models.model import ObjectDetectionModel
import threading
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Computer Vision MLOps API",
    description="API for object detection using Faster R-CNN",
    version="1.0.0"
)

_model: Optional[ObjectDetectionModel] = None
_model_lock = threading.Lock()


def get_model() -> ObjectDetectionModel:
    """
    Lazily initialize the model when `/predict` is called.
    This avoids heavy ML imports during app startup and test collection.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ObjectDetectionModel()
    return _model

@app.get("/")
def read_root():
    return {"message": "Welcome to the Computer Vision MLOps API"}

@app.get("/health")
def health_check():
    """Endpoint for health checks"""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in an image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with detection results
    """
    logger.info(f"Received prediction request for file: {file.filename}")
    start_time = time.time()
    
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name
        
        # Make prediction
        result = get_model().predict(temp_path)
        
        # Add latency information
        result['api_latency'] = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        logger.info(f"Successfully processed file: {file.filename}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")