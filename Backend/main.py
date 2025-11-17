from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
import io
import os
from typing import Dict
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

app = FastAPI()

# Allowing frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "hackstate_123.@"),
    "database": os.getenv("DB_NAME", "indian_livestock"),
    "pool_name": "mypool",
    "pool_size": 5
}

# Creating connection pool
try:
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
    print("Database connection pool created successfully")
except Error as e:
    print(f"Error creating connection pool: {e}")
    connection_pool = None

# Context manager for database connections
@contextmanager
def get_db_connection():
    connection = None
    try:
        connection = connection_pool.get_connection()
        yield connection
    except Error as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if connection and connection.is_connected():
            connection.close()

# Loading Model
try:
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 44)  
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "Hack_ResNet50_model.pth")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class Mapping
CLASS_TO_BREED_ID = {
    0: 11, 1: 12, 2: 13, 3: 1, 4: 14, 5: 2, 6: 15, 7: 16, 8: 17, 9: 18,
    10: 19, 11: 20, 12: 21, 13: 3, 14: 22, 15: 23, 16: 24, 17: 25, 18: 26,
    19: 27, 20: 28, 21: 29, 22: 30, 23: 31, 24: 4, 25: 5, 26: 32, 27: 6,
    28: 7, 29: 33, 30: 34, 31: 8, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39,
    37: 40, 38: 41, 39: 9, 40: 42, 41: 10, 42: 43, 43: 44
}

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------- Prediction Function -----------
def predict_image(image_bytes: bytes) -> dict:
    """Predict breed from image bytes"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = transform(image).unsqueeze(0)
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(img_t)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            _, pred = torch.max(outputs, 1)
        
        confidence = float(probabilities[0][pred.item()])
        
        # Handle unknown predictions based on confidence threshold
        CONFIDENCE_THRESHOLD = 0.75
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "predicted_index": None,
                "confidence": confidence,
                "is_unknown": True,
                "message": "Unknown / Not a cattle breed",
                "top5_predictions": [
                    {
                        "index": int(idx),
                        "confidence": float(prob)
                    }
                    for idx, prob in zip(top5_indices[0], top5_prob[0])
                ]
            }

        # Return valid prediction
        return {
            "predicted_index": pred.item(),
            "confidence": confidence,
            "is_unknown": False,
            "top5_predictions": [
                {
                    "index": int(idx),
                    "confidence": float(prob)
                }
                for idx, prob in zip(top5_indices[0], top5_prob[0])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

# ----------- Helper Functions -----------
def get_complete_breed_info(breed_id: int, cursor):
    """Get complete breed information from all related tables"""
    cursor.execute("SELECT * FROM breeds WHERE breed_id = %s", (breed_id,))
    breed_info = cursor.fetchone()
    if not breed_info:
        return None
    
    cursor.execute("SELECT * FROM economic_usage WHERE breed_id = %s", (breed_id,))
    economic_info = cursor.fetchone()
    
    cursor.execute("SELECT * FROM physical_characteristics WHERE breed_id = %s", (breed_id,))
    physical_info = cursor.fetchone()
    
    cursor.execute("SELECT * FROM production_data WHERE breed_id = %s", (breed_id,))
    production_info = cursor.fetchone()
    
    cursor.execute("SELECT * FROM reproduction_health WHERE breed_id = %s", (breed_id,))
    reproduction_info = cursor.fetchone()
    
    complete_info = {
        "breed": breed_info,
        "economic_usage": economic_info,
        "physical_characteristics": physical_info,
        "production_data": production_info,
        "reproduction_health": reproduction_info
    }
    return complete_info

# ----------- API Endpoints -----------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Indian Livestock Breed Recognition API",
        "database": "indian_livestock",
        "tables": ["breeds", "economic_usage", "physical_characteristics", "production_data", "reproduction_health"]
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> Dict:
    """Predict cattle/buffalo breed from uploaded image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        prediction_result = predict_image(image_bytes)

        # Handle unknown predictions gracefully
        if prediction_result.get("is_unknown"):
            return {
                "success": True,
                "prediction": "Unknown / Not a cattle breed",
                "confidence": prediction_result["confidence"],
                "message": "The model could not confidently identify this as a cattle breed.",
                "top5_predictions": prediction_result["top5_predictions"]
            }

        breed_index = prediction_result["predicted_index"]
        breed_id = CLASS_TO_BREED_ID.get(breed_index)

        if breed_id is None:
            raise HTTPException(status_code=500, detail=f"Invalid breed index {breed_index}")

        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            complete_info = get_complete_breed_info(breed_id, cursor)
            cursor.close()

        if not complete_info or not complete_info["breed"]:
            raise HTTPException(status_code=404, detail="Breed information not found")

        return {
            "success": True,
            "predicted_breed_index": breed_index,
            "breed_id": breed_id,
            "confidence": prediction_result["confidence"],
            "top5_predictions": prediction_result["top5_predictions"],
            "data": complete_info
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ----------- Additional Endpoints -----------
@app.get("/breeds/")
async def get_all_breeds():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT breed_id, breed_name, animal_type, origin_country, origin_state, origin_region 
                FROM breeds ORDER BY breed_name
            """)
            breeds = cursor.fetchall()
            cursor.close()
        return {"success": True, "count": len(breeds), "breeds": breeds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch breeds: {str(e)}")

@app.get("/breeds/{breed_id}")
async def get_breed(breed_id: int):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            complete_info = get_complete_breed_info(breed_id, cursor)
            cursor.close()
        if not complete_info or not complete_info["breed"]:
            raise HTTPException(status_code=404, detail="Breed not found")
        return {"success": True, "breed_id": breed_id, "data": complete_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch breed: {str(e)}")

@app.get("/stats/")
async def get_stats():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT animal_type, COUNT(*) as count FROM breeds GROUP BY animal_type")
            type_stats = cursor.fetchall()
            cursor.execute("SELECT COUNT(*) as total FROM breeds")
            total = cursor.fetchone()
            cursor.close()
        return {"success": True, "total_breeds": total["total"], "by_type": type_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    if connection_pool:
        connection_pool.close()
        print("Database connection pool closed")