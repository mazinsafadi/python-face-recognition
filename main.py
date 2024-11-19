from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import os
import numpy as np
from typing import Dict
from deepface import DeepFace
import logging
from datetime import datetime
from scipy.spatial.distance import cosine
import psutil
import tensorflow as tf
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TensorFlow Configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure GPU is disabled

# Set model for consistent embedding generation
FACE_RECOGNITION_MODEL = "VGG-Face"  # Change to "Facenet512" for smaller embeddings

app = FastAPI()

# Configure CORS
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File for storing user data
USERS_FILE = os.getenv("USERS_FILE", "./users.json")

# Load users from file
def load_users() -> Dict:
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

# Save users to file
def save_users(users: Dict):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        raise

# Extract face embedding
def get_face_embedding(image_array):
    temp_img_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            cv2.imwrite(temp_img.name, image_array)
            temp_img_path = temp_img.name

        embedding = DeepFace.represent(
            img_path=temp_img_path,
            model_name=FACE_RECOGNITION_MODEL,  # Use the global model
            enforce_detection=True,
            detector_backend="opencv"
        )
        if not embedding:
            logger.warning("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in the image")
        return embedding[0]["embedding"]
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process face in image")
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)

# Compare face embeddings
def compare_faces(embedding1, embedding2, threshold=0.4):
    try:
        if len(embedding1) != len(embedding2):
            raise ValueError(f"Embedding size mismatch: {len(embedding1)} != {len(embedding2)}")
        distance = cosine(embedding1, embedding2)
        logger.info(f"Face comparison distance: {distance} (threshold: {threshold})")
        return distance < threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        raise

@app.post("/api/signup")
async def signup(photo: UploadFile = File(...), name: str = Form(...)):
    logger.info(f"Signup attempt for user: {name}")
    try:
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_embedding = get_face_embedding(image)

        users = load_users()

        for existing_user in users.values():
            if compare_faces(existing_user["embedding"], face_embedding):
                logger.warning(f"Face already registered under name: {existing_user['name']}")
                raise HTTPException(status_code=400, detail="Face already registered")

        users[name] = {
            "name": name,
            "embedding": face_embedding
        }
        save_users(users)
        logger.info(f"Successfully registered new user: {name}")

        return {"message": f"Successfully registered {name}", "name": name}

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signin")
async def signin(photo: UploadFile = File(...)):
    logger.info("Signin attempt")
    try:
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_embedding = get_face_embedding(image)

        users = load_users()

        for user in users.values():
            try:
                if compare_faces(user["embedding"], face_embedding):
                    logger.info(f"Successful login for user: {user['name']}")
                    return {"message": f"Welcome back, {user['name']}", "name": user["name"]}
            except ValueError as ve:
                logger.warning(f"Embedding size mismatch for user {user['name']}: {ve}")

        logger.warning("Face not recognized")
        raise HTTPException(status_code=401, detail="Face not recognized")

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Error during signin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users():
    try:
        users = load_users()
        user_list = list(users.keys())
        return {"users": user_list}
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed.")
    return {"message": "API is running"}

@app.head("/")
async def root_head():
    logger.info("Health check HEAD request accessed.")
    return {"message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
