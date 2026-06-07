import os
import sys
import glob
import uuid
import shutil
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

# --- Ensure we can import from the root repository ---
# This appends the root folder (signature-verification) to Python's path
# so that it can find `inference.py` and other core model files regardless
# of where this server script is executed from.
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(API_DIR, '..', '..'))
sys.path.append(ROOT_DIR)

# Now we can safely import from the root's inference module without changing its process
from inference import load_model, predict_baseline, predict_siamese, build_explanation, DEFAULT_BASELINE_THRESHOLD, DEFAULT_SIAMESE_THRESHOLD

# --- Directories specific to the API ---
# We keep app-specific files strictly inside the app/api directory
TEMP_UPLOADS_DIR = os.path.join(API_DIR, "temp_uploads")
EXPLANATIONS_DIR = os.path.join(API_DIR, "explanations")

# Initialize the FastAPI application
app = FastAPI(title="Signature Verification API", description="API for signature verification model.")

# --- CORS Middleware ---
# This is crucial for web applications. It allows a React frontend (which runs on a different 
# port like 3000) to make HTTP requests to this backend (running on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving ---
# We create an explanations folder inside app/api and "mount" it. 
# This means any file saved here can be directly accessed via a URL.
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)
app.mount("/explanations", StaticFiles(directory=EXPLANATIONS_DIR), name="explanations")

# Global variables to hold the loaded models so they are only loaded once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
baseline_model = None
siamese_model = None

@app.on_event("startup")
async def startup_event():
    """
    Runs automatically when the API starts. 
    Loads the PyTorch models into memory so that subsequent API requests are very fast.
    """
    global baseline_model, siamese_model
    print(f"Starting API Server. Loading models on {device}...")
    baseline_model = load_model("baseline", device)
    siamese_model = load_model("siamese", device)
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    print("API is ready to receive requests.")

def cleanup_temp_uploads():
    """
    Background task to keep the TEMP_UPLOADS_DIR clean.
    It sorts files by age and deletes all but the 2 most recent files,
    so we don't consume unnecessary disk space over time.
    """
    files = glob.glob(os.path.join(TEMP_UPLOADS_DIR, "*"))
    files.sort(key=os.path.getmtime)  # Sort by modification time (oldest first)
    
    if len(files) > 2:
        for file_to_delete in files[:-2]:
            try:
                os.remove(file_to_delete)
            except Exception as e:
                print(f"Error deleting temp file {file_to_delete}: {e}")

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Helper function to save an uploaded image to disk temporarily.
    Generates a unique UUID filename to prevent accidental overwrites if multiple users upload at once.
    """
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{upload_file.filename}"
    file_path = os.path.join(TEMP_UPLOADS_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

@app.post("/verify/baseline")
async def verify_baseline(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    threshold: Optional[float] = Form(DEFAULT_BASELINE_THRESHOLD),
    explain: Optional[bool] = Form(False)
):
    """
    Endpoint for single-image (baseline) signature verification.
    """
    image_path = await save_upload_file(image)
    
    # Run the core inference logic without modifying how it works
    prob, prediction = predict_baseline(baseline_model, image_path, threshold, device)
    
    response = {
        "probability": prob,
        "prediction": prediction,
        "threshold": threshold,
        "explanations": None
    }

    if explain:
        unique_id = str(uuid.uuid4())
        # Generate heatmaps using the core logic, passing our API-specific explanation folder
        outputs = build_explanation(
            'baseline', 
            baseline_model, 
            image_path, 
            output_dir=EXPLANATIONS_DIR,
            device=device,
            file_prefix=unique_id
        )
        # Format the absolute paths into relative URL paths for the frontend
        response["explanations"] = {
            "overlay": f"/explanations/{os.path.basename(outputs['overlay'])}",
            "heatmap": f"/explanations/{os.path.basename(outputs['heatmap'])}"
        }

    # Trigger the cleanup script *after* sending the HTTP response
    background_tasks.add_task(cleanup_temp_uploads)
    return response

@app.post("/verify/siamese")
async def verify_siamese(
    background_tasks: BackgroundTasks,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    threshold: Optional[float] = Form(DEFAULT_SIAMESE_THRESHOLD),
    explain: Optional[bool] = Form(False)
):
    """
    Endpoint for pairwise (siamese) signature verification.
    Compares image_a against image_b.
    """
    image_a_path = await save_upload_file(image_a)
    image_b_path = await save_upload_file(image_b)
    
    prob, prediction = predict_siamese(siamese_model, image_a_path, image_b_path, threshold, device)
    
    response = {
        "probability": prob,
        "prediction": prediction,
        "threshold": threshold,
        "explanations": None
    }

    if explain:
        unique_id = str(uuid.uuid4())
        outputs = build_explanation(
            'siamese', 
            siamese_model, 
            None, 
            image_a_path, 
            image_b_path,
            output_dir=EXPLANATIONS_DIR,
            device=device,
            file_prefix=unique_id
        )
        response["explanations"] = {
            "overlay_a": f"/explanations/{os.path.basename(outputs['overlay_a'])}",
            "overlay_b": f"/explanations/{os.path.basename(outputs['overlay_b'])}",
            "heatmap_a": f"/explanations/{os.path.basename(outputs['heatmap_a'])}",
            "heatmap_b": f"/explanations/{os.path.basename(outputs['heatmap_b'])}"
        }

    background_tasks.add_task(cleanup_temp_uploads)
    return response

if __name__ == "__main__":
    # This block allows you to start the API simply by running:
    # python server.py
    # This is much easier than running the full uvicorn command!
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
