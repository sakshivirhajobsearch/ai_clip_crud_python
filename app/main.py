import uuid
import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image

from app import models, schemas, crud, database, clip_model

app = FastAPI()

# Create tables
models.Base.metadata.create_all(bind=database.engine)

# Initialize CLIP wrapper
clip = clip_model.CLIPWrapper()

# CORS Middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "AI CLIP CRUD app is running"}

@app.post("/items/", response_model=schemas.ImageText)
async def create_item(
    text: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex}_{image.filename}"
    image_path = os.path.join(images_dir, unique_filename)

    # Save uploaded image
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    await image.close()

    # Validate and convert image
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception:
        # Cleanup invalid image file
        os.remove(image_path)
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Encode image embedding
    image_emb = clip.encode_image([pil_image])

    # Create DB entry
    db_item = crud.create_image_text_pair(db, text, image_path, image_emb)
    return db_item

@app.get("/items/", response_model=List[schemas.ImageText])
def read_items(db: Session = Depends(get_db)):
    return crud.get_all_image_text_pairs(db)

@app.get("/items/{item_id}", response_model=schemas.ImageText)
def read_item(item_id: int, db: Session = Depends(get_db)):
    db_item = crud.get_image_text_pair(db, item_id)
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.put("/items/{item_id}", response_model=schemas.ImageText)
def update_item(item_id: int, text: str = Form(...), db: Session = Depends(get_db)):
    db_item = crud.update_image_text_pair(db, item_id, text)
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    success = crud.delete_image_text_pair(db, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True}

@app.get("/search/", response_model=List[schemas.ImageText])
def search_items(query: str, db: Session = Depends(get_db)):
    results = crud.search_by_text(db, query)
    return results

@app.get("/images/{image_name}")
def get_image(image_name: str):
    image_path = os.path.join("images", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")
