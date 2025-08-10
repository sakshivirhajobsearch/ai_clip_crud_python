import numpy as np
from typing import List, Optional
from sqlalchemy.orm import Session
from app import models

def create_image_text_pair(
    db: Session, text: str, image_path: str, embedding: np.ndarray
) -> models.ImageTextPair:
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu().numpy()
    emb_bytes = embedding.tobytes()
    db_item = models.ImageTextPair(text=text, image_path=image_path, embedding=emb_bytes)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_all_image_text_pairs(db: Session) -> List[models.ImageTextPair]:
    return db.query(models.ImageTextPair).all()

def get_image_text_pair(db: Session, item_id: int) -> Optional[models.ImageTextPair]:
    return db.query(models.ImageTextPair).filter(models.ImageTextPair.id == item_id).first()

def update_image_text_pair(db: Session, item_id: int, text: str) -> Optional[models.ImageTextPair]:
    item = get_image_text_pair(db, item_id)
    if not item:
        return None
    item.text = text
    db.commit()
    db.refresh(item)
    return item

def delete_image_text_pair(db: Session, item_id: int) -> bool:
    item = get_image_text_pair(db, item_id)
    if not item:
        return False
    db.delete(item)
    db.commit()
    return True

def search_by_text(db: Session, query: str) -> List[models.ImageTextPair]:
    return db.query(models.ImageTextPair).filter(models.ImageTextPair.text.ilike(f"%{query}%")).all()

def embedding_from_bytes(emb_bytes: bytes, dtype=np.float32, embedding_dim=512) -> np.ndarray:
    return np.frombuffer(emb_bytes, dtype=dtype).reshape(embedding_dim)

def search_by_embedding(db: Session, clip, query: str, top_k=5) -> List[models.ImageTextPair]:
    items = get_all_image_text_pairs(db)
    query_emb = clip.encode_text([query])[0]  # (embedding_dim,)
    sims = []
    for item in items:
        item_emb = embedding_from_bytes(item.embedding)
        sim = np.dot(query_emb, item_emb)  # cosine similarity (assuming normalized)
        sims.append((item, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in sims[:top_k]]
