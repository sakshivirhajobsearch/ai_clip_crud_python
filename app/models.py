from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base

class ImageTextPair(Base):
    __tablename__ = "image_text_pairs"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True, nullable=False)
    image_path = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # store embedding bytes
