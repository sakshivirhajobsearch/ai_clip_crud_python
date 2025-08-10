from pydantic import BaseModel

class ImageTextBase(BaseModel):
    text: str

class ImageTextCreate(ImageTextBase):
    pass

class ImageTextUpdate(ImageTextBase):
    pass

class ImageText(ImageTextBase):
    id: int
    image_path: str

    class Config:
        from_attributes = True  # Pydantic v2 replacement for orm_mode = True
