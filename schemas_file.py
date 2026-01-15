# schemas_file.py
from pydantic import BaseModel
from typing import Optional

class FileResponse(BaseModel):
    id: int
    filename: str
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

