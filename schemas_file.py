from pydantic import BaseModel, ConfigDict

class FileResponse(BaseModel):
    id: int
    filename: str
    status: str
    ingested_chunks: int | None

    model_config = ConfigDict(from_attributes=True)
