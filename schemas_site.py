from pydantic import BaseModel, ConfigDict

class SiteCreate(BaseModel):
    url: str
    scope: str
    type: str

class SiteResponse(SiteCreate):
    id: int
    status: str

    model_config = ConfigDict(from_attributes=True)

class ReingestResponse(BaseModel):
    status: str
    site_id: int
