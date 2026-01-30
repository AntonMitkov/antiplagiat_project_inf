from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CheckRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    originality: float

@app.post("/check", response_model=CheckResponse)
def check_text(req: CheckRequest):
    plagiat_score = 0.37
    originality = round(100*(1-plagiat_score), 2)

    return {"originality": originality}