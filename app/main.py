from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.api.models import AnalysisResponse
from services.image_analyzer import ImageAnalyzer
from services.text_analyzer import TextAnalyzer

app = FastAPI()
text_analyzer = TextAnalyzer()
image_analyzer = ImageAnalyzer()

class TextRequest(BaseModel):
    content: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    try:
        return text_analyzer.analyze(request.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return image_analyzer.analyze(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))