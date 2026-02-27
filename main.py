from fastapi import FastAPI, Query, HTTPException
from transformers import pipeline

app = FastAPI(title="AI Text Detector API")

# Инициализируем пайплайн для классификации текста
# При первом запуске скачается модель (около 500 МБ)
try:
    pipe = pipeline("text-classification", model="roberta-base-openai-detector")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    pipe = None

@app.get("/predict")
async def predict_ai_generated(text: str = Query(..., min_length=10, description="Текст для анализа")):
    """
    Принимает текст и возвращает вероятность того, что он был сгенерирован ИИ.
    """
    if pipe is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Модель выдает результат в формате: 
        # [{'label': 'Fake', 'score': 0.99}] или [{'label': 'Real', 'score': 0.99}]
        result = pipe(text)[0]
        
        label = result['label']
        score = result['score']

        # Если модель говорит 'Fake', значит это ИИ. 
        # Если 'Real', то вероятность ИИ = 1 - score.
        ai_probability = score if label == 'Fake' else 1 - score

        return {
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "ai_probability": round(ai_probability, 4),
            "is_ai_generated": ai_probability > 0.5,
            "label_from_model": label,
            "raw_score": round(score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)