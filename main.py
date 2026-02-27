import uvicorn
import torch
from fastapi import FastAPI, Query, HTTPException
from transformers import pipeline

app = FastAPI(title="AI Text Detector API (Long Text Support)")

# Выбираем устройство: GPU (0) или CPU (-1)
device = 0 if torch.cuda.is_available() else -1

print(f"Загрузка модели на {'GPU' if device == 0 else 'CPU'}...")

try:
    # Инициализируем пайплайн
    # Модель: roberta-base-openai-detector (классифицирует Real/Fake)
    pipe = pipeline(
        "text-classification", 
        model="roberta-base-openai-detector",
        device=device
    )
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    pipe = None

def split_text_into_chunks(text, max_words=300):
    """
    Разбивает текст на куски по количеству слов.
    300 слов — безопасный порог, чтобы точно уложиться в 512 токенов.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks

@app.get("/predict")
async def predict_ai_generated(text: str = Query(..., min_length=10, description="Текст для анализа")):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Текст пуст")

    try:
        # 1. Разбиваем длинный текст на части
        chunks = split_text_into_chunks(text)
        
        # 2. Прогоняем все части через модель одной пачкой (batch)
        # truncation=True гарантирует, что мы не превысим 512 токенов ни при каких условиях
        results = pipe(chunks, truncation=True, max_length=512)

        # 3. Агрегируем результаты
        total_ai_probability = 0
        chunk_details = []

        for i, res in enumerate(results):
            label = res['label']
            score = res['score']

            # Если модель говорит 'Fake', это вероятность ИИ.
            # Если 'Real', то вероятность ИИ = 1 - score.
            ai_prob = score if label == 'Fake' else 1 - score
            
            total_ai_probability += ai_prob
            
            chunk_details.append({
                "chunk_id": i + 1,
                "ai_probability": round(ai_prob, 4),
                "label": label
            })

        # Средняя вероятность по всем фрагментам
        final_ai_probability = total_ai_probability / len(chunks)

        return {
            "is_ai_generated": final_ai_probability > 0.5,
            "ai_probability": round(final_ai_probability, 4),
            "text_length_chars": len(text),
            "chunks_processed": len(chunks),
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "detailed_analysis": chunk_details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)