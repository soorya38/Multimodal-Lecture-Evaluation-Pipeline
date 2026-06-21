from fastapi import FastAPI

app = FastAPI(
    title="Multimodal Lecture Evaluation Pipeline",
    version="1.0.0"
)

@app.get("/health")
async def health():
    return {"status": "ok"}