import os
import time

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Vertex AI 설정 (환경변수로 override 가능)
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "prism-485101")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _initialized = True


def gemini_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
) -> str:
    _ensure_init()

    start_time = time.time()

    gemini_model = GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            stop_sequences=[s for s in stop_sequences if s] or None if stop_sequences else None,
        ),
    )

    if check_time:
        elapsed = time.time() - start_time
        print(f"[GEMINI/VertexAI] Model: {model}, Response time: {elapsed:.2f}s")

    return response.text
