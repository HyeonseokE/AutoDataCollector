import time
import google.generativeai as genai

GOOGLE_API_KEY = "Input_Your_Path"
genai.configure(api_key=GOOGLE_API_KEY)


def gemini_response(
    prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
) -> str:
    start_time = time.time()

    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature
        )
    )

    if check_time:
        elapsed = time.time() - start_time
        print(f"[GEMINI] Model: {model}, Response time: {elapsed:.2f}s")

    return response.text
