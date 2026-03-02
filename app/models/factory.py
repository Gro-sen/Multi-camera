import os


def create_models():
    provider = os.getenv("MODEL_PROVIDER", "aliyun").lower()

    if provider == "ollama":
        from app.models.ollama_vision import OllamaVisionModel
        from app.models.ollama_reasoning import OllamaReasoningModel
        return OllamaVisionModel(), OllamaReasoningModel()

    from app.models.vision import AlibabaVisionModel
    from app.models.reasoning import AlibabaReasoningModel
    return AlibabaVisionModel(), AlibabaReasoningModel()