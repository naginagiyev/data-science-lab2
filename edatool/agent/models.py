import os
from smolagents import LiteLLMModel, OpenAIModel

# builds the smolagents model from the settings (OpenAI or LiteLLM) with the...
# API key from the environment
def build_model(settings):
    agent_settings = settings["agent"]
    api_key = os.environ.get(agent_settings["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set the environment variable {agent_settings['api_key_env']} or pass a smolagents model.")
    model_class = LiteLLMModel if agent_settings["provider"] == "litellm" else OpenAIModel
    return model_class(model_id=agent_settings["model_id"], api_key=api_key, temperature=agent_settings["temperature"])
