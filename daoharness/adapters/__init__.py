"""A ready-made model port, standard library only: any OpenAI-compatible endpoint."""
from .openai_compatible import MODES, PRESETS, OpenAICompatibleModel, parse_json_object

__all__ = ["MODES", "PRESETS", "OpenAICompatibleModel", "parse_json_object"]
