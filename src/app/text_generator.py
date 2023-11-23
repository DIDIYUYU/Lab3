import logging
from abc import abstractmethod, ABC

from pydantic import BaseModel
from transformers import pipeline, Pipeline

# https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.SummarizationPipeline
# https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate


class TextGeneratorParams(BaseModel):
    min_length: int = 30
    max_length: int = 130
    do_sample: bool = False


class TextGenerator(ABC):
    @abstractmethod
    def generate_text(self, text: str, params: TextGeneratorParams) -> str:
        ...


class TransformersTextGenerator(TextGenerator):
    @classmethod
    def from_model_name(cls, model_name: str):
        summarizer = cls(
            pipeline(
                "text-generation",
                model=model_name,
            )
        )
        return summarizer

    def __init__(self, ready_pipeline: Pipeline):
        self._pipeline = ready_pipeline

    def generate_text(self, text: str, params: TextGeneratorParams) -> str:
        result = self._pipeline(text, **params.model_dump())
        return result[0]['generated_text']

