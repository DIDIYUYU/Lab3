from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.text_generator import TextGenerator, TextGeneratorParams

root_router = APIRouter()


class TextGenerationRequest(BaseModel):
    text: str
    params: TextGeneratorParams = TextGeneratorParams()


class TextGenerationResponse(BaseModel):
    generated_text: str


@root_router.post('/text-generation')
def generate_text(
    request: TextGenerationRequest,
    text_generator: Annotated[TextGenerator, Depends()],
) -> TextGenerationResponse:
    generated_text = text_generator.generate_text(request.text, request.params)
    return TextGenerationResponse(generated_text=generated_text)
