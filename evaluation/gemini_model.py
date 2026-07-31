from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from utils import get_model_from_gcp


class GeminiDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = get_model_from_gcp()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[type] = None):
        model = self.load_model()

        if schema is not None:
            structured_model = model.with_structured_output(schema)
            response = structured_model.invoke(prompt)
            return response

        response = model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str, schema: Optional[type] = None):
        model = self.load_model()

        if schema is not None:
            structured_model = model.with_structured_output(schema)
            response = await structured_model.ainvoke(prompt)
            return response

        response = await model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "gemini-2.5-flash"