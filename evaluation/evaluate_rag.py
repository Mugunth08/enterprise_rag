from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from evaluation.gemini_model import GeminiDeepEvalModel


test_case = LLMTestCase(
    input="What is a mixture?",
    actual_output="Mixtures are combinations that can be separated physically.",
    expected_output="A mixture is a physical combination of two or more substances that can be separated by physical methods.",
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=GeminiDeepEvalModel(),
)

evaluate(
    test_cases=[test_case],
    metrics=[metric],
)