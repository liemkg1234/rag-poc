from cocoindex import op
from cocoindex.typing import Float32, Vector, TypeAttr
from typing import Annotated, List

import openai
import backoff


class OpenAIEmbed(op.FunctionSpec):
    """
    Use OpenAI-compatible embedding API to convert text to embeddings.
    """
    model: str
    base_url: str
    api_key: str
    dimension: int


@op.executor_class(gpu=True, cache=True, behavior_version=1)
class OpenAIEmbedExecutor:
    spec: OpenAIEmbed
    _client: openai.OpenAI

    def prepare(self):
        self._client = openai.OpenAI(
            base_url=self.spec.base_url,
            api_key=self.spec.api_key,
        )

    def analyze(self, text):
        return Annotated[
            Vector[Float32, self.spec.dimension],
            TypeAttr("cocoindex.io/vector_origin_text", text.analyzed_value)
        ]

    @backoff.on_exception(lambda: backoff.expo(base=0.3, factor=1.1, max_value=1.0), openai.RateLimitError, max_tries=10)
    def __call__(self, text: str) -> List[Float32]:
        response = self._client.embeddings.create(
            model=self.spec.model,
            input=text,
            encoding_format="float",
        )
        return [Float32(x) for x in response.data[0].embedding]
