from cocoindex import op
from cocoindex.typing import Float32, Vector, TypeAttr, Range, NamedTuple
from typing import Annotated, List, Dict

class Chunk(NamedTuple):
    location: Range
    text: str

class CustomSplitter(op.FunctionSpec):
    """
    """

@op.executor_class(cache=False, behavior_version=1)
class CustomSplitterExecutor:
    spec: CustomSplitter


    def analyze(self, content):
        """
        :param content (str) - Content of file
        :return: dict
                - location: tuple[int, int]
                - text: str
        """
        return Annotated[
            list[Chunk],
            TypeAttr("cocoindex.io/chunk_base_text", content.analyzed_value)
        ]

    def __call__(self, content: str, *, chunk_size: int = 1000, chunk_overlap: int = 0, language: str = "markdown") -> list[Chunk]:
        chunks = []
        pos = 0
        length = len(content)
        while pos < length:
            end_pos = min(pos + chunk_size, length)
            chunk_text = content[pos:end_pos]
            chunks.append(Chunk(location=(pos, end_pos), text=chunk_text))
            pos = end_pos
        return chunks
