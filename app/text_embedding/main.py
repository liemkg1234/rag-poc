import cocoindex
from functions.openai_embed import OpenAIEmbed
from functions.llama_index_split import CustomSplitter
from cocoindex.typing import Range, NamedTuple

COLLECTION_NAME = "VanBanPhatLuat_NganHangNhaNuocVietNam_collection"
DATASET_PATH = "../../dataset/preprocessed/Văn bản pháp luật - Ngân hàng nhà nước Việt Nam"
DATABASE_URL = "postgresql://cocoindex:cocoindex@localhost:5432/cocoindex"

class Chunk(NamedTuple):
    location: Range
    text: str

@cocoindex.transform_flow()
def embedder(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
    """
    Embed the text using a likely OpenAI Server model.
    This is a shared logic between indexing and querying, so extract it as a function.
    """
    # return text.transform(OpenAIEmbed(
    #     model="multilingual-e5-large-instruct",
    #     base_url="http://localhost:8111/v1",
    #     api_key="<NONE>",
    #     dimension=1024,
    # ))

    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

@cocoindex.transform_flow()
def chunker(content: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[Chunk]]:
    """
    chunk the content
    """

    return content.transform(CustomSplitter(
        # model="multilingual-e5-large-instruct",
        # base_url="http://localhost:8111/v1",
        # api_key="<NONE>",
        # dimension=1024,
    ))



@cocoindex.flow_def(name="TextEmbedding")
def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    Define a flow to process text documents and generate embeddings.
    """

    # Load files
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path=DATASET_PATH))

    # Collector
    doc_embeddings = data_scope.add_collector()

    # Chunk
    with data_scope["files"].row() as file:
        # file["chunks"] = file["content"].transform(
        #     cocoindex.functions.SplitRecursively(),
        #     language="markdown", chunk_size=2000, chunk_overlap=500)
        # print(file["chunks"])


        file["chunks"] = chunker(file["content"])
        print(file["chunks"])


        # Embed
        with file["chunks"].row() as chunk:
            chunk["embedding"] = embedder(chunk["text"])

            # Add into a collection
            doc_embeddings.collect(
                filename=file["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"])

    # Export
    doc_embeddings.export(
        COLLECTION_NAME,
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)])


if __name__ == "__main__":
    cocoindex.init(
        cocoindex.Settings(
            database=cocoindex.DatabaseConnectionSpec(url=DATABASE_URL),
        )
    )
