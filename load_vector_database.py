from typing import Dict

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import RecursiveRetriever
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode
import os

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def load_index_and_nodes(persist_dir: str):

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    vector_index_loaded = load_index_from_storage(
        storage_context, embed_model="local:BAAI/bge-small-en")

    all_nodes_dict = storage_context.docstore.docs

    return all_nodes_dict, vector_index_loaded


def build_recursive_retriever(vector_index: BaseIndex, all_nodes_dict: Dict[str, BaseNode]):
    vector_retriever = vector_index.as_retriever(similarity_top_k=15)
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_nodes_dict,
        verbose=False,
    )
    return retriever_metadata


def build_query_engine(retriever_metadata: RecursiveRetriever):
    llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-large",
        top_n=3
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever_metadata, llm=llm, node_postprocessors=[reranker])

    return query_engine


def initialize_rag_pipeline(persist_dir: str):

    all_nodes_dict, vector_index = load_index_and_nodes(
        persist_dir=persist_dir)

    retriever = build_recursive_retriever(
        vector_index=vector_index,
        all_nodes_dict=all_nodes_dict
    )

    query_engine = build_query_engine(retriever_metadata=retriever)

    return query_engine