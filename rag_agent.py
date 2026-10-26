from typing import List

from llama_index.readers.file import PDFReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import resolve_embed_model
from pathlib import Path
import os
from dotenv import load_dotenv
import copy
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.retrievers import RecursiveRetriever

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#this function creates docs instances out of pdfs
def get_documents(): #you can put the path of the file in the parameter as well
    loader = PDFReader()
    docs = loader.load_data(file=Path("assets/monopoly-manual.pdf"))


    doc_text = "\n\n".join([d.get_content() for d in docs])
    docs = [Document(text=doc_text)]
    
    
    return docs



#splits doc into chunks and each chunk gets an id
def split_documents(docs: List[Document]): 
    node_parser = SentenceSplitter(chunk_size=1024)

    base_nodes = node_parser.get_nodes_from_documents(docs) #source:llamaindex docs
    # set node ids to be a constant
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"

    return base_nodes



def get_retriever(llm: GoogleGenAI, nodes):
    all_nodes_dict = {n.node_id: n for n in nodes}

    embed_model = resolve_embed_model("local:BAAI/bge-small-en")
    base_index = VectorStoreIndex(nodes, embed_model=embed_model)
    vector_retriever_metadata = base_index.as_retriever(similarity_top_k=15) #top n most relevant chunks
    retriever_metadata = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_metadata},
        node_dict=all_nodes_dict,
        verbose=False,
    )

    reranker = FlagEmbeddingReranker(
        model= "BAAI/bge-reranker-large",
        top_n = 3 #from the top k top n most relevant chunks are chosen
    )
    query_engine = RetrieverQueryEngine.from_args(retriever_metadata, llm=llm, node_postprocessors=[reranker])

    return query_engine

async def run_extractor(llm : GoogleGenAI,base_nodes: List[BaseNode]):
    extractors = [
        SummaryExtractor(summaries=["self"], show_progress=True, llm=llm),
        QuestionsAnsweredExtractor(questions=5, show_progress=True, llm=llm),
    ]
    # run metadata extractor across base nodes, get back dictionaries
    node_to_metadata = {}
    for extractor in extractors:
        metadata_dicts = await extractor.aextract(base_nodes)
        for node, metadata in zip(base_nodes, metadata_dicts):
            if node.node_id not in node_to_metadata:
                node_to_metadata[node.node_id] = metadata
            else:
                node_to_metadata[node.node_id].update(metadata)

    return node_to_metadata

def get_all_nodes(base_nodes: List[BaseNode], node_to_metadata: dict):
    # all nodes consists of source nodes, along with metadata
    all_nodes = copy.deepcopy(base_nodes)
    for node_id, metadata in node_to_metadata.items():
        for val in metadata.values():
            all_nodes.append(IndexNode(text=val, index_id=node_id))

    return all_nodes



llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)
async def initialize_rag_pipeline():

    doc = get_documents()
    BaseNodes = split_documents(doc)
    metadatanodes = await run_extractor(llm,BaseNodes)
    allNodes = get_all_nodes(BaseNodes, metadatanodes)
    retrieverTest = get_retriever(llm, allNodes)

    return retrieverTest


# retrieverResult = retrieverTest.query("How do we keep our games short and sweet? Explain it a bit") #PUT YOUR PROMPT HERE TO CHECK THE ANSWER
# print(str(retrieverResult))



