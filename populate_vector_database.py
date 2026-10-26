import asyncio

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.google_genai import GoogleGenAI
import os

from rag_agent import *

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def save_to_disk(nodes):
    embed_model = resolve_embed_model("local:BAAI/bge-small-en")

    vector_index_metadata = VectorStoreIndex(nodes,embed_model=embed_model)

    vector_index_metadata.storage_context.persist(persist_dir="storage/monopoly_rag_data")

async def populate_vector_store():
    llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=GEMINI_API_KEY)

    docs = get_documents()
    base_nodes = split_documents(docs)
    metadata_nodes = await run_extractor(llm=llm,base_nodes=base_nodes)
    all_nodes = get_all_nodes(base_nodes=base_nodes,node_to_metadata=metadata_nodes)

    save_to_disk(all_nodes)

    print("saved.")

asyncio.run(populate_vector_store())