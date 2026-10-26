from fastapi import FastAPI
from UserRequest import UserRequest
from load_vector_database import initialize_rag_pipeline
from contextlib import asynccontextmanager

app_state={}


@asynccontextmanager
async def lifespan(app:FastAPI):
    retriever =  initialize_rag_pipeline(persist_dir="storage/monopoly_rag_data")

    app_state['retriever'] = retriever

    yield

    app_state.clear()

app = FastAPI(lifespan=lifespan)



@app.post("/query")
async def query(user_request:UserRequest):
    query = user_request.query

    retriever = app_state["retriever"]
    if not retriever:
        return {"error":"Retriever does not exist."}

    response = await retriever.aquery(query)

    return {"response" : str(response)}
