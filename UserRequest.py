from pydantic import BaseModel


class UserRequest(BaseModel):
    query: str