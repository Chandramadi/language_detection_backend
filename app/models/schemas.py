
from typing import List
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]
