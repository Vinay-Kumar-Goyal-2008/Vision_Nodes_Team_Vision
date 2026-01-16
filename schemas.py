from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    agentIds: List[str]
    responseMode: str = "sync"
    externalUserId: Optional[str] = None
    filePath: Optional[str] = None
    fileName: Optional[str] = None
