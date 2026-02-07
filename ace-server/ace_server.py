# ace_server.py
from fastapi import FastAPI
from pydantic import BaseModel

# import your ACE functions / model init here
# from ace_core import ace_once, load_mem, save_mem

app = FastAPI()

class GenReq(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate")
def generate(req: GenReq):
    # response = ace_once(req.prompt, mem)
    response = "TODO: wire your ace_once here"
    return {"response": response}
