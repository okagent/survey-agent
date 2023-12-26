from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import List

from agent import agent_executor
import json
import asyncio

app = FastAPI()


class Message(BaseModel):
    type: str
    content: str


class GenerateArgs(BaseModel):
    messages: List[Message]
    stream: bool = False


@app.post("/chat/generate")
def generate(args: GenerateArgs):
    def run_model():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        question = args.messages[-1].content
        generated_text = agent_executor.run(question)
        yield "data:" + json.dumps(
            {
                "token": {"id": -1, "text": "", "special": False, "logprob": 0},
                "generated_text": generated_text,
                "details": None,
            }
        ) + "\n"

    return StreamingResponse(run_model())
