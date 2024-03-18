from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import List

from agent import run_agent
import json
import asyncio

app = FastAPI()


class Message(BaseModel):
    type: str
    content: str


class ConversationInfo(BaseModel):
    conversationId: str
    userId: str


class GenerateArgs(BaseModel):
    messages: List[Message]
    stream: bool = False
    conversationInfo: ConversationInfo = None


def prettify_response(text):
    prettified_text = ""
    for line in text.split("\n"):
        if line.startswith("Thought:"):
            prettified_text += "<p style='color:#065f46'>" + line + "</p>"
        elif line.startswith("Action:"):
            prettified_text += "<p style='color:#b91c1c'>" + line + "</p>"
        elif line.startswith("Action Input:"):
            prettified_text += "<p style='color:#4338ca'>" + line + "</p>"
        elif line.startswith("Observation:"):
            observation = line.split("Observation:")[1].strip()
            try:
                observation = json.loads(eval(observation))
                prettified_text += (
                    "<p style='color:#92400e'>Observation:</p>\n\n```json\n"
                    + json.dumps(observation, indent=4)
                    + "\n```\n\n"
                )
            except:
                prettified_text += "<p style='color:#92400e'>" + line + "</p>"
        else:
            prettified_text += line
        prettified_text += "\n"
    return prettified_text


@app.post("/chat/generate")
def generate(args: GenerateArgs):
    def run_model():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        question = args.messages[-1].content

        generated_text, ans = run_agent(
            question,
            uid=args.conversationInfo.userId if args.conversationInfo else "test_user",
            session_id=args.conversationInfo.conversationId
            if args.conversationInfo
            else None,
        )
        # fetch 'leave' if it exists @shiwei
        pass

        prettified_text = prettify_response(generated_text)

        yield "data:" + json.dumps(
            {
                "token": {"id": -1, "text": "", "special": False, "logprob": 0},
                "generated_text": prettified_text,
                "details": None,
            }
        ) + "\n"

    return StreamingResponse(run_model())
