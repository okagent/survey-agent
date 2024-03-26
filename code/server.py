from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import List

from agent import run_agent
import json
import asyncio

from utils import default_user

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

	def split_text_into_segments(text):
		# 分割词列表
		separators = ['Thought:', 'Action:', 'Action Input:', 'Observation:']

		# 将文本按照换行符切分成多行
		lines = text.split('\n')
		segments = []
		current_segment = []

		for line in lines:
			# 如果当前行是分割词之一，开始一个新的段落
			if any(line.startswith(separator) for separator in separators):
				if current_segment:
					segments.append('\n'.join(current_segment))
					current_segment = []

			current_segment.append(line)
			if line.startswith('Observation:'):
				segments.append('\n'.join(current_segment))
				current_segment = []

		# 添加最后一个段落，如果有的话
		if current_segment:
			segments.append('\n'.join(current_segment))

		return segments

	def to_md_json(text):
		orig_text = text
		try:
			text = json.loads(text)
			text = json.dumps(text, indent=4)
		except:
			text = orig_text

		return "```json\n" + text  + "\n```"

	for line in split_text_into_segments(text):
		if line.startswith("Thought:"):
			prettified_text += "<p style='color:#065f46'>" + line + "</p>"
		elif line.startswith("Action:"):
			prettified_text += "<p style='color:#b91c1c'>" + line + "</p>"
		elif line.startswith("Action Input:"):
			actioninput = line.split("Action Input:")[1].strip()

			prettified_text += (
				"<p style='color:#4338ca'>Action Input:</p>\n\n"
				+ to_md_json(actioninput)
				+ "\n\n"
			)
		elif line.startswith("Observation:"):
			observation = line.split("Observation:")[1].strip()
				
			prettified_text += (
				"<p style='color:#92400e'>Observation:</p>\n\n" 
				+ to_md_json(observation)
				+ "\n\n"
			)
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

		try:
			generated_text = run_agent(
				question,
				uid=args.conversationInfo.userId if args.conversationInfo else default_user,
				session_id=args.conversationInfo.conversationId
				if args.conversationInfo
				else None,
			)
			text = prettify_response(generated_text)
		except Exception as e:
			text = "Error: " + str(e)
		# fetch 'leave' if it exists @shiwei
		pass

		yield "data:" + json.dumps(
			{
				"token": {"id": -1, "text": "", "special": False, "logprob": 0},
				"generated_text": text,
				"details": None,
			}
		) + "\n"

	return StreamingResponse(run_model())
