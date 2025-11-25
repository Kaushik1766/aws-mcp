import json
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import current_time, use_aws

app = FastAPI()
# model = BedrockModel(model_id="amazon.nova-micro-v1:0", region_name="us-east-1")
agent = Agent(tools=[current_time, use_aws], callback_handler=None)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/stream")
async def stream_response(request: PromptRequest):
    async def generate():
        try:
            async for event in agent.stream_async(request.prompt):
                if "data" in event:
                    # Format each data chunk for Server-Sent Events and encode as bytes
                    chunk = event["data"]
                    if chunk:
                        # Ensure the chunk is a string before encoding
                        if not isinstance(chunk, str):
                            chunk = str(chunk)
                        # Format as "data: {chunk}\n\n" and encode
                        yield f"data: {chunk}\n\n".encode("utf-8")
        except Exception as e:
            # Yield error message formatted for SSE
            error_message = f"Error: {str(e)}"
            yield f"data: {error_message}\n\n".encode("utf-8")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
