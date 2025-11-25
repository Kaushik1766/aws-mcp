import asyncio
import json

from aws_lambda_typing.events import APIGatewayProxyEventV2
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import current_time, use_aws

model = BedrockModel(model_id="amazon.nova-micro-v1:0", region_name="us-east-1")

agent = Agent(tools=[current_time, use_aws], model=model)


async def main():
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            async for e in agent.stream_async(user_input):
                # print(e)
                if "data" in e:
                    print(e["data"], end="")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    asyncio.run(main())


def handler(event: APIGatewayProxyEventV2, context):
    # user_input = event.get("input", "user didnt specify input")
    user_input = event.get("body")
    print(user_input)
    response = agent(user_input)
    print(response.__str__())
    return {"body": json.dumps({"response": str(response)}), "statusCode": 200}
