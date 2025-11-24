import json

from aws_lambda_typing.events import APIGatewayProxyEventV2
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import current_time, use_aws

model = BedrockModel(model_id="amazon.nova-pro-v1:0", region_name="us-east-1")

agent = Agent(tools=[current_time, use_aws], model=model)


def main():
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            response = agent(user_input)
            print(f"Agent: {response}")
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()


def handler(event: APIGatewayProxyEventV2, context):
    # user_input = event.get("input", "user didnt specify input")
    user_input = event["body"]
    print(user_input)
    response = agent(user_input)
    print(response.__str__())
    return {"body": json.dumps({"response": str(response)}), "statusCode": 200}
