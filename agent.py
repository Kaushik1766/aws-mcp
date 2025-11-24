from strands import Agent
from strands_tools import current_time, use_aws

agent = Agent(
    tools = [
        current_time,
        use_aws
    ]
)