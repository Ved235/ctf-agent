import asyncio
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from cai.tools.reconnaissance.generic_linux_command import generic_linux_command

load_dotenv(override=True)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that executes linux commands and returns the output. Don't invent any output, just format it and display",
        model=os.environ["CAI_MODEL"],
        tools=[generic_linux_command],
    )

    result = await Runner.run(agent, "Find and return the model i am using in my filesytem.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
