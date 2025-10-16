"""
Replace ReAct algorithm with LangChain tool calling approach
- Remove manual ReAct prompt template and parsing logic
- Implement .bind_tools() for direct tool integration with LLM
- Replace agent scratchpad tracking with message-based conversation flow
- Simplify tool execution using built-in ToolMessage handling
- Maintain same functionality with cleaner, more maintainable code
"""


# Importing Required Libraries
from typing import List

from dotenv import load_dotenv
from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

# Loading Environment Variables
load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Return the length of text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')  # stripping away non-alphabetic characters

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    print("Hello LangChain Tools (.bind_tools)!")
    tools = [get_text_length]

    # Defining the basic llm with the callbacks
    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    # Defining the llm with the tools
    llm_with_tools = llm.bind_tools(tools)

    # Starting the conversation
    messages = [HumanMessage(content="What is the length of the word: Hello")]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools execute them and return the results
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:

                # tool call is typically a dict with the keys like id, type, name , args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id = tool_call_id
                    )
                )

                # Continue loop to allow model to use observations
                continue



        # No tool call -> Final Answer
        print(ai_message.content)
        break
