# Import Required Libraries
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Loading Environment Variables
load_dotenv()

# Creating a list of tools that the llm can use
tools = [TavilySearch()]


llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using  langchain in the UAE on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
