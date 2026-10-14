from src.agent import agent

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [("human", "Explain roaming activation")]
        }

    )
    print(result['messages'][-1].content)