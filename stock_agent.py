from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

web_agent = Agent(
		name = "Web Agent",
		model = Groq(id = "llama-3.3-70b-versatile"),
		tools = [DuckDuckGo()],
		markdown = True,
		instructions = ["Always include sources"],
		show_tool_calls = True
)


finance_agent = Agent(
		name = "Finance Agent",
		role = "Get Financial Agent",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    markdown = True,
    instructions = ["Use table to display data"]
)

agent_team = Agent(
		team = [web_agent, finance_agent],
		model = Groq(id = "llama-3.3-70b-versatile"),
		instruction = ["Always include sources", "Use table to display data"],
		show_tool_calls = True,
		markdown = True
)

message = "Summarize and compare analyst recommendation and fundamentals for TSLA and NVDA"
sanitized_message = message.encode('ascii', errors='ignore').decode()
agent_team.print_response(sanitized_message, stream=True)