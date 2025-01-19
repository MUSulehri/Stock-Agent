from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import os
import re

# Force reload environment variables
load_dotenv(override=True)

print(f"GROQ API Key loaded: {'GROQ_API_KEY' in os.environ}")
print(f"First few chars of key: {os.getenv('GROQ_API_KEY')[:10] if os.getenv('GROQ_API_KEY') else 'Not found'}")

app = FastAPI()

class StockRequest(BaseModel):
	stock_prompt: str

class StockResponse(BaseModel):
	stock_result: str

@app.post("/stock")
async def get_stock_data(request: StockRequest):

	web_agent = Agent(
		name = "Web Agent",
		model = Groq(id = "llama-3.3-70b-versatile"),
		tools = [DuckDuckGo()],
		markdown = True,
		# instructions = ["Always include sources"],
		# show_tool_calls = True
	)

	finance_agent = Agent(
		name = "Finance Agent",
		role = "Get Financial Agent",
		model = Groq(id = "llama-3.3-70b-versatile"),
		tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
		# markdown = True,
		# instructions = ["Use table to display data"]
	)

	agent_team = Agent(
		team = [web_agent, finance_agent],
		model = Groq(id = "llama-3.3-70b-versatile"),
		instructions = [
			"Use a table format to display data for clarity.",
			"Ensure responses are concise and do not exceed a specified length.",
			"Provide captions for tables, such as 'Analyst Recommendations' or 'Stock Fundamentals'.",
			"Analyst Recommendations should include a conclusive recommendation (e.g., Buy, Hold, or Sell).",
			"Stock Fundamentals should include metrics like Open, High, Low, Market Cap, P/E ratio, Div yield, 52-wk high and 52-wk low",
			"Handle errors gracefully and provide a fallback message if data cannot be retrieved."
		],
		# show_tool_calls = True,
		markdown = True
	)

	sanitized_message = request.stock_prompt.encode('ascii', errors='ignore').decode()
	response = agent_team.run(sanitized_message)
	response_dict = dict(response)

	content = re.sub(r'\nRunning:[\s\S]+?(?=\n###)', '', response_dict['content'])
	print(content)

	return StockResponse(stock_result=content)