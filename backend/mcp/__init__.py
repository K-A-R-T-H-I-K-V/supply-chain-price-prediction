"""MCP (Model Context Protocol) integration"""
from .weather_server import WeatherMCPServer, get_weather_mcp
from .news_server import get_news_mcp
from .finance_server import get_finance_mcp

__all__ = [
	'WeatherMCPServer', 'get_weather_mcp',
	'get_news_mcp', 'get_finance_mcp'
]
