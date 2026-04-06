"""
Локальный MCP-сервер (stdio) для проверки клиента.

Запуск: python scripts/minimal_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

app = FastMCP("challenge-minimal")


@app.tool()
def ping() -> str:
    """Проверка связи."""
    return "pong"


@app.tool()
def echo(message: str) -> str:
    """Вернуть переданную строку."""
    return message


if __name__ == "__main__":
    app.run(transport="stdio")
