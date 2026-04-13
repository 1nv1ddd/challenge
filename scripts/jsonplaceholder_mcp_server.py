"""
MCP-сервер вокруг публичного mock API JSONPlaceholder (без ключей).

Инструменты:
  - get_post — пост по id
  - get_user — пользователь по id

Запуск: python scripts/jsonplaceholder_mcp_server.py
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

app = FastMCP("jsonplaceholder-api")

_JSON_URL = "https://jsonplaceholder.typicode.com"


@app.tool()
def get_post(post_id: int) -> str:
    """Вернуть заголовок и текст поста блога с jsonplaceholder.typicode.com по числовому id."""
    r = httpx.get(f"{_JSON_URL}/posts/{int(post_id)}", timeout=15.0)
    r.raise_for_status()
    data = r.json()
    payload = {"id": data.get("id"), "title": data.get("title"), "body": data.get("body")}
    return json.dumps(payload, ensure_ascii=False)


@app.tool()
def get_user(user_id: int) -> str:
    """Вернуть имя, email и телефон пользователя jsonplaceholder по id."""
    r = httpx.get(f"{_JSON_URL}/users/{int(user_id)}", timeout=15.0)
    r.raise_for_status()
    data = r.json()
    payload = {
        "id": data.get("id"),
        "name": data.get("name"),
        "email": data.get("email"),
        "phone": data.get("phone"),
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    app.run(transport="stdio")
