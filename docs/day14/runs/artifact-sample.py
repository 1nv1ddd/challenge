# Принято execution loop: committed_with_warnings
# Прогон: 6d6a4c0c, 2026-08-06T13:32:21+00:00
# Warning (не блокирует): [medium] В httpx.get не указан таймаут, возможна блокировка выполнения.

import httpx

def fetch_partner_data(client_id: str, api_key: str) -> dict | None:
    url = f"https://api.partner.example/v1/clients/{client_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = httpx.get(url, headers=headers, verify=True)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None
