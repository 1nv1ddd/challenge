FROM python:3.12-slim

WORKDIR /app

COPY bot/requirements.txt /app/bot/requirements.txt
RUN pip install --no-cache-dir -r /app/bot/requirements.txt

COPY bot /app/bot
COPY data/support_tickets.json /app/data/support_tickets.json

CMD ["python", "-m", "bot.main"]
