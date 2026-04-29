"""Telegram-бот поддержки PolarLine (Day 33).

Поток:
- юзер пишет любой вопрос → бот форвардит на бэк с префиксом /support → возвращает AI-ответ
  с inline-кнопками «Помогло / Не помогло».
- «Не помогло» → создаёт тикет (data/support_tickets.json), уведомляет юзера, что
  оператор ответит здесь же.
- /admin → список открытых тикетов с inline-кнопками; админ может ответить
  юзеру (текст уйдёт в чат) и закрыть тикет.

Env:
- TELEGRAM_BOT_TOKEN — обязательно
- ADMIN_TG_IDS — через запятую (без пробелов), telegram user id'ы операторов
- BACKEND_URL — http://app:8000 в docker-compose, http://localhost:8000 при локальном запуске
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_APPENDIX_RE = re.compile(r"\n+##\s*(Источники|Цитаты)\b", re.IGNORECASE)


def _strip_rag_appendix(text: str) -> str:
    """Срезает блоки `## Источники` / `## Цитаты` (Day-24 автогенерация)."""
    m = _APPENDIX_RE.search(text)
    if not m:
        return text
    return text[: m.start()].rstrip()

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.tickets import (  # noqa: E402
    append_message,
    create_ticket,
    get_ticket,
    list_open_tickets,
    set_status,
    short_preview,
)


BACKEND_URL = os.environ.get("BACKEND_URL", "http://app:8000").rstrip("/")
TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
ADMIN_IDS: set[int] = {
    int(x) for x in (os.environ.get("ADMIN_TG_IDS") or "").split(",") if x.strip().isdigit()
}
ADMIN_PASSWORD = (os.environ.get("ADMIN_PASSWORD") or "12346").strip()
DEFAULT_PROVIDER = os.environ.get("BOT_PROVIDER", "routerai")
DEFAULT_MODEL = os.environ.get("BOT_MODEL", "openai/gpt-4o-mini")

# Авторизованные в этой сессии бот-процесса админы. Сбрасывается при рестарте контейнера —
# повторная аутентификация через `/admin <пароль>` обязательна.
_authed_admins: set[int] = set()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("support-bot")

router = Router()

# Кэшируем последний AI-ответ на чат — чтобы при «Не помогло» сложить его в тикет.
_last_exchange: dict[int, dict[str, str]] = {}


# --- помощники ---

def _is_admin(user_id: int) -> bool:
    """Авторизованный админ в текущей сессии (вошёл через /admin <пароль>)."""
    return user_id in _authed_admins


def _is_eligible_admin(user_id: int) -> bool:
    """Tg-id есть в ADMIN_TG_IDS, но это ещё не значит что юзер ввёл пароль."""
    return user_id in ADMIN_IDS


def _kb_helped(message_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="✅ Помогло", callback_data=f"helped:{message_id}"),
            InlineKeyboardButton(text="❌ Не помогло", callback_data=f"escalate:{message_id}"),
        ]]
    )


def _kb_admin_ticket(tid: str, status: str) -> InlineKeyboardMarkup:
    rows = [[
        InlineKeyboardButton(text="✏️ Ответить юзеру", callback_data=f"adm_reply:{tid}"),
    ]]
    if status != "closed":
        rows.append([
            InlineKeyboardButton(text="✅ Закрыть тикет", callback_data=f"adm_close:{tid}"),
        ])
    rows.append([
        InlineKeyboardButton(text="↩️ Список", callback_data="adm_list"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_admin_list(tickets: list[dict]) -> InlineKeyboardMarkup:
    rows = []
    for t in tickets[:20]:
        label = f"{t.get('id','?')} · {t.get('priority','?')} · {(t.get('title') or '—')[:30]}"
        rows.append([
            InlineKeyboardButton(text=label, callback_data=f"adm_view:{t.get('id')}"),
        ])
    if not rows:
        rows.append([InlineKeyboardButton(text="(нет открытых)", callback_data="adm_noop")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _ask_backend(user_text: str, conversation_id: str) -> str:
    """Стучимся в /api/chat с префиксом /support и собираем ответ."""
    payload = {
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "conversation_id": conversation_id,
        "messages": [{"role": "user", "content": f"/support {user_text}"}],
        "temperature": 0.2,
        "task_workflow": False,
    }
    text_parts: list[str] = []
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST", f"{BACKEND_URL}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data.startswith("[META]") or data == "[DONE]" or data.startswith("[ERROR]"):
                    if data.startswith("[ERROR]"):
                        text_parts.append(f"\n\n⚠️ {data}")
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, str):
                    text_parts.append(chunk)
    raw = "".join(text_parts).strip()
    return _strip_rag_appendix(raw) or "(пустой ответ от модели)"


# --- состояния FSM для админского reply ---

class AdminReply(StatesGroup):
    waiting_text = State()


# --- команды юзеров ---

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 Привет! Я AI-ассистент поддержки <b>AIChatHub</b>.\n\n"
        "Просто опиши проблему — я отвечу на основе базы знаний (FAQ + handbook). "
        "Если ответ не подойдёт — нажми «❌ Не помогло», и тикет уйдёт оператору.\n\n"
        "Команды: /myid · /admin &lt;пароль&gt; (для операторов)"
    )


@router.message(Command("myid"))
async def cmd_myid(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    await message.answer(
        f"Ваш Telegram ID: <code>{uid}</code>\n\n"
        f"Чтобы получить доступ к /admin, добавьте этот ID в env <code>ADMIN_TG_IDS</code> бота."
    )


# --- /admin ---

@router.message(Command("admin"))
async def cmd_admin(message: Message, command: CommandObject) -> None:
    uid = message.from_user.id if message.from_user else 0
    args = (command.args or "").strip()

    # Не в whitelist'е ADMIN_TG_IDS — никогда не авторизуем, даже с правильным паролем.
    if not _is_eligible_admin(uid):
        await message.answer("⛔ Эта команда — только для операторов.")
        return

    # Если пароль передан — проверяем и авторизуем (или отклоняем).
    if args:
        if args == ADMIN_PASSWORD:
            _authed_admins.add(uid)
            await message.answer("🔓 Пароль принят. Доступ открыт до перезапуска бота.")
        else:
            await message.answer("❌ Неверный пароль.")
            return

    # Без пароля — должны быть уже авторизованы.
    if uid not in _authed_admins:
        await message.answer(
            "🔒 Нужна авторизация. Напиши: <code>/admin &lt;пароль&gt;</code>"
        )
        return

    tickets = await list_open_tickets(20)
    head = f"🎫 <b>Открытые тикеты:</b> {len(tickets)}"
    if not tickets:
        await message.answer(f"{head}\n\nВ очереди пусто.")
        return
    body = "\n".join(short_preview(t) for t in tickets)
    await message.answer(f"{head}\n\n<pre>{body}</pre>", reply_markup=_kb_admin_list(tickets))


@router.callback_query(F.data == "adm_list")
async def cb_admin_list(cq: CallbackQuery) -> None:
    if not _is_admin(cq.from_user.id):
        await cq.answer("⛔", show_alert=True)
        return
    tickets = await list_open_tickets(20)
    body = "\n".join(short_preview(t) for t in tickets) or "В очереди пусто."
    await cq.message.edit_text(
        f"🎫 <b>Открытые тикеты:</b> {len(tickets)}\n\n<pre>{body}</pre>",
        reply_markup=_kb_admin_list(tickets),
    )
    await cq.answer()


@router.callback_query(F.data.startswith("adm_view:"))
async def cb_admin_view(cq: CallbackQuery) -> None:
    if not _is_admin(cq.from_user.id):
        await cq.answer("⛔", show_alert=True)
        return
    tid = cq.data.split(":", 1)[1]
    t = await get_ticket(tid)
    if not t:
        await cq.answer("Тикет не найден", show_alert=True)
        return
    text = _format_ticket_for_admin(t)
    await cq.message.edit_text(text, reply_markup=_kb_admin_ticket(tid, str(t.get("status", "open"))))
    await cq.answer()


@router.callback_query(F.data.startswith("adm_reply:"))
async def cb_admin_reply(cq: CallbackQuery, state: FSMContext) -> None:
    if not _is_admin(cq.from_user.id):
        await cq.answer("⛔", show_alert=True)
        return
    tid = cq.data.split(":", 1)[1]
    await state.update_data(reply_ticket=tid)
    await state.set_state(AdminReply.waiting_text)
    await cq.message.answer(
        f"Напиши ответ для тикета <b>{tid}</b> следующим сообщением. "
        f"Отмена: /cancel"
    )
    await cq.answer()


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    cur = await state.get_state()
    if cur is None:
        return
    await state.clear()
    await message.answer("Отменено.")


@router.message(AdminReply.waiting_text)
async def admin_reply_text(message: Message, state: FSMContext, bot: Bot) -> None:
    if not _is_admin(message.from_user.id):
        await state.clear()
        return
    data = await state.get_data()
    tid = data.get("reply_ticket")
    text = (message.text or "").strip()
    await state.clear()
    if not tid or not text:
        await message.answer("Пустой ответ — отменил.")
        return
    t = await append_message(tid, "agent", text)
    if not t:
        await message.answer(f"Тикет {tid} не найден.")
        return
    chat_id = t.get("tg_chat_id")
    if chat_id:
        try:
            await bot.send_message(
                int(chat_id),
                f"💬 <b>Оператор поддержки</b> (тикет <code>{tid}</code>):\n\n{text}",
            )
        except Exception as e:  # noqa: BLE001
            await message.answer(f"⚠️ Не смог отправить юзеру: {e}")
            return
    await message.answer(f"✅ Ответ отправлен в тикет {tid}.")


@router.callback_query(F.data.startswith("adm_close:"))
async def cb_admin_close(cq: CallbackQuery, bot: Bot) -> None:
    if not _is_admin(cq.from_user.id):
        await cq.answer("⛔", show_alert=True)
        return
    tid = cq.data.split(":", 1)[1]
    t = await set_status(tid, "closed")
    if not t:
        await cq.answer("не найден", show_alert=True)
        return
    chat_id = t.get("tg_chat_id")
    if chat_id:
        try:
            await bot.send_message(
                int(chat_id), f"✅ Ваш тикет <code>{tid}</code> закрыт оператором."
            )
        except Exception as e:  # noqa: BLE001
            log.warning("notify close failed: %s", e)
    await cq.message.edit_text(
        _format_ticket_for_admin(t), reply_markup=_kb_admin_ticket(tid, "closed")
    )
    await cq.answer("Закрыт")


@router.callback_query(F.data == "adm_noop")
async def cb_noop(cq: CallbackQuery) -> None:
    await cq.answer()


def _format_ticket_for_admin(t: dict) -> str:
    history = "\n".join(
        f"  - <i>{m.get('role','?')}</i>: {str(m.get('text','')).replace('<', '&lt;')[:300]}"
        for m in (t.get("messages") or [])
    )
    return (
        f"🎫 <b>{t.get('id','?')}</b>\n"
        f"тариф: <b>{t.get('priority','?')}</b> · статус: <b>{t.get('status','?')}</b>\n"
        f"автор: <code>{t.get('tg_username') or t.get('tg_full_name') or 'unknown'}</code> "
        f"(chat_id <code>{t.get('tg_chat_id') or '—'}</code>)\n"
        f"тема: <i>{(t.get('title') or '').replace('<', '&lt;')}</i>\n\n"
        f"<pre>{history}</pre>"
    )


# --- основной поток: любой текст юзера ---

@router.callback_query(F.data.startswith("helped:"))
async def cb_helped(cq: CallbackQuery) -> None:
    await cq.message.edit_reply_markup(reply_markup=None)
    await cq.message.answer("👍 Рад, что помог! Если будет ещё вопрос — пиши.")
    await cq.answer()


@router.callback_query(F.data.startswith("escalate:"))
async def cb_escalate(cq: CallbackQuery, bot: Bot) -> None:
    chat_id = cq.message.chat.id
    last = _last_exchange.get(chat_id)
    if not last:
        await cq.answer("Не нашёл, на какой вопрос эскалировать", show_alert=True)
        return
    user = cq.from_user
    t = await create_ticket(
        title=last["question"][:120],
        user_text=last["question"],
        ai_answer=last["answer"],
        tg_chat_id=chat_id,
        tg_username=user.username,
        tg_full_name=user.full_name,
        priority="medium",
    )
    await cq.message.edit_reply_markup(reply_markup=None)
    await cq.message.answer(
        f"📝 Создал тикет <code>{t['id']}</code>. Оператор ответит здесь же — "
        f"уведомление придёт в этом чате."
    )
    # Уведомляем только авторизованных админов (вошедших через /admin <пароль>).
    for admin_id in list(_authed_admins):
        try:
            await bot.send_message(
                admin_id,
                f"🆕 Новый тикет <code>{t['id']}</code> от "
                f"<code>{user.username or user.full_name}</code>:\n\n"
                f"<i>{last['question'][:300]}</i>\n\n/admin",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("notify admin %s failed: %s", admin_id, e)
    await cq.answer()


_ESCALATE_MARKER = "[NEED_HUMAN]"


async def _auto_escalate(
    bot: Bot, message: Message, question: str, ai_brief: str
) -> None:
    """Если модель прислала [NEED_HUMAN] — заводим тикет без кнопок."""
    chat_id = message.chat.id
    user = message.from_user
    t = await create_ticket(
        title=question[:120],
        user_text=question,
        ai_answer=f"{_ESCALATE_MARKER} {ai_brief}".strip(),
        tg_chat_id=chat_id,
        tg_username=user.username if user else None,
        tg_full_name=user.full_name if user else "—",
        priority="medium",
    )
    await message.answer(
        f"🤖 Я не нашёл точного ответа в базе знаний — передал вопрос оператору.\n\n"
        f"Тикет <code>{t['id']}</code> создан, ответ придёт сюда же.",
    )
    for admin_id in list(_authed_admins):
        try:
            await bot.send_message(
                admin_id,
                f"🆕 Новый тикет <code>{t['id']}</code> от "
                f"<code>{(user.username if user else None) or (user.full_name if user else '—')}</code> "
                f"(<i>авто-эскалация: AI не знает ответа</i>):\n\n"
                f"<i>{question[:300]}</i>\n\n/admin",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("notify admin %s failed: %s", admin_id, e)


@router.message(F.text)
async def on_message(message: Message, bot: Bot) -> None:
    if message.text and message.text.startswith("/"):
        # необработанная команда
        return
    chat_id = message.chat.id
    question = message.text or ""
    thinking = await message.answer("💭 думаю…")
    try:
        answer = await _ask_backend(question, conversation_id=f"tg-{chat_id}")
    except httpx.HTTPError as e:
        log.exception("backend error")
        await thinking.edit_text(f"⚠️ Бэк не отвечает: {e}")
        return

    # Авто-эскалация: первая строка ответа начинается с [NEED_HUMAN].
    head = answer.lstrip()
    if head.startswith(_ESCALATE_MARKER):
        # Снимаем маркер, остаток первой строки — краткое описание для оператора.
        rest = head[len(_ESCALATE_MARKER):].strip()
        brief = rest.splitlines()[0] if rest else question[:120]
        await thinking.delete()
        await _auto_escalate(bot, message, question, brief)
        return

    _last_exchange[chat_id] = {"question": question, "answer": answer}
    text = answer if len(answer) <= 3500 else answer[:3500] + "…"
    # AI-текст идёт plain text — внутри встречаются `<пароль>`, `<id>` и прочие
    # ломающие HTML-парсер штуки; свои бот-сообщения остаются HTML.
    await thinking.edit_text(
        text, parse_mode=None, reply_markup=_kb_helped(thinking.message_id)
    )


# --- entry point ---

async def main() -> None:
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан")
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    log.info("Bot ready. Backend=%s, admins=%s", BACKEND_URL, sorted(ADMIN_IDS))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
