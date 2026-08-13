#!/usr/bin/env python3
"""
CBZ/ZIP -> PDF Telegram Bot
===========================

A production-ready Telegram bot that converts .cbz (or plain .zip) comic
archives into a single PDF file.

Key features
------------
- Accepts .cbz / .zip files (rejects everything else with a clear message).
- Extracts images, sorts them naturally (page1, page2, ..., page10 - not
  page1, page10, page2), and stitches them into one PDF.
- STRICT global concurrency limit of 1: only one conversion job runs at a
  time across the whole bot. Everything else waits in an asyncio.Queue.
- Users get a live queue position message, and it is updated as the queue
  moves.
- Real-time progress messages: Queued -> Downloading -> Extracting ->
  Converting -> Uploading -> Done.
- All temp files/directories are removed after each job (success or
  failure) via try/finally.
- Graceful error handling: corrupted zip, empty archive, unsupported/
  unreadable images, oversized files, Telegram API errors.

Requirements
------------
    pip install "python-telegram-bot>=20,<22" Pillow

Set your bot token as an environment variable before running:
    export TELEGRAM_BOT_TOKEN="123456:ABC-your-token-here"
    python cbz_to_pdf_bot.py

Tech stack: Python 3.10+, python-telegram-bot v20+ (async), Pillow, zipfile,
asyncio.Queue + a single background worker task.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Safety limits - tune as needed for your server.
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".cbz", ".zip"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Base directory for all temporary work. Each job gets its own uuid subdir.
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "cbz2pdf_bot"

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cbz2pdf_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet PTB's http logs


# --------------------------------------------------------------------------
# Queue job model
# --------------------------------------------------------------------------

@dataclass
class ConversionJob:
    job_id: str
    chat_id: int
    file_id: str
    file_name: str
    file_size: int
    status_message_id: Optional[int] = None
    queue_position: int = 0
    work_dir: Path = field(default=None)  # type: ignore[assignment]


class JobQueue:
    """Wraps asyncio.Queue and keeps track of positions for user feedback."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ConversionJob] = asyncio.Queue()
        self._pending: list[ConversionJob] = []  # for position lookups
        self._lock = asyncio.Lock()

    async def put(self, job: ConversionJob) -> int:
        async with self._lock:
            self._pending.append(job)
            position = len(self._pending)  # 1-indexed, includes itself
        await self._queue.put(job)
        return position

    async def get(self) -> ConversionJob:
        job = await self._queue.get()
        async with self._lock:
            if job in self._pending:
                self._pending.remove(job)
        return job

    def task_done(self) -> None:
        self._queue.task_done()

    async def position_of(self, job: ConversionJob) -> int:
        async with self._lock:
            try:
                return self._pending.index(job) + 1
            except ValueError:
                return 0


job_queue = JobQueue()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def natural_sort_key(name: str):
    """Split a filename into digit / non-digit chunks so that
    'page2.jpg' sorts before 'page10.jpg'."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", name)
    ]


async def safe_edit(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, text: str) -> None:
    """Edit a status message, ignoring 'message is not modified' errors."""
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, parse_mode=ParseMode.HTML
        )
    except TelegramError as e:
        if "not modified" not in str(e).lower():
            logger.warning("Failed to edit status message: %s", e)


def extract_images_from_archive(archive_path: Path, extract_dir: Path) -> list[Path]:
    """Extract an archive and return a naturally-sorted list of image paths.

    Raises ValueError with a user-friendly message on any problem.
    """
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("This doesn't look like a valid .cbz/.zip archive (it may be corrupted).")

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                raise ValueError(f"Archive is corrupted (bad file inside: {bad_file}).")
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise ValueError("Archive is corrupted and could not be opened.")

    image_paths = [
        p
        for p in extract_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError("No supported images (jpg/png/webp/bmp/gif) were found inside the archive.")

    image_paths.sort(key=lambda p: natural_sort_key(str(p.relative_to(extract_dir))))
    return image_paths


def build_pdf_from_images(image_paths: list[Path], output_pdf: Path) -> tuple[int, int]:
    """Convert a sorted list of image paths into a single PDF.

    Returns (pages_written, pages_skipped). Skips unreadable images
    individually rather than failing the whole job, unless literally
    none of them can be read.
    """
    frames: list[Image.Image] = []
    skipped = 0

    try:
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                img.load()  # force read now so we catch errors early
                if img.mode in ("RGBA", "P", "LA"):
                    img = img.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)
            except (UnidentifiedImageError, OSError):
                skipped += 1
                logger.info("Skipping unreadable image: %s", img_path)
                continue

        if not frames:
            raise ValueError("None of the images inside the archive could be read/decoded.")

        first, rest = frames[0], frames[1:]
        first.save(output_pdf, save_all=True, append_images=rest, format="PDF")
        return len(frames), skipped
    finally:
        for f in frames:
            try:
                f.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Telegram handlers
# --------------------------------------------------------------------------

WELCOME_TEXT = (
    "👋 <b>CBZ → PDF Converter Bot</b>\n\n"
    "Send me a <code>.cbz</code> or <code>.zip</code> comic archive and I'll "
    "convert it into a single PDF, page by page.\n\n"
    f"📦 Max file size: {MAX_FILE_SIZE_MB} MB\n"
    "⚙️ I process one file at a time — if I'm busy, you'll be queued "
    "and I'll tell you your position.\n\n"
    "Just send the file to get started!"
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(WELCOME_TEXT)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q_len = job_queue._queue.qsize()
    busy = context.bot_data.get("worker_busy", False)
    if not busy and q_len == 0:
        await update.message.reply_text("✅ I'm idle — send a file and I'll start right away.")
    else:
        await update.message.reply_text(
            f"⏳ Currently processing 1 file, with {q_len} file(s) waiting in queue."
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if doc is None:
        return

    chat_id = update.effective_chat.id
    file_name = doc.file_name or "archive"
    ext = Path(file_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            "❌ Unsupported file type. Please send a <code>.cbz</code> or <code>.zip</code> file.",
            parse_mode=ParseMode.HTML,
        )
        return

    if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"❌ File is too large ({doc.file_size / (1024*1024):.1f} MB). "
            f"Max allowed is {MAX_FILE_SIZE_MB} MB."
        )
        return

    job = ConversionJob(
        job_id=str(uuid.uuid4())[:8],
        chat_id=chat_id,
        file_id=doc.file_id,
        file_name=file_name,
        file_size=doc.file_size or 0,
    )

    position = await job_queue.put(job)

    if position == 1 and not context.bot_data.get("worker_busy", False):
        status_msg = await update.message.reply_text("⏳ Starting conversion shortly...")
    else:
        status_msg = await update.message.reply_text(
            f"📥 Added to the queue.\nCurrent position: <b>#{position}</b>\n"
            "I'll notify you with progress once it's your turn.",
            parse_mode=ParseMode.HTML,
        )

    job.status_message_id = status_msg.message_id


async def handle_wrong_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catches non-document messages (photos sent as images, plain text, etc.)."""
    await update.message.reply_text(
        "📎 Please send your comic archive as a <b>file/document</b> "
        "(<code>.cbz</code> or <code>.zip</code>), not as a photo or text.",
        parse_mode=ParseMode.HTML,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception while processing update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Something went wrong unexpectedly. Please try again.",
            )
        except TelegramError:
            pass


# --------------------------------------------------------------------------
# Background worker (strict 1-task-at-a-time)
# --------------------------------------------------------------------------

async def process_job(job: ConversionJob, context: ContextTypes.DEFAULT_TYPE) -> None:
    work_dir = BASE_TMP_DIR / job.job_id
    extract_dir = work_dir / "extracted"
    archive_path = work_dir / f"input{Path(job.file_name).suffix.lower()}"
    output_pdf = work_dir / f"{Path(job.file_name).stem}.pdf"

    async def update_status(text: str) -> None:
        if job.status_message_id:
            await safe_edit(context, job.chat_id, job.status_message_id, text)

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download
        await update_status("📥 Downloading file...")
        tg_file = await context.bot.get_file(job.file_id)
        await tg_file.download_to_drive(custom_path=str(archive_path))

        # 2. Extract & sort
        await update_status("📂 Extracting and sorting images...")
        image_paths = await asyncio.to_thread(extract_images_from_archive, archive_path, extract_dir)

        # 3. Convert
        await update_status(f"🛠 Converting {len(image_paths)} images to PDF...")
        page_count, skipped = await asyncio.to_thread(build_pdf_from_images, image_paths, output_pdf)
        skip_note = f" ({skipped} unreadable image(s) skipped)" if skipped else ""

        # 4. Upload
        await update_status("📤 Uploading PDF...")
        with open(output_pdf, "rb") as f:
            await context.bot.send_document(
                chat_id=job.chat_id,
                document=f,
                filename=f"{Path(job.file_name).stem}.pdf",
                caption=f"✅ Done! {page_count} page(s) converted{skip_note}.",
            )

        await update_status(f"✅ Conversion complete — {page_count} page(s){skip_note}. PDF sent above.")

    except ValueError as e:
        # Expected/user-facing errors (corrupt archive, no images, etc.)
        logger.info("Job %s failed with user error: %s", job.job_id, e)
        await update_status(f"❌ Conversion failed: {e}")

    except TelegramError as e:
        logger.warning("Job %s failed with Telegram error: %s", job.job_id, e)
        await update_status(f"❌ Telegram error while handling your file: {e}")

    except Exception as e:  # noqa: BLE001 - last-resort safety net per job
        logger.exception("Job %s failed with unexpected error", job.job_id)
        await update_status("❌ An unexpected error occurred while converting your file. Please try again.")

    finally:
        # Always clean up temp files/dirs for this job, success or failure.
        shutil.rmtree(work_dir, ignore_errors=True)


async def notify_queue_positions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """After a job starts/finishes, refresh position messages for those still waiting."""
    async with job_queue._lock:
        pending_snapshot = list(job_queue._pending)

    for idx, waiting_job in enumerate(pending_snapshot, start=1):
        if waiting_job.status_message_id:
            await safe_edit(
                context,
                waiting_job.chat_id,
                waiting_job.status_message_id,
                f"📥 Waiting in queue.\nCurrent position: <b>#{idx}</b>",
            )


async def queue_worker(application: Application) -> None:
    """The single background task that guarantees global concurrency = 1."""
    context = ContextTypes.DEFAULT_TYPE(application=application)
    logger.info("Queue worker started.")
    while True:
        job = await job_queue.get()
        application.bot_data["worker_busy"] = True
        try:
            logger.info("Processing job %s (%s) for chat %s", job.job_id, job.file_name, job.chat_id)
            await process_job(job, context)
        except Exception:
            logger.exception("Uncaught error while processing job %s", job.job_id)
        finally:
            job_queue.task_done()
            application.bot_data["worker_busy"] = False
            await notify_queue_positions(context)


async def post_init(application: Application) -> None:
    BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    application.bot_data["worker_busy"] = False
    # Launch the single global worker task.
    application.create_task(queue_worker(application))
    logger.info("Bot initialized. Temp dir: %s", BASE_TMP_DIR)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit(
            "ERROR: Set the TELEGRAM_BOT_TOKEN environment variable before running this bot.\n"
            "Example: export TELEGRAM_BOT_TOKEN='123456:ABC-your-token'"
        )

    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))

    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(
        MessageHandler(~filters.COMMAND & ~filters.Document.ALL, handle_wrong_type)
    )

    application.add_error_handler(error_handler)

    logger.info("Starting bot (long polling)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
