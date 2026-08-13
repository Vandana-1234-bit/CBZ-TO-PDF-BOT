#!/usr/bin/env python3
"""
CBZ/ZIP -> PDF Telegram Bot (Pyrogram / MTProto edition)
=========================================================

Same feature set as the python-telegram-bot version, but built on
**Pyrogram**, which talks to Telegram directly over MTProto instead of
going through the regular Bot API HTTP server. This removes the Bot API's
20 MB download limit — with Pyrogram a bot can download/upload files up
to **2 GB** (4 GB for Telegram Premium accounts) with zero extra setup.

Key features
------------
- Accepts .cbz / .zip files up to 200 MB (configurable, well under
  Pyrogram's 2 GB ceiling).
- Extracts images, sorts them naturally (page2 < page10), stitches them
  into one PDF using Pillow.
- STRICT global concurrency limit of 1: only one conversion job runs at a
  time across the whole bot, via asyncio.Queue + one background worker.
- Live queue-position messages, updated as the queue moves.
- Real-time progress: Downloading -> Extracting -> Converting -> Uploading -> Done.
- All temp files/directories are removed after each job (success or
  failure) via try/finally.
- Graceful error handling for corrupted archives, unreadable images,
  oversized files, and Telegram/network errors.

Requirements
------------
    pip install pyrogram tgcrypto Pillow

You need THREE credentials (all free):
    1. API_ID    - from https://my.telegram.org  (API Development Tools)
    2. API_HASH  - from https://my.telegram.org  (API Development Tools)
    3. BOT_TOKEN - from @BotFather on Telegram

Set them as environment variables before running:
    export API_ID="12345678"
    export API_HASH="abcdef1234567890abcdef1234567890"
    export BOT_TOKEN="123456:ABC-your-token-here"
    python main.py

Tech stack: Python 3.10+, Pyrogram (MTProto), tgcrypto (fast crypto),
Pillow, zipfile, asyncio.Queue + a single background worker task.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.errors import RPCError

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_ID = os.environ.get("API_ID", "")
API_HASH = os.environ.get("API_HASH", "")
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Safety limits - tune as needed for your server.
# Pyrogram (MTProto) supports up to 2 GB (4 GB for Premium), but keep this
# reasonable for your server's RAM/disk/CPU.
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
logging.getLogger("pyrogram").setLevel(logging.WARNING)  # quiet Pyrogram's internal logs


# --------------------------------------------------------------------------
# Queue job model
# --------------------------------------------------------------------------

@dataclass
class ConversionJob:
    job_id: str
    chat_id: int
    message: Message           # the original message containing the document
    file_name: str
    file_size: int
    status_message: Optional[Message] = None


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

    def qsize(self) -> int:
        return self._queue.qsize()

    async def snapshot(self) -> list[ConversionJob]:
        async with self._lock:
            return list(self._pending)


job_queue = JobQueue()
worker_busy = False  # simple global flag, only touched by the single worker


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


async def safe_edit(message: Optional[Message], text: str) -> None:
    """Edit a status message, ignoring 'message not modified' / harmless errors."""
    if message is None:
        return
    try:
        await message.edit_text(text)
    except RPCError as e:
        if "MESSAGE_NOT_MODIFIED" not in str(e):
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
                if img.mode != "RGB":
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
# Pyrogram client
# --------------------------------------------------------------------------

app = Client(
    "cbz2pdf_bot_session",
    api_id=int(API_ID) if API_ID else None,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
)

WELCOME_TEXT = (
    "👋 **CBZ → PDF Converter Bot**\n\n"
    "Send me a `.cbz` or `.zip` comic archive and I'll convert it into a "
    "single PDF, page by page.\n\n"
    f"📦 Max file size: {MAX_FILE_SIZE_MB} MB\n"
    "⚙️ I process one file at a time — if I'm busy, you'll be queued "
    "and I'll tell you your position.\n\n"
    "Just send the file to get started!"
)


@app.on_message(filters.command(["start", "help"]))
async def cmd_start(client: Client, message: Message) -> None:
    await message.reply_text(WELCOME_TEXT)


@app.on_message(filters.command("status"))
async def cmd_status(client: Client, message: Message) -> None:
    q_len = job_queue.qsize()
    if not worker_busy and q_len == 0:
        await message.reply_text("✅ I'm idle — send a file and I'll start right away.")
    else:
        await message.reply_text(
            f"⏳ Currently processing 1 file, with {q_len} file(s) waiting in queue."
        )


@app.on_message(filters.document)
async def handle_document(client: Client, message: Message) -> None:
    doc = message.document
    file_name = doc.file_name or "archive"
    ext = Path(file_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        await message.reply_text(
            "❌ Unsupported file type. Please send a `.cbz` or `.zip` file."
        )
        return

    if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
        await message.reply_text(
            f"❌ File is too large ({doc.file_size / (1024*1024):.1f} MB). "
            f"Max allowed is {MAX_FILE_SIZE_MB} MB."
        )
        return

    job = ConversionJob(
        job_id=str(uuid.uuid4())[:8],
        chat_id=message.chat.id,
        message=message,
        file_name=file_name,
        file_size=doc.file_size or 0,
    )

    position = await job_queue.put(job)

    if position == 1 and not worker_busy:
        status_msg = await message.reply_text("⏳ Starting conversion shortly...")
    else:
        status_msg = await message.reply_text(
            f"📥 Added to the queue.\nCurrent position: **#{position}**\n"
            "I'll notify you with progress once it's your turn."
        )

    job.status_message = status_msg


@app.on_message(filters.text & ~filters.command(["start", "help", "status"]) | filters.photo | filters.video)
async def handle_wrong_type(client: Client, message: Message) -> None:
    await message.reply_text(
        "📎 Please send your comic archive as a **file/document** "
        "(`.cbz` or `.zip`), not as a photo, video, or plain text."
    )


# --------------------------------------------------------------------------
# Background worker (strict 1-task-at-a-time)
# --------------------------------------------------------------------------

async def process_job(job: ConversionJob) -> None:
    work_dir = BASE_TMP_DIR / job.job_id
    extract_dir = work_dir / "extracted"
    archive_path = work_dir / f"input{Path(job.file_name).suffix.lower()}"
    output_pdf = work_dir / f"{Path(job.file_name).stem}.pdf"

    async def update_status(text: str) -> None:
        await safe_edit(job.status_message, text)

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download (MTProto -> up to 2GB, no Bot API 20MB cap)
        await update_status("📥 Downloading file...")
        await job.message.download(file_name=str(archive_path))

        # 2. Extract & sort
        await update_status("📂 Extracting and sorting images...")
        image_paths = await asyncio.to_thread(extract_images_from_archive, archive_path, extract_dir)

        # 3. Convert
        await update_status(f"🛠 Converting {len(image_paths)} images to PDF...")
        page_count, skipped = await asyncio.to_thread(build_pdf_from_images, image_paths, output_pdf)
        skip_note = f" ({skipped} unreadable image(s) skipped)" if skipped else ""

        # 4. Upload
        await update_status("📤 Uploading PDF...")
        await job.message.reply_document(
            document=str(output_pdf),
            file_name=f"{Path(job.file_name).stem}.pdf",
            caption=f"✅ Done! {page_count} page(s) converted{skip_note}.",
        )

        await update_status(f"✅ Conversion complete — {page_count} page(s){skip_note}. PDF sent above.")

    except ValueError as e:
        # Expected/user-facing errors (corrupt archive, no images, etc.)
        logger.info("Job %s failed with user error: %s", job.job_id, e)
        await update_status(f"❌ Conversion failed: {e}")

    except RPCError as e:
        logger.warning("Job %s failed with Telegram error: %s", job.job_id, e)
        await update_status(f"❌ Telegram error while handling your file: {e}")

    except Exception:  # noqa: BLE001 - last-resort safety net per job
        logger.exception("Job %s failed with unexpected error", job.job_id)
        await update_status("❌ An unexpected error occurred while converting your file. Please try again.")

    finally:
        # Always clean up temp files/dirs for this job, success or failure.
        shutil.rmtree(work_dir, ignore_errors=True)


async def notify_queue_positions() -> None:
    """After a job starts/finishes, refresh position messages for those still waiting."""
    pending_snapshot = await job_queue.snapshot()
    for idx, waiting_job in enumerate(pending_snapshot, start=1):
        await safe_edit(
            waiting_job.status_message,
            f"📥 Waiting in queue.\nCurrent position: **#{idx}**",
        )


async def queue_worker() -> None:
    """The single background task that guarantees global concurrency = 1."""
    global worker_busy
    logger.info("Queue worker started.")
    while True:
        job = await job_queue.get()
        worker_busy = True
        try:
            logger.info("Processing job %s (%s) for chat %s", job.job_id, job.file_name, job.chat_id)
            await process_job(job)
        except Exception:
            logger.exception("Uncaught error while processing job %s", job.job_id)
        finally:
            job_queue.task_done()
            worker_busy = False
            await notify_queue_positions()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

async def main() -> None:
    missing = [name for name, val in (("API_ID", API_ID), ("API_HASH", API_HASH), ("BOT_TOKEN", BOT_TOKEN)) if not val]
    if missing:
        raise SystemExit(
            f"ERROR: Missing required environment variable(s): {', '.join(missing)}.\n"
            "Get API_ID / API_HASH from https://my.telegram.org and BOT_TOKEN from @BotFather.\n"
            "Example:\n"
            "  export API_ID='12345678'\n"
            "  export API_HASH='abcdef1234567890abcdef1234567890'\n"
            "  export BOT_TOKEN='123456:ABC-your-token'\n"
        )

    BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)

    await app.start()
    logger.info("Bot started. Temp dir: %s", BASE_TMP_DIR)

    worker_task = asyncio.create_task(queue_worker())

    logger.info("Bot is up and polling for messages...")
    try:
        await asyncio.Event().wait()  # run forever
    finally:
        worker_task.cancel()
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
