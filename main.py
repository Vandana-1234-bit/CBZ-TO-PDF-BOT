#!/usr/bin/env python3
"""
CBZ/ZIP -> PDF Telegram Bot (Telethon / MTProto edition)
==========================================================

Same feature set as before, but built on **Telethon**, which talks to
Telegram directly over MTProto instead of going through the regular Bot
API HTTP server. This removes the Bot API's 20 MB download limit — with
Telethon a bot can download/upload files up to **2 GB** (4 GB for
Telegram Premium accounts) with zero extra setup.

Key features
------------
- Accepts .cbz / .zip files up to 200 MB (configurable, well under
  Telethon's 2 GB ceiling).
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
    pip install telethon Pillow

You need THREE credentials (all free):
    1. API_ID    - from https://my.telegram.org  (API Development Tools)
    2. API_HASH  - from https://my.telegram.org  (API Development Tools)
    3. BOT_TOKEN - from @BotFather on Telegram

Set them as environment variables before running:
    export API_ID="12345678"
    export API_HASH="abcdef1234567890abcdef1234567890"
    export BOT_TOKEN="123456:ABC-your-token-here"
    python main.py

Tech stack: Python 3.10+, Telethon (MTProto), Pillow, zipfile,
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
import img2pdf
from telethon import TelegramClient, events
from telethon.tl.custom.message import Message
from telethon.errors import RPCError, MessageNotModifiedError

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_ID = os.environ.get("API_ID", "")
API_HASH = os.environ.get("API_HASH", "")
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Safety limits - tune as needed for your server.
# Telethon (MTProto) supports up to 2 GB (4 GB for Premium), but keep this
# reasonable for your server's RAM/disk/CPU.
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".cbz", ".zip"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Base directory for all temporary work. Each job gets its own uuid subdir.
BASE_TMP_DIR = Path(tempfile.gettempdir()) / "cbz2pdf_bot"

# Hard timeouts so a stalled network call can never hang the bot forever.
# Without these, one bad download/upload would freeze the whole queue since
# only one job is processed at a time.
DOWNLOAD_TIMEOUT_SECONDS = 600   # 10 minutes
UPLOAD_TIMEOUT_SECONDS = 600     # 10 minutes

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cbz2pdf_bot")
logging.getLogger("telethon").setLevel(logging.WARNING)  # quiet Telethon's internal logs


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

    async def put(self, job: ConversionJob, currently_busy: bool = False) -> int:
        async with self._lock:
            self._pending.append(job)
            # +1 extra if a job is already being actively processed right
            # now (that job is not in _pending since the worker already
            # dequeued it), so the displayed position accounts for it too.
            position = len(self._pending) + (1 if currently_busy else 0)
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
        await message.edit(text)
    except MessageNotModifiedError:
        pass
    except RPCError as e:
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


def build_pdf_from_images(image_paths: list[Path], output_pdf: Path, work_dir: Path) -> tuple[int, int]:
    """Convert a sorted list of image paths into a single PDF.

    Uses img2pdf, which streams/embeds image bytes directly into the PDF
    without decoding full-resolution pixel data for every page at once.
    This keeps memory usage low even for archives with 150-300+ pages,
    unlike loading every page as a decoded Pillow Image simultaneously.

    Returns (pages_written, pages_skipped). Skips unreadable/unsupported
    images individually rather than failing the whole job, unless
    literally none of them can be used.
    """
    # img2pdf natively supports JPEG, PNG, TIFF. Anything else (webp, bmp,
    # gif, etc.) gets converted to PNG on disk first, one image at a time,
    # so we never hold more than one full decoded image in memory.
    NATIVE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    convert_dir = work_dir / "converted"
    convert_dir.mkdir(parents=True, exist_ok=True)

    usable_paths: list[str] = []
    skipped = 0

    for idx, img_path in enumerate(image_paths):
        try:
            if img_path.suffix.lower() in NATIVE_EXTS:
                # Quick sanity check that it actually opens before trusting it.
                with Image.open(img_path) as im:
                    im.verify()
                usable_paths.append(str(img_path))
            else:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    converted_path = convert_dir / f"{idx:05d}.png"
                    im.save(converted_path, format="PNG")
                usable_paths.append(str(converted_path))
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
            logger.info("Skipping unreadable image: %s", img_path)
            continue

    if not usable_paths:
        raise ValueError("None of the images inside the archive could be read/decoded.")

    try:
        pdf_bytes = img2pdf.convert(usable_paths)
    except Exception as e:
        raise ValueError(f"Failed to assemble PDF from images: {e}")

    output_pdf.write_bytes(pdf_bytes)
    return len(usable_paths), skipped


# --------------------------------------------------------------------------
# Telethon client
# --------------------------------------------------------------------------

client = TelegramClient(
    "cbz2pdf_bot_session",
    api_id=int(API_ID) if API_ID else None,
    api_hash=API_HASH,
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


@client.on(events.NewMessage(pattern="/start"))
@client.on(events.NewMessage(pattern="/help"))
async def cmd_start(event: events.NewMessage.Event) -> None:
    await event.reply(WELCOME_TEXT)


@client.on(events.NewMessage(pattern="/status"))
async def cmd_status(event: events.NewMessage.Event) -> None:
    q_len = job_queue.qsize()
    if not worker_busy and q_len == 0:
        await event.reply("✅ I'm idle — send a file and I'll start right away.")
    else:
        await event.reply(
            f"⏳ Currently processing 1 file, with {q_len} file(s) waiting in queue."
        )


@client.on(events.NewMessage(func=lambda e: e.document is not None))
async def handle_document(event: events.NewMessage.Event) -> None:
    message: Message = event.message
    doc = message.document

    # Extract the original filename from document attributes.
    file_name = "archive"
    for attr in doc.attributes:
        if hasattr(attr, "file_name") and attr.file_name:
            file_name = attr.file_name
            break

    ext = Path(file_name).suffix.lower()

    # Some clients (especially forwarded messages from channels) don't
    # attach a proper filename attribute, so `ext` can end up empty/wrong.
    # Fall back to the document's mime_type in that case before rejecting.
    if ext not in ALLOWED_EXTENSIONS:
        mime = (doc.mime_type or "").lower()
        mime_to_ext = {
            "application/zip": ".zip",
            "application/x-zip-compressed": ".zip",
            "application/x-cbz": ".cbz",
            "application/vnd.comicbook+zip": ".cbz",
            "application/octet-stream": None,  # ambiguous, handled below
        }
        guessed_ext = mime_to_ext.get(mime)
        if guessed_ext:
            ext = guessed_ext
            if file_name == "archive" or not file_name.lower().endswith(ext):
                file_name = f"{file_name}{ext}" if "." not in file_name else file_name
        elif mime == "application/octet-stream" and file_name != "archive":
            # Generic binary mime + a filename that just lacks the right
            # suffix (common with forwarded .cbz files) -> trust an
            # extensionless name only if nothing better is available.
            pass

    if ext not in ALLOWED_EXTENSIONS:
        logger.info(
            "Rejected document: filename=%r mime_type=%r attrs=%r",
            file_name, doc.mime_type, [type(a).__name__ for a in doc.attributes],
        )
        await event.reply(
            "❌ Unsupported file type (or I couldn't detect the file name/type). "
            "Please send a `.cbz` or `.zip` file — if you forwarded this from a "
            "channel, try downloading it and re-uploading it directly instead."
        )
        return

    file_size = doc.size or 0
    if file_size and file_size > MAX_FILE_SIZE_BYTES:
        await event.reply(
            f"❌ File is too large ({file_size / (1024*1024):.1f} MB). "
            f"Max allowed is {MAX_FILE_SIZE_MB} MB."
        )
        return

    job = ConversionJob(
        job_id=str(uuid.uuid4())[:8],
        chat_id=event.chat_id,
        message=message,
        file_name=file_name,
        file_size=file_size,
    )

    # IMPORTANT: create + attach the status message BEFORE the job is put
    # on the queue. If we did it after `put()`, the worker could pick the
    # job up immediately (when idle) and start editing job.status_message
    # while it's still None, silently dropping the first progress updates.
    will_start_immediately = job_queue.qsize() == 0 and not worker_busy
    if will_start_immediately:
        status_msg = await event.reply("⏳ Starting conversion shortly...")
    else:
        status_msg = await event.reply("📥 Added to the queue.\nFiguring out your position...")
    job.status_message = status_msg

    position = await job_queue.put(job, currently_busy=worker_busy)

    if not will_start_immediately:
        await safe_edit(
            status_msg,
            f"📥 Added to the queue.\nCurrent position: **#{position}**\n"
            "I'll notify you with progress once it's your turn.",
        )

    logger.info(
        "Queued job %s (%s, %.1f MB) at position %s",
        job.job_id, file_name, job.file_size / (1024 * 1024), position,
    )


@client.on(events.NewMessage(func=lambda e: (
    e.document is None
    and not (e.raw_text or "").startswith(("/start", "/help", "/status"))
)))
async def handle_wrong_type(event: events.NewMessage.Event) -> None:
    await event.reply(
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
        await update_status("📥 Downloading file... (0%)")
        logger.info("Job %s: download starting", job.job_id)

        last_reported = -1
        main_loop = asyncio.get_running_loop()

        def dl_progress(current: int, total: int) -> None:
            nonlocal last_reported
            pct = int(current * 100 / total) if total else 0
            if pct != last_reported and pct % 10 == 0:
                last_reported = pct
                # progress_callback runs in the same event loop for Telethon,
                # but schedule safely regardless.
                asyncio.run_coroutine_threadsafe(
                    update_status(f"📥 Downloading file... ({pct}%)"), main_loop
                )

        try:
            await asyncio.wait_for(
                client.download_media(job.message, file=str(archive_path), progress_callback=dl_progress),
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise ValueError(
                f"Download timed out after {DOWNLOAD_TIMEOUT_SECONDS // 60} minutes. "
                "This is usually a network issue on the server — please try again."
            )

        if not archive_path.exists():
            raise ValueError("Download finished but the file could not be found on disk.")

        logger.info("Job %s: download finished (%.1f MB)", job.job_id, archive_path.stat().st_size / (1024*1024))

        # 2. Extract & sort
        await update_status("📂 Extracting and sorting images...")
        image_paths = await asyncio.to_thread(extract_images_from_archive, archive_path, extract_dir)
        logger.info("Job %s: found %d images", job.job_id, len(image_paths))

        # 3. Convert
        await update_status(f"🛠 Converting {len(image_paths)} images to PDF...")
        page_count, skipped = await asyncio.to_thread(build_pdf_from_images, image_paths, output_pdf, work_dir)
        skip_note = f" ({skipped} unreadable image(s) skipped)" if skipped else ""
        logger.info("Job %s: PDF built with %d pages (%d skipped)", job.job_id, page_count, skipped)

        # 4. Upload
        await update_status("📤 Uploading PDF...")
        try:
            await asyncio.wait_for(
                client.send_file(
                    job.chat_id,
                    str(output_pdf),
                    caption=f"✅ Done! {page_count} page(s) converted{skip_note}.",
                    force_document=True,
                    file_name=f"{Path(job.file_name).stem}.pdf",
                    reply_to=job.message.id,
                ),
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise ValueError(
                f"Upload timed out after {UPLOAD_TIMEOUT_SECONDS // 60} minutes. "
                "This is usually a network issue on the server — please try again."
            )

        await update_status(f"✅ Conversion complete — {page_count} page(s){skip_note}. PDF sent above.")
        logger.info("Job %s: completed successfully", job.job_id)

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

    await client.start(bot_token=BOT_TOKEN)
    logger.info("Bot started. Temp dir: %s", BASE_TMP_DIR)

    asyncio.create_task(queue_worker())

    logger.info("Bot is up and polling for messages...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
