import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

from app.database import init_db, BASE_DIR, LOGS_DIR, DOWNLOADS_DIR, TEMP_DIR
from app.routes import dashboard, research, videos, voice, content, publishing, settings, auto_edit, pipeline

# Ensure .env is loaded
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("app.main")


from app.services.folder_watcher import FinalFolderWatcher
from app.services.content_service import ContentService
from app.database import SessionLocal


def auto_generate_content_callback(video_id: str):
    """Automatically triggered by Folder Watcher when final video arrives."""
    db = SessionLocal()
    try:
        cs = ContentService()
        cs.generate_content_for_video(db, video_id)
    except Exception as e:
        logger.error(f"Auto content generation failed for {video_id}: {e}")
    finally:
        db.close()


# Global folder watcher instance with automatic content callback
folder_watcher = FinalFolderWatcher(on_ready_callback=auto_generate_content_callback)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure directories and database schema exist
    logger.info("Initializing AI Content Factory local environment...")
    init_db()
    try:
        folder_watcher.start()
    except Exception as e:
        logger.error(f"Failed to start folder watcher: {e}")
    logger.info("AI Content Factory initialized successfully.")
    yield
    # Shutdown
    logger.info("Shutting down AI Content Factory...")
    try:
        folder_watcher.stop()
    except Exception as e:
        logger.error(f"Failed to stop folder watcher cleanly: {e}")


app = FastAPI(
    title="AI Content Factory",
    description="Local-first automated content production management system",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static and download folders
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/downloads", StaticFiles(directory=str(DOWNLOADS_DIR)), name="downloads")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")

# Include Routers
app.include_router(dashboard.router)
app.include_router(research.router)
app.include_router(videos.router)
app.include_router(voice.router)
app.include_router(auto_edit.router)
app.include_router(content.router)
app.include_router(publishing.router)
app.include_router(settings.router)
app.include_router(pipeline.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "AI Content Factory", "version": "1.0.0"}
