import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Required directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DOWNLOADS_DIR = BASE_DIR / "downloads"
ORIGINAL_DIR = DOWNLOADS_DIR / "original"
PACKAGES_DIR = DOWNLOADS_DIR / "packages"
FINAL_DIR = DOWNLOADS_DIR / "final"
TEMP_DIR = BASE_DIR / "temp"

REQUIRED_DIRS = [DATA_DIR, LOGS_DIR, DOWNLOADS_DIR, ORIGINAL_DIR, PACKAGES_DIR, FINAL_DIR, TEMP_DIR]

DB_PATH = DATA_DIR / "content.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create SQLite engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_directories():
    """Ensure all required project storage folders exist."""
    for folder in REQUIRED_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def get_db():
    """Dependency for providing database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_schema(target_engine=None):
    """Non-destructively check and add new columns to existing SQLite tables."""
    eng = target_engine or engine
    try:
        with eng.connect() as conn:
            # Check videos table columns
            res = conn.execute(text("PRAGMA table_info(videos)")).fetchall()
            existing_cols = {r[1] for r in res}

            video_columns = [
                ("provider", "VARCHAR(64) DEFAULT 'douyin'"),
                ("canonical_source_id", "VARCHAR(255)"),
                ("media_hash", "VARCHAR(64)"),
                ("edit_mode", "VARCHAR(32) DEFAULT 'AUTO_EDIT'"),
                ("auto_edit_status", "VARCHAR(64)"),
                ("subtitle_region", "TEXT"),
                ("edit_plan_path", "TEXT"),
                ("final_video_path", "TEXT"),
                ("render_error", "TEXT")
            ]

            for col_name, col_type in video_columns:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}"))
                    conn.commit()

            # Ensure indices exist
            index_statements = [
                "CREATE INDEX IF NOT EXISTS ix_videos_provider ON videos(provider)",
                "CREATE INDEX IF NOT EXISTS ix_videos_canonical_source_id ON videos(canonical_source_id)",
                "CREATE INDEX IF NOT EXISTS ix_videos_media_hash ON videos(media_hash)",
            ]
            for idx_stmt in index_statements:
                try:
                    conn.execute(text(idx_stmt))
                    conn.commit()
                except Exception:
                    pass

            # Check publishing table columns
            res_pub = conn.execute(text("PRAGMA table_info(publishing)")).fetchall()
            existing_pub_cols = {r[1] for r in res_pub}

            pub_columns = [
                ("scheduled_at", "DATETIME"),
                ("published_at", "DATETIME"),
                ("publish_status", "VARCHAR(64) DEFAULT 'NOT_PUBLISHED'"),
                ("platform_post_id", "TEXT"),
                ("error_message", "TEXT")
            ]

            for col_name, col_type in pub_columns:
                if col_name not in existing_pub_cols:
                    conn.execute(text(f"ALTER TABLE publishing ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
    except Exception as e:
        # If table doesn't exist yet, create_all will create it with all columns
        pass


def init_db():
    """Initialize database tables and directories."""
    ensure_directories()
    import app.models  # noqa: F401 - ensure models are imported
    Base.metadata.create_all(bind=engine)
    migrate_schema()
