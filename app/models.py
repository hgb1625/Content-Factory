from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(32), unique=True, nullable=False, index=True)  # P0001, P0002...
    niche = Column(String(255), nullable=False)
    name_vietnamese = Column(String(255), nullable=False)
    name_chinese = Column(String(255), nullable=True)
    douyin_keywords = Column(Text, nullable=True)
    content_angle = Column(Text, nullable=True)
    hook = Column(Text, nullable=True)
    status = Column(String(64), default="NEW", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    videos = relationship("Video", back_populates="product")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(32), unique=True, nullable=False, index=True)  # V0001, V0002...
    product_id = Column(String(32), ForeignKey("products.product_id", ondelete="SET NULL"), nullable=True, index=True)
    douyin_url = Column(String(1024), unique=True, nullable=False, index=True)
    views = Column(String(64), nullable=True)  # Optional
    thumbnail = Column(Text, nullable=True)    # Optional
    local_file = Column(Text, nullable=True)
    downloaded = Column(Boolean, default=False)
    approved = Column(Boolean, default=False)
    used = Column(Boolean, default=False)
    status = Column(String(64), default="FOUND", index=True)
    provider = Column(String(64), default="douyin", index=True)
    canonical_source_id = Column(String(255), nullable=True, index=True)
    media_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Phase 12 Auto Edit fields
    edit_mode = Column(String(32), default="AUTO_EDIT")  # AUTO_EDIT or CAPCUT_MANUAL
    auto_edit_status = Column(String(64), nullable=True)
    subtitle_region = Column(Text, nullable=True)  # JSON string of normalized {x, y, width, height, mode}
    edit_plan_path = Column(Text, nullable=True)
    final_video_path = Column(Text, nullable=True)
    render_error = Column(Text, nullable=True)

    product = relationship("Product", back_populates="videos")
    voices = relationship("Voice", back_populates="video", cascade="all, delete-orphan")
    content = relationship("Content", back_populates="video", uselist=False, cascade="all, delete-orphan")
    publishing = relationship("Publishing", back_populates="video", uselist=False, cascade="all, delete-orphan")


class Voice(Base):
    __tablename__ = "voices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(32), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False, index=True)
    script = Column(Text, nullable=False)
    tts_engine = Column(String(64), default="VieNeu-TTS")
    voice_name = Column(String(64), nullable=True)
    audio_file = Column(Text, nullable=True)
    status = Column(String(64), default="PENDING", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    video = relationship("Video", back_populates="voices")


class Content(Base):
    __tablename__ = "content"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(32), ForeignKey("videos.video_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    facebook_personal_caption = Column(Text, nullable=True)
    facebook_personal_hashtags = Column(Text, nullable=True)

    facebook_page_caption = Column(Text, nullable=True)
    facebook_page_hashtags = Column(Text, nullable=True)

    tiktok_caption = Column(Text, nullable=True)
    tiktok_hashtags = Column(Text, nullable=True)

    threads_caption = Column(Text, nullable=True)
    threads_hashtags = Column(Text, nullable=True)

    instagram_caption = Column(Text, nullable=True)
    instagram_hashtags = Column(Text, nullable=True)

    shopee_caption = Column(Text, nullable=True)
    shopee_hashtags = Column(Text, nullable=True)

    youtube_title = Column(Text, nullable=True)
    youtube_description = Column(Text, nullable=True)
    youtube_hashtags = Column(Text, nullable=True)

    status = Column(String(64), default="PENDING", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="content")


class Publishing(Base):
    __tablename__ = "publishing"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(32), ForeignKey("videos.video_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    facebook_personal = Column(Boolean, default=False)
    facebook_page = Column(Boolean, default=False)
    tiktok = Column(Boolean, default=False)
    threads = Column(Boolean, default=False)
    instagram = Column(Boolean, default=False)
    shopee = Column(Boolean, default=False)
    youtube = Column(Boolean, default=False)

    published_count = Column(Integer, default=0)
    status = Column(String(64), default="NOT PUBLISHED", index=True)  # NOT PUBLISHED, PARTIAL, COMPLETED
    publish_date = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    # Automated publishing and scheduling fields
    scheduled_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    publish_status = Column(String(64), default="NOT_PUBLISHED", index=True)
    platform_post_id = Column(Text, nullable=True)  # JSON-encoded map of platform -> post_id / URL
    error_message = Column(Text, nullable=True)

    video = relationship("Video", back_populates="publishing")


class SocialAccount(Base):
    """
    Connected social media accounts metadata.
    Sensitive access tokens and client secrets are NEVER stored in plaintext in SQLite;
    they are kept securely in environment variables / .env configuration.
    """
    __tablename__ = "social_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(64), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    account_id = Column(String(255), nullable=True)
    is_connected = Column(Boolean, default=False)
    status = Column(String(64), default="NOT_CONNECTED")  # READY, NOT_CONNECTED, MANUAL_REQUIRED, FAILED
    last_checked_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)


class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)

