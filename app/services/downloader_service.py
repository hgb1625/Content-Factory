import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import hashlib
import httpx
from sqlalchemy.orm import Session

from app.models import Video
from app.database import ORIGINAL_DIR
from app.services.douyin_service import (
    normalize_douyin_url,
    extract_douyin_video_id,
    get_canonical_douyin_url,
    get_next_video_id
)
from app.services.ffmpeg_utils import run_ffprobe

logger = logging.getLogger("app.services.downloader")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate cryptographic SHA-256 hash of a media file."""
    if not file_path.is_file():
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_downloaded_media(file_path: Path) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate downloaded video file using ffprobe:
    - valid media container
    - duration > 0
    - video stream exists with width > 0 and height > 0
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False, {}, "File không tồn tại hoặc rỗng (0 bytes)."

    try:
        cmd = [
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        proc = run_ffprobe(cmd, timeout=30.0)
        if proc.returncode != 0:
            return False, {}, f"ffprobe lỗi (mã {proc.returncode}): {proc.stderr.strip()}"

        info = json.loads(proc.stdout)
        fmt = info.get("format", {})
        duration = float(fmt.get("duration", 0))
        if duration <= 0:
            return False, {}, "Thời lượng video bằng 0 hoặc không hợp lệ."

        streams = info.get("streams", [])
        has_video = False
        width, height = 0, 0
        video_codec = ""
        audio_codec = ""

        for s in streams:
            if s.get("codec_type") == "video":
                has_video = True
                width = int(s.get("width", 0))
                height = int(s.get("height", 0))
                video_codec = s.get("codec_name", "")
            elif s.get("codec_type") == "audio":
                audio_codec = s.get("codec_name", "")

        if not has_video or width <= 0 or height <= 0:
            return False, {}, "Không tìm thấy luồng video hợp lệ hoặc độ phân giải không hợp lệ."

        meta = {
            "duration": round(duration, 2),
            "width": width,
            "height": height,
            "video_codec": video_codec,
            "audio_codec": audio_codec,
            "file_size": file_path.stat().st_size
        }
        return True, meta, None
    except Exception as e:
        return False, {}, f"Lỗi kiểm tra ffprobe: {e}"


class SnapTikTokDownloader:
    """
    Provider for resolving and downloading Douyin videos via SnapTikTok.
    Handles session initialization and parsing of direct CDN / proxy URLs.
    """
    PAGE_URL = "https://snaptiktok.to/vi/douyin-downloader"
    SEARCH_API = "https://snaptiktok.to/api/ajaxSearch"

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Referer": "https://snaptiktok.to/vi/douyin-downloader",
        "Origin": "https://snaptiktok.to",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "*/*",
    }

    def resolve_video_url(self, canonical_url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Connect to SnapTikTok and resolve direct MP4 download link.
        Flow:
        - GET SnapTikTok page first to establish session and collect cookies
        - POST to /api/ajaxSearch with canonical URL
        - Parse download links (direct CDN or snapcdn.app/dl)
        Returns: (success, media_url, error_message)
        """
        import subprocess
        import tempfile

        cookie_file = Path(tempfile.gettempdir()) / f"snaptiktok_sess_{os.getpid()}.txt"

        # 1. Native curl.exe with HTTP/1.1 and TLS 1.2 (optimal compatibility with Cloudflare)
        try:
            # Step 1: GET SnapTikTok page to establish session/cookies
            cmd_get = [
                "curl.exe", "--http1.1", "--tlsv1.2", "-s", "--connect-timeout", "6", "--max-time", "15",
                "--resolve", "snaptiktok.to:443:172.67.128.40",
                "-c", str(cookie_file),
                "-H", f"User-Agent: {self.DEFAULT_HEADERS['User-Agent']}",
                "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                self.PAGE_URL
            ]
            subprocess.run(cmd_get, capture_output=True, timeout=20.0)

            # Step 2: POST /api/ajaxSearch with cookies
            cmd_post = [
                "curl.exe", "--http1.1", "--tlsv1.2", "-s", "--connect-timeout", "6", "--max-time", "15",
                "--resolve", "snaptiktok.to:443:172.67.128.40",
                "-b", str(cookie_file),
                "-c", str(cookie_file),
                "-X", "POST", self.SEARCH_API,
                "-H", f"User-Agent: {self.DEFAULT_HEADERS['User-Agent']}",
                "-H", f"Referer: {self.PAGE_URL}",
                "-H", f"Origin: {self.DEFAULT_HEADERS['Origin']}",
                "-H", "X-Requested-With: XMLHttpRequest",
                "-H", "Content-Type: application/x-www-form-urlencoded; charset=UTF-8",
                "--data-urlencode", f"q={canonical_url}",
                "--data-urlencode", "cursor=0",
                "--data-urlencode", "page=0",
                "--data-urlencode", "lang=vi"
            ]
            proc = subprocess.run(cmd_post, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20.0)

            # Fallback attempt without --resolve if needed
            if proc.returncode != 0 or not proc.stdout.strip():
                cmd_post_fallback = [arg for arg in cmd_post if arg != "--resolve" and arg != "snaptiktok.to:443:172.67.128.40"]
                proc = subprocess.run(cmd_post_fallback, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20.0)

            if proc.returncode == 0 and proc.stdout.strip():
                try:
                    data = json.loads(proc.stdout)
                    if data.get("status") == "ok" and data.get("data"):
                        html_snippet = data["data"]
                        links = re.findall(r'href="([^"]+)"', html_snippet)
                        chosen_link = None
                        # Prioritize direct Douyin CDN link, then snapcdn proxy, then any mp4/video link
                        for l in links:
                            clean_l = l.replace("&amp;", "&")
                            if "zjcdn.com" in clean_l or "douyinvod.com" in clean_l:
                                chosen_link = clean_l
                                break
                            elif "snapcdn.app/dl" in clean_l:
                                if not chosen_link:
                                    chosen_link = clean_l
                            elif "mp4" in clean_l or "video" in clean_l:
                                if not chosen_link:
                                    chosen_link = clean_l

                        if chosen_link:
                            return True, chosen_link, None
                        return False, None, "Không tìm thấy link tải MP4 trong phản hồi của SnapTikTok"
                    elif data.get("msg"):
                        return False, None, data.get("msg")
                except json.JSONDecodeError:
                    pass
        except Exception as ce:
            logger.debug(f"curl.exe resolver attempt encountered: {ce}")
        finally:
            if cookie_file.exists():
                try:
                    cookie_file.unlink()
                except Exception:
                    pass

        # 2. Fallback via httpx
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                try:
                    r_init = client.get(self.PAGE_URL, headers=self.DEFAULT_HEADERS)
                    if r_init.status_code != 200:
                        return False, None, f"Không thể kết nối SnapTikTok (HTTP {r_init.status_code})"
                except Exception as he:
                    return False, None, f"Lỗi kết nối trang SnapTikTok: {he}"

                payload = {
                    "q": canonical_url,
                    "cursor": "0",
                    "page": "0",
                    "lang": "vi"
                }
                r_search = client.post(self.SEARCH_API, headers=self.DEFAULT_HEADERS, data=payload)
                if r_search.status_code != 200:
                    return False, None, f"SnapTikTok API trả về HTTP {r_search.status_code}"

                data = r_search.json()
                if data.get("status") != "ok" or not data.get("data"):
                    msg = data.get("msg") or "SnapTikTok không tìm thấy dữ liệu video"
                    return False, None, msg

                html_snippet = data["data"]
                links = re.findall(r'href="([^"]+)"', html_snippet)
                chosen_link = None
                for l in links:
                    clean_l = l.replace("&amp;", "&")
                    if "zjcdn.com" in clean_l or "douyinvod.com" in clean_l:
                        chosen_link = clean_l
                        break
                    elif "snapcdn.app/dl" in clean_l:
                        if not chosen_link:
                            chosen_link = clean_l
                    elif "mp4" in clean_l or "video" in clean_l:
                        if not chosen_link:
                            chosen_link = clean_l

                if not chosen_link:
                    return False, None, "Không tìm thấy đường link tải MP4 trong phản hồi của SnapTikTok"

                return True, chosen_link, None
        except Exception as e:
            return False, None, f"Lỗi phân giải SnapTikTok: {e}"

    def download_file(self, media_url: str, target_path: Path, timeout: float = 60.0) -> Tuple[bool, Optional[str]]:
        """
        Safely stream download to target_path.part then validate with ffprobe before renaming.
        Removes .part if download or validation fails.
        """
        import subprocess

        part_path = target_path.with_suffix(target_path.suffix + ".part")
        if part_path.exists():
            try:
                part_path.unlink()
            except Exception:
                pass

        downloaded_ok = False
        err_detail = None

        # Determine appropriate referer
        referer = "https://www.douyin.com/" if ("douyin" in media_url or "zjcdn" in media_url) else "https://snaptiktok.to/"

        # 1. Try with curl.exe
        try:
            cmd = [
                "curl.exe", "--http1.1", "-sL", "--connect-timeout", "10", "--max-time", str(int(timeout)),
                "-H", f"User-Agent: {self.DEFAULT_HEADERS['User-Agent']}",
                "-H", f"Referer: {referer}",
                media_url,
                "-o", str(part_path)
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
            if proc.returncode == 0 and part_path.exists() and part_path.stat().st_size > 0:
                downloaded_ok = True
            elif proc.returncode != 0:
                err_detail = f"curl exited with code {proc.returncode}"
        except Exception as ce:
            err_detail = str(ce)

        # 2. Fallback to httpx stream
        if not downloaded_ok:
            try:
                cdn_headers = {
                    "User-Agent": self.DEFAULT_HEADERS["User-Agent"],
                    "Referer": referer
                }
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    with client.stream("GET", media_url, headers=cdn_headers) as resp:
                        if resp.status_code == 200:
                            ctype = resp.headers.get("content-type", "").lower()
                            if ctype and "text/html" in ctype:
                                return False, "Phản hồi không phải video (nhận được text/html)"
                            with open(part_path, "wb") as f:
                                for chunk in resp.iter_bytes(chunk_size=65536):
                                    f.write(chunk)
                            if part_path.exists() and part_path.stat().st_size > 0:
                                downloaded_ok = True
                        else:
                            err_detail = f"Máy chủ media trả về HTTP {resp.status_code}"
            except Exception as he:
                err_detail = str(he)

        if not downloaded_ok or not part_path.exists() or part_path.stat().st_size == 0:
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            return False, f"Lỗi tải file media: {err_detail or 'file rỗng'}"

        # 3. Validate with ffprobe: duration > 0, video stream with width > 0 and height > 0
        is_valid, meta, probe_err = validate_downloaded_media(part_path)
        if not is_valid:
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            return False, f"Xác minh video ffprobe thất bại: {probe_err}"

        # 4. Rename .part to final target
        try:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)
            return True, None
        except Exception as e:
            return False, f"Không thể lưu file video: {e}"



class DownloaderService:
    def __init__(self):
        self.original_dir = ORIGINAL_DIR
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.snaptiktok = SnapTikTokDownloader()

    def download_video(self, url: str, target_path: str) -> Dict[str, Any]:
        """
        Attempt automatic download via SnapTikTok provider with safe streaming and ffprobe verification.
        If resolution or download fails, returns status MANUAL_REQUIRED.
        """
        orig_url = normalize_douyin_url(url)
        modal_id = extract_douyin_video_id(orig_url)
        canonical_url = get_canonical_douyin_url(orig_url)

        logger.info(
            f"DownloaderService: Provider=SnapTikTok, original_url={orig_url}, "
            f"canonical_url={canonical_url}, modal_id={modal_id}"
        )

        target_p = Path(target_path)
        target_p.parent.mkdir(parents=True, exist_ok=True)

        # 1. Resolve direct download link
        res_ok, media_url, err_msg = self.snaptiktok.resolve_video_url(canonical_url)
        if not res_ok or not media_url:
            logger.warning(f"SnapTikTok resolver failed for {canonical_url}: {err_msg}")
            return {
                "success": False,
                "status": "MANUAL_REQUIRED",
                "provider": "SnapTikTok",
                "error": err_msg,
                "message": f"Không thể tự động tải qua SnapTikTok ({err_msg}). Vui lòng sử dụng tính năng Đính Kèm Video Cục Bộ (Add Local Video)."
            }

        # 2. Download media stream safely
        dl_ok, dl_err = self.snaptiktok.download_file(media_url, target_p)
        if not dl_ok:
            logger.warning(f"SnapTikTok media download failed for {canonical_url}: {dl_err}")
            return {
                "success": False,
                "status": "MANUAL_REQUIRED",
                "provider": "SnapTikTok",
                "error": dl_err,
                "message": f"Tải tệp video thất bại ({dl_err}). Vui lòng sử dụng tính năng Đính Kèm Video Cục Bộ (Add Local Video)."
            }

        # 3. Retrieve probed metadata
        is_valid, meta, _ = validate_downloaded_media(target_p)
        file_size = meta.get("file_size", target_p.stat().st_size)
        duration = meta.get("duration", 0)
        res_str = f"{meta.get('width', 0)}x{meta.get('height', 0)}"

        logger.info(
            f"DownloaderService: Download success via SnapTikTok -> file={target_p.name}, "
            f"size={file_size}, duration={duration}s, res={res_str}"
        )

        return {
            "success": True,
            "status": "DOWNLOADED",
            "provider": "SnapTikTok",
            "target_path": str(target_p),
            "file_size": file_size,
            "duration": duration,
            "resolution": res_str,
            "video_codec": meta.get("video_codec", ""),
            "audio_codec": meta.get("audio_codec", ""),
            "modal_id": modal_id,
            "canonical_url": canonical_url,
            "message": f"Đã tự động tải video thành công ({file_size} bytes, {duration}s)"
        }

    def download_and_attach(
        self,
        db: Session,
        url: str = "",
        product_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download video via SnapTikTok and attach to a sequential Video record in SQLite DB.
        Prevents duplicates by checking original URL.
        """
        video = None
        orig_url = normalize_douyin_url(url)

        if video_id:
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if not video:
                return {"success": False, "error": f"Không tìm thấy video với mã {video_id}"}
            if not orig_url and video.douyin_url:
                orig_url = video.douyin_url

        if not orig_url:
            return {"success": False, "error": "Douyin URL không được để trống."}

        # If video_id not specified, check duplicate by URL
        if not video:
            existing = db.query(Video).filter(Video.douyin_url == orig_url).first()
            if existing:
                target_filename = f"{existing.video_id}_original.mp4"
                target_path = self.original_dir / target_filename
                if existing.downloaded and target_path.exists() and target_path.stat().st_size > 0:
                    return {
                        "success": True,
                        "video_id": existing.video_id,
                        "local_file": existing.local_file,
                        "status": existing.status,
                        "message": f"Video đã tồn tại và đã được tải về ({existing.video_id})."
                    }
                video = existing
            else:
                new_vid_id = get_next_video_id(db)
                video = Video(
                    video_id=new_vid_id,
                    product_id=product_id if product_id else None,
                    douyin_url=orig_url,
                    downloaded=False,
                    approved=False,
                    used=False,
                    status="FOUND"
                )
                db.add(video)
                db.flush()

        target_filename = f"{video.video_id}_original.mp4"
        target_path = self.original_dir / target_filename

        dl_res = self.download_video(orig_url, str(target_path))
        if not dl_res.get("success"):
            db.commit()
        # Calculate media hash and enforce media-level duplicate detection
        m_hash = calculate_file_sha256(target_path)
        if m_hash:
            dup_media = db.query(Video).filter(Video.media_hash == m_hash, Video.video_id != video.video_id).first()
            if dup_media:
                if target_path.exists():
                    target_path.unlink()
                logger.warning(f"Duplicate media hash {m_hash} matches existing video {dup_media.video_id}. Download rejected.")
                return {
                    "success": False,
                    "status": "DUPLICATE_MEDIA_HASH",
                    "media_hash": m_hash,
                    "duplicate_video_id": dup_media.video_id,
                    "error": f"Media content hash already exists in system for {dup_media.video_id} (duplicate video content)."
                }
            video.media_hash = m_hash

        rel_path = f"original/{target_filename}"
        video.local_file = rel_path
        video.downloaded = True
        video.status = "DOWNLOADED"
        db.commit()
        db.refresh(video)

        dl_res["video_id"] = video.video_id
        dl_res["local_file"] = rel_path
        dl_res["media_hash"] = m_hash
        dl_res["message"] = f"Đã lưu video gốc cho {video.video_id} thành công!"
        return dl_res

    def attach_local_video(
        self,
        db: Session,
        video_id: str,
        source_path_or_bytes: Union[str, Path, bytes],
        original_filename: str = ""
    ) -> Dict[str, Any]:
        """
        Validate and attach a legitimately obtained local video to the Video record.
        Saves as: downloads/original/{video_id}_original.mp4
        """
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video với mã {video_id}"}

        target_filename = f"{video_id}_original.mp4"
        target_path = self.original_dir / target_filename

        # If bytes were provided
        if isinstance(source_path_or_bytes, bytes):
            if len(source_path_or_bytes) == 0:
                return {"success": False, "error": "File video có kích thước 0 byte. File không hợp lệ."}
            ext = Path(original_filename).suffix.lower() if original_filename else ".mp4"
            if ext and ext not in SUPPORTED_EXTENSIONS:
                return {"success": False, "error": f"Định dạng {ext} không được hỗ trợ. Vui lòng dùng: {', '.join(SUPPORTED_EXTENSIONS)}"}
            with open(target_path, "wb") as f:
                f.write(source_path_or_bytes)
        else:
            src = Path(source_path_or_bytes)
            if not src.exists():
                return {"success": False, "error": f"File nguồn không tồn tại: {src}"}
            if src.stat().st_size == 0:
                return {"success": False, "error": "File video rỗng (0 byte). Không hợp lệ."}
            ext = src.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return {"success": False, "error": f"Định dạng {ext} không được hỗ trợ. Vui lòng dùng: {', '.join(SUPPORTED_EXTENSIONS)}"}
            # Copy file to downloads/original/Vxxxx_original.mp4
            shutil.copy2(src, target_path)

        # Calculate media hash and enforce media-level duplicate detection
        m_hash = calculate_file_sha256(target_path)
        if m_hash:
            dup_media = db.query(Video).filter(Video.media_hash == m_hash, Video.video_id != video.video_id).first()
            if dup_media:
                if target_path.exists():
                    target_path.unlink()
                return {
                    "success": False,
                    "status": "DUPLICATE_MEDIA_HASH",
                    "media_hash": m_hash,
                    "duplicate_video_id": dup_media.video_id,
                    "error": f"Media content hash matches existing video {dup_media.video_id}."
                }
            video.media_hash = m_hash

        # Update Video record in DB
        rel_path = f"original/{target_filename}"
        video.local_file = rel_path
        video.downloaded = True
        video.status = "DOWNLOADED"
        db.commit()
        db.refresh(video)

        logger.info(f"Local video attached to {video_id}: {target_path} (Status: DOWNLOADED, SHA256: {m_hash[:12]}...)")
        return {
            "success": True,
            "video_id": video_id,
            "local_file": rel_path,
            "media_hash": m_hash,
            "status": "DOWNLOADED",
            "message": f"Đã lưu video gốc cho {video_id} thành công!"
        }

    def download_batch(
        self,
        db: Session,
        items: List[Dict[str, Any]],
        timeout_per_video: float = 60.0
    ) -> Dict[str, Any]:
        """
        Execute batch download for multiple video source candidates.
        Enforces:
        - Validates each media item with ffprobe.
        - Calculates and stores SHA-256 media hash.
        - Rejects duplicate media hashes across the batch and existing database.
        - Never silently marks incomplete batch as success.
        """
        results: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []
        seen_batch_hashes = set()

        logger.info(f"Starting batch download for {len(items)} items...")

        for it in items:
            prod_id = it.get("product_id")
            source = it.get("source_candidate") or {}
            source_url = getattr(source, "canonical_url", None) or source.get("canonical_url", "")
            download_url = getattr(source, "download_url", None) or source.get("download_url", "") or source_url
            canonical_id = getattr(source, "canonical_source_id", None) or source.get("canonical_source_id", "")
            provider_id = getattr(source, "provider", None) or source.get("provider", "douyin")

            # Create video record
            vid_id = get_next_video_id(db)
            video = Video(
                video_id=vid_id,
                product_id=prod_id,
                douyin_url=source_url or download_url or f"https://source/{vid_id}",
                provider=provider_id,
                canonical_source_id=canonical_id,
                downloaded=False,
                approved=True,
                used=False,
                status="DOWNLOADING"
            )
            db.add(video)
            db.flush()

            target_filename = f"{vid_id}_original.mp4"
            target_path = self.original_dir / target_filename

            # If download_url points to a local file
            if Path(download_url).is_file():
                attach_res = self.attach_local_video(db, vid_id, Path(download_url))
                if attach_res.get("success"):
                    m_hash = attach_res.get("media_hash")
                    if m_hash in seen_batch_hashes:
                        # Reject duplicate in same batch
                        if target_path.exists():
                            target_path.unlink()
                        video.status = "DUPLICATE_MEDIA_HASH"
                        db.commit()
                        failed_items.append({
                            "product_id": prod_id,
                            "video_id": vid_id,
                            "status": "DUPLICATE_MEDIA_HASH",
                            "error": f"Duplicate media hash {m_hash} within current batch."
                        })
                        continue
                    seen_batch_hashes.add(m_hash)
                    results.append(attach_res)
                else:
                    failed_items.append({
                        "product_id": prod_id,
                        "video_id": vid_id,
                        "status": "DOWNLOAD_FAILED",
                        "error": attach_res.get("error")
                    })
                continue

            # Remote URL download via streaming
            dl_res = self.snaptiktok.download_file(download_url, target_path, timeout=timeout_per_video)
            if not dl_res[0]:
                video.status = "DOWNLOAD_FAILED"
                db.commit()
                failed_items.append({
                    "product_id": prod_id,
                    "video_id": vid_id,
                    "status": "DOWNLOAD_FAILED",
                    "error": dl_res[1]
                })
                continue

            # Validate ffprobe
            is_valid, meta, probe_err = validate_downloaded_media(target_path)
            if not is_valid:
                if target_path.exists():
                    target_path.unlink()
                video.status = "INVALID_MEDIA"
                db.commit()
                failed_items.append({
                    "product_id": prod_id,
                    "video_id": vid_id,
                    "status": "INVALID_MEDIA",
                    "error": probe_err
                })
                continue

            # Calculate SHA-256
            m_hash = calculate_file_sha256(target_path)
            if m_hash in seen_batch_hashes:
                if target_path.exists():
                    target_path.unlink()
                video.status = "DUPLICATE_MEDIA_HASH"
                db.commit()
                failed_items.append({
                    "product_id": prod_id,
                    "video_id": vid_id,
                    "status": "DUPLICATE_MEDIA_HASH",
                    "error": f"Duplicate media hash {m_hash} within current batch."
                })
                continue

            # Check DB duplicate media hash
            dup_db = db.query(Video).filter(Video.media_hash == m_hash, Video.video_id != vid_id).first()
            if dup_db:
                if target_path.exists():
                    target_path.unlink()
                video.status = "DUPLICATE_MEDIA_HASH"
                db.commit()
                failed_items.append({
                    "product_id": prod_id,
                    "video_id": vid_id,
                    "status": "DUPLICATE_MEDIA_HASH",
                    "error": f"Duplicate media hash {m_hash} matches existing video {dup_db.video_id}."
                })
                continue

            seen_batch_hashes.add(m_hash)
            video.media_hash = m_hash
            video.local_file = f"original/{target_filename}"
            video.downloaded = True
            video.status = "DOWNLOADED"
            db.commit()

            results.append({
                "success": True,
                "product_id": prod_id,
                "video_id": vid_id,
                "local_file": video.local_file,
                "media_hash": m_hash,
                "duration": meta.get("duration")
            })

        all_success = (len(results) == len(items)) and not failed_items
        return {
            "success": all_success,
            "status": "DOWNLOADED" if all_success else "INCOMPLETE",
            "total_requested": len(items),
            "downloaded_count": len(results),
            "videos": results,
            "failed_items": failed_items
        }
