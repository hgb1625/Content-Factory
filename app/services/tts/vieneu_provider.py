import json
import logging
import os
import math
import struct
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.services.tts.base import BaseTTSProvider

logger = logging.getLogger("app.services.tts.vieneu")


def create_pcm_wav_file(output_path: Path, duration_seconds: float = 2.0, sample_rate: int = 24000) -> bool:
    """Generate a standard valid PCM WAV audio file (pure standard Python library)."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int(sample_rate * duration_seconds)
        # Generate a gentle pleasant tone
        frequency = 440.0  # A4
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            raw_data = bytearray()
            for i in range(num_samples):
                # Apply envelope fade out
                envelope = max(0.0, 1.0 - (i / num_samples))
                val = int(math.sin(2.0 * math.pi * frequency * (i / sample_rate)) * 16000 * envelope)
                raw_data.extend(struct.pack('<h', val))
            wav_file.writeframes(raw_data)
        return True
    except Exception as e:
        logger.error(f"Failed to create WAV file: {e}")
        return False


class VieNeuProvider(BaseTTSProvider):
    def __init__(self):
        self.engine_name = "VieNeu-TTS"
        self._find_python_env()

    def _find_python_env(self):
        """Locate isolated Python environment with VieNeu installed."""
        try:
            from app.database import BASE_DIR
        except ImportError:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        custom_env = os.getenv("VIENEU_PYTHON_PATH") or os.getenv("VIENEU_PYTHON")
        candidate_paths = []
        if custom_env:
            candidate_paths.append(Path(custom_env))

        candidate_paths.extend([
            BASE_DIR / ".venv_vieneu_new" / "Scripts" / "python.exe",
            BASE_DIR / ".venv_vieneu_new" / "bin" / "python",
            BASE_DIR / ".venv_vieneu" / "Scripts" / "python.exe",
            BASE_DIR / ".venv_vieneu" / "bin" / "python",
        ])
        self._python_exe = None
        for p in candidate_paths:
            if p.exists():
                self._python_exe = str(p.resolve())
                break

        self.worker_script = str((BASE_DIR / "app" / "services" / "tts" / "vieneu_worker.py").resolve())
        self._installed = self._python_exe is not None and Path(self.worker_script).exists()

    def is_available(self) -> bool:
        self._find_python_env()
        return self._installed

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Dynamically detect and return verified voice presets from installed VieNeu package."""
        self._find_python_env()
        candidate_json_paths = []
        if self._python_exe:
            py_path = Path(self._python_exe)
            candidate_json_paths.extend([
                py_path.parent.parent / "Lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json",
                py_path.parent.parent / "lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json",
            ])

        try:
            from app.database import BASE_DIR
            candidate_json_paths.extend([
                BASE_DIR / ".venv_vieneu_new" / "Lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json",
                BASE_DIR / ".venv_vieneu" / "Lib" / "site-packages" / "vieneu" / "assets" / "voices_v3_turbo.json",
            ])
        except Exception:
            pass

        for p in candidate_json_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    presets = data.get("presets", {})
                    if presets:
                        voices = []
                        # Ensure Trúc Ly is first if present
                        ordered_keys = sorted(presets.keys(), key=lambda k: (0 if k == "Trúc Ly" else (1 if k == "Minh Quân Pro" else 2), k))
                        for k in ordered_keys:
                            meta = presets[k]
                            desc = meta.get("description", "")
                            label = f"{k} — {desc}" if desc else k
                            voices.append({"id": k, "name": label})
                        return voices
                except Exception as e:
                    logger.warning(f"Failed to parse voices_v3_turbo.json at {p}: {e}")

        # Fallback list of real VieNeu voices
        return [
            {"id": "Trúc Ly", "name": "Trúc Ly — Nữ · Bắc · Phong cách tự nhiên"},
            {"id": "Minh Quân Pro", "name": "Minh Quân Pro — Nam · Bắc · Phong cách tự nhiên"},
            {"id": "Thùy Dung", "name": "Thùy Dung — Nữ · Nam · Phong cách tin tức"},
            {"id": "Mai Anh", "name": "Mai Anh — Nữ · Bắc · Phong cách tin tức"},
            {"id": "Anh Khôi", "name": "Anh Khôi — Nam · Bắc · Phong cách kể chuyện"},
            {"id": "Quang Sơn", "name": "Quang Sơn — Nam · Trung · Phong cách tự nhiên"},
            {"id": "Ngọc Huyền", "name": "Ngọc Huyền — Nữ · Bắc · Giọng đọc tự nhiên"},
            {"id": "Mỹ Duyên", "name": "Mỹ Duyên — Nữ · Nam · Phong cách đọc truyện"},
            {"id": "Thái Sơn", "name": "Thái Sơn — Nam · Nam · Phong cách kể chuyện"},
            {"id": "Đoan Trang", "name": "Đoan Trang — Nữ · Bắc · Phong cách tự nhiên"},
        ]

    def synthesize(self, text: str, output_path: str, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize text to speech using real VieNeu-TTS via isolated worker subprocess.
        Returns VOICE_READY on success, or BLOCKED_EXTERNAL / VOICE_ERROR on failure.
        NEVER falls back to fake test audio in production.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        self._find_python_env()

        if not self._installed or not self._python_exe:
            logger.warning("VieNeu-TTS environment (.venv_vieneu) not found. Marked BLOCKED_EXTERNAL.")
            return {
                "success": False,
                "status": "BLOCKED_EXTERNAL",
                "error": "Môi trường VieNeu-TTS chưa được cấu hình tại .venv_vieneu. Trạng thái: BLOCKED_EXTERNAL.",
                "output_path": str(out_file)
            }

        target_voice = voice_name or "Trúc Ly"
        cmd = [
            self._python_exe,
            self.worker_script,
            "--text", text,
            "--output", str(out_file),
            "--voice", target_voice
        ]

        logger.info(f"Synthesizing with real VieNeu-TTS ({target_voice}): {text[:50]}...")

        try:
            import subprocess
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120.0
            )

            if proc.returncode != 0:
                err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Worker exited with code {proc.returncode}"
                try:
                    parsed_err = json.loads(proc.stdout)
                    if "error" in parsed_err:
                        err_msg = parsed_err["error"]
                except Exception:
                    pass

                logger.error(f"VieNeu worker error: {err_msg}")
                return {
                    "success": False,
                    "status": "VOICE_ERROR",
                    "error": f"Lỗi tạo giọng nói VieNeu: {err_msg}",
                    "output_path": str(out_file)
                }

            # Parse JSON output from worker
            output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if not output_lines:
                return {
                    "success": False,
                    "status": "VOICE_ERROR",
                    "error": "VieNeu worker không trả về dữ liệu kết quả.",
                    "output_path": str(out_file)
                }

            last_line = output_lines[-1]
            try:
                res = json.loads(last_line)
            except Exception as e:
                return {
                    "success": False,
                    "status": "VOICE_ERROR",
                    "error": f"Lỗi phân tích kết quả worker: {e}",
                    "output_path": str(out_file)
                }

            if not res.get("success"):
                return {
                    "success": False,
                    "status": "VOICE_ERROR",
                    "error": res.get("error", "Lỗi tạo giọng nói VieNeu"),
                    "output_path": str(out_file)
                }

            return {
                "success": True,
                "status": "VOICE_READY",
                "output_path": str(out_file),
                "voice_name": res.get("voice", target_voice),
                "duration": res.get("duration", 0.0),
                "sample_rate": res.get("sample_rate", 48000),
                "file_size": res.get("file_size", 0)
            }

        except subprocess.TimeoutExpired:
            logger.error("VieNeu synthesis timed out after 120s.")
            return {
                "success": False,
                "status": "VOICE_ERROR",
                "error": "Quá trình tạo giọng nói VieNeu vượt quá thời gian chờ (120s).",
                "output_path": str(out_file)
            }
        except Exception as e:
            logger.error(f"VieNeu-TTS synthesis runtime exception: {e}")
            return {
                "success": False,
                "status": "VOICE_ERROR",
                "error": f"Lỗi tạo giọng nói: {str(e)}",
                "output_path": str(out_file)
            }

