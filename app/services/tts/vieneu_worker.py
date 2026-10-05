import argparse
import json
import os
import sys
import wave
from pathlib import Path

# Ensure UTF-8 I/O on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


def measure_wav_duration(wav_path: Path) -> float:
    """Measure exact WAV duration using standard wave module."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 3)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="VieNeu-TTS Subprocess Worker")
    parser.add_argument("--text", type=str, default="", help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Path to write output WAV")
    parser.add_argument("--voice", type=str, default="Trúc Ly", help="Preset voice name")
    parser.add_argument("--mode", type=str, default="v3turbo", choices=["v3turbo", "v3nano"], help="Model mode")
    parser.add_argument("--json-stdin", action="store_true", help="Read payload from JSON stdin")

    args = parser.parse_args()

    text = args.text
    voice = args.voice
    output_path = args.output
    mode = args.mode

    if args.json_stdin:
        try:
            stdin_data = sys.stdin.read()
            if stdin_data.strip():
                payload = json.loads(stdin_data)
                text = payload.get("text", text)
                voice = payload.get("voice", voice)
                output_path = payload.get("output", output_path)
                mode = payload.get("mode", mode)
        except Exception as e:
            print(json.dumps({"success": False, "error": f"Failed to parse stdin JSON: {e}"}))
            sys.exit(1)

    if not text.strip():
        print(json.dumps({"success": False, "error": "No text provided for synthesis."}))
        sys.exit(1)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Suppress Hugging Face download warnings
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        from vieneu import Vieneu

        # Initialize TTS instance
        tts = Vieneu(mode=mode)

        # Infer speech
        # Voice matching: if user passed a standard voice id like vi-VN-Standard-A, map to a good default
        voice_map = {
            "vi-VN-Standard-A": "Trúc Ly",
            "vi-VN-Standard-B": "Minh Quân Pro",
            "vi-VN-Standard-C": "Thùy Dung",
            "vi-VN-Standard-D": "Thái Sơn"
        }
        resolved_voice = voice_map.get(voice, voice)

        audio = tts.infer(text=text, voice=resolved_voice)
        tts.save(audio, str(out_file))

        if not out_file.exists() or out_file.stat().st_size == 0:
            print(json.dumps({"success": False, "error": "Synthesis finished but output file is empty or missing."}))
            sys.exit(1)

        dur = measure_wav_duration(out_file)
        sample_rate = getattr(tts, "sample_rate", 48000)

        result = {
            "success": True,
            "output": str(out_file),
            "duration": dur,
            "sample_rate": sample_rate,
            "voice": resolved_voice,
            "file_size": out_file.stat().st_size
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        err_msg = str(e)
        print(json.dumps({"success": False, "error": err_msg}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
