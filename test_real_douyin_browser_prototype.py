"""
Real Browser Test for Douyin Browser Prototype

Performs exactly ONE real browser test using the dedicated profile:
runtime/douyin_browser_profile

Invariants:
- If login or CAPTCHA appears, STOP immediately and report:
  USER ACTION REQUIRED — LOGIN TO DOUYIN
- Do not attempt automatic bypass or solving.
"""

import sys
import time
from pathlib import Path

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.douyin_browser_service import (
    douyin_browser_service,
    STATE_LOGIN_REQUIRED,
    STATE_VERIFICATION_REQUIRED,
    STATE_CONNECTED,
    STATE_BROWSER_OPEN,
    STATE_NOT_CONNECTED
)

def run_real_browser_test():
    print("=" * 70)
    print("DOUYIN REAL BROWSER PROTOTYPE TEST")
    print("Dedicated Profile Path:", douyin_browser_service.profile_dir.resolve())
    print("=" * 70)

    # 1. Open dedicated browser
    print("\n[Step 1] Opening dedicated browser instance...")
    open_res = douyin_browser_service.open_browser()
    print("Open Result:", open_res.get("message"))
    print("Initial State:", open_res.get("state"))

    if not open_res.get("success") and open_res.get("state") != STATE_CONNECTED:
        print("[FAIL] Failed to open browser:", open_res.get("error"))
        return

    # Wait for page readiness
    time.sleep(3.0)

    # 2. Check initial browser state
    print("\n[Step 2] Checking truthful browser state...")
    status_res = douyin_browser_service.get_status()
    print(f"Current State: {status_res['state']}")
    print(f"Active URL   : {status_res.get('url')}")
    print(f"Active Title : {status_res.get('title')}")
    print(f"Message      : {status_res.get('message')}")

    # 3. Perform test search on rendered Douyin web UI
    keyword = "宿舍迷你电饭煲"
    print(f"\n[Step 3] Executing sequential search for keyword: '{keyword}'...")
    search_res = douyin_browser_service.search(keyword, limit=5)

    print("\n[Search Execution Summary]")
    print(f"Success       : {search_res.get('success')}")
    print(f"Reported State: {search_res.get('state')}")

    if search_res.get("state") == STATE_LOGIN_REQUIRED:
        print("\n" + "!" * 70)
        print("USER ACTION REQUIRED — LOGIN TO DOUYIN")
        print("Douyin requested account login. Automatic progress halted safely.")
        print("!" * 70)
        return

    if search_res.get("state") == STATE_VERIFICATION_REQUIRED:
        print("\n" + "!" * 70)
        print("USER ACTION REQUIRED — LOGIN TO DOUYIN")
        print("Douyin verification / CAPTCHA intermediate page detected.")
        print("Automatic progress halted safely.")
        print("!" * 70)
        return

    if search_res.get("success"):
        videos = search_res.get("videos", [])
        print(f"[PASS] Successfully extracted {len(videos)} videos:")
        for v in videos:
            print(f"  - [{v['video_id']}] {v['title'][:40]} -> {v['canonical_url']}")
    else:
        print(f"[HALTED] Search ended with state {search_res.get('state')}: {search_res.get('error') or search_res.get('message')}")

if __name__ == "__main__":
    run_real_browser_test()
