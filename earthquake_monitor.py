"""
earthquake_monitor.py
P2P地震情報 API (https://api.p2pquake.net/v2/) を利用し、
旭川・江別エリアで震度5強以上を検知したら Supabase の ingest_state テーブルを更新する。
GitHub Actions から5分ごとに呼び出される想定。
"""

import os
import sys
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(override=True)

# P2PQuake JSON API v2 — 地震情報 (code=551) を最大10件取得
P2P_API_URL = "https://api.p2pquake.net/v2/history?codes=551&limit=10"

# 対象地域（pref または addr に含まれるキーワード）
TARGET_PREF = "北海道"
TARGET_AREAS = ["旭川", "江別"]

# scale値 → 震度文字列（P2PQuake の内部表現 × 10 が震度）
SCALE_TO_INTENSITY: dict[int, str] = {
    10: "1",
    20: "2",
    30: "3",
    40: "4",
    45: "5弱",
    50: "5強",
    55: "6弱",
    60: "6強",
    70: "7",
}

# 震度5強以上のscale閾値
INTENSITY_THRESHOLD_SCALE = 50

# 過去何分以内のイベントを対象にするか
CHECK_WINDOW_MINUTES = 30

# ingest_state テーブルの地震ステータス専用レコードID
EARTHQUAKE_SITE_ID = -1


def get_supabase():
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or ""
    )
    if not url or not key:
        print("[earthquake_monitor] SUPABASE_URL or key is missing", file=sys.stderr)
        sys.exit(1)
    return create_client(url, key)


def fetch_p2p_quakes() -> list:
    """P2PQuake APIから最新の地震リストを取得"""
    try:
        resp = requests.get(
            P2P_API_URL,
            timeout=15,
            headers={"User-Agent": "chatbot-emergency-monitor/1.0"},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[earthquake_monitor] P2PQuake API fetch failed: {e}", file=sys.stderr)
        return []


def get_local_max_scale(points: list) -> tuple[int, str]:
    """
    points から旭川・江別の観測点のみを対象に最大scale を返す。
    対象地域の観測点がない場合は -1 を返す（他地域ではトリガーしない）。
    戻り値: (max_scale, 検出アドレス)
    """
    target_max = -1
    target_addr = ""

    for p in points:
        if p.get("pref", "") != TARGET_PREF:
            continue
        addr = p.get("addr", "")
        if not any(area in addr for area in TARGET_AREAS):
            continue
        scale = p.get("scale", -1)
        if scale > target_max:
            target_max = scale
            target_addr = addr

    return target_max, target_addr


def find_active_emergency(events: list) -> dict | None:
    """
    旭川・江別エリアで震度5強以上かつ過去30分以内のイベントを探す。
    見つかった場合はイベント情報dictを返す。なければNone。
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=CHECK_WINDOW_MINUTES)
    jst = timezone(timedelta(hours=9))

    for event in events:
        if event.get("code") != 551:
            continue

        eq = event.get("earthquake", {})
        points = event.get("points", [])

        # P2PQuake の time は JST 文字列: "2026/06/05 05:54:00"
        time_str = eq.get("time", "")
        try:
            at = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S").replace(tzinfo=jst)
            at_utc = at.astimezone(timezone.utc)
        except Exception:
            continue

        if at_utc < cutoff:
            continue

        local_scale, local_addr = get_local_max_scale(points)

        if local_scale < INTENSITY_THRESHOLD_SCALE:
            continue

        intensity = SCALE_TO_INTENSITY.get(local_scale, f"不明({local_scale})")
        hypocenter = eq.get("hypocenter", {}).get("name", "不明")
        event_id = str(event.get("id", time_str))

        return {
            "event_id": event_id,
            "intensity": intensity,
            "area": local_addr or hypocenter,
            "detected_at": at.isoformat(),
        }

    return None


def update_ingest_state(supabase, is_active: bool, event_info: dict | None):
    """
    ingest_state テーブルの site_id=-1 を地震ステータスとして upsert。
    route.ts が読む列:
      status    : 'emergency' | 'idle'
      last_url  : 震度文字列 (例: "5強")
      last_error: 検出エリア名
    """
    data = {
        "site_id": EARTHQUAKE_SITE_ID,
        "status": "emergency" if is_active else "idle",
        "last_url": event_info["intensity"] if event_info else None,
        "last_error": event_info["area"] if event_info else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase.table("ingest_state").upsert(data, on_conflict="site_id").execute()

    label = "EMERGENCY" if is_active else "normal"
    print(
        f"[earthquake_monitor] {label} | "
        f"intensity={data.get('last_url')} | area={data.get('last_error')}"
    )


def main():
    print(f"[earthquake_monitor] start (P2PQuake API) | target={TARGET_AREAS}")

    supabase = get_supabase()
    events = fetch_p2p_quakes()

    if not events:
        print("[earthquake_monitor] no events fetched, skipping DB update")
        return

    event_info = find_active_emergency(events)
    is_active = event_info is not None

    update_ingest_state(supabase, is_active, event_info)

    if is_active:
        print(
            f"[earthquake_monitor] EMERGENCY: 震度{event_info['intensity']} "
            f"/ {event_info['area']} / {event_info['detected_at']}"
        )
    else:
        print("[earthquake_monitor] no emergency in the last 30 minutes")


if __name__ == "__main__":
    main()
