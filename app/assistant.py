"""
Analytics assistant — answers natural-language questions using live store data.
No external LLM required; uses metrics, funnel, heatmap, anomalies, and breakdown.
"""

from __future__ import annotations

import re
from typing import Any


def _fmt_inr(n: float) -> str:
    return f"₹{int(round(n)):,}"


def _zone_label(name: str) -> str:
    return name.replace("_", " ").title()


def answer_question(
    question: str,
    *,
    metrics: dict[str, Any],
    funnel: dict[str, Any],
    heatmap: dict[str, Any],
    anomalies: list[dict[str, Any]],
    breakdown: dict[str, Any] | None = None,
    store_id: str = "ST1008",
    date: str = "",
) -> dict[str, Any]:
    q = (question or "").strip().lower()
    if not q:
        return {
            "answer": "Ask me about conversion, revenue, queue length, zones, alerts, funnel drop-off, or door traffic.",
            "suggestions": _default_suggestions(),
        }

    stages = funnel.get("stages") or {}
    zones = heatmap.get("zones") or {}
    avs = heatmap.get("attention_vs_sales") or {}
    headline = (breakdown or {}).get("headline") or {}

    conv_pct = (metrics.get("conversion_rate") or 0) * 100
    visitors = metrics.get("unique_visitors", 0)
    buyers = metrics.get("buyers", 0)
    revenue = metrics.get("revenue_inr", 0)
    queue = metrics.get("current_queue_depth", 0)
    rpv = metrics.get("revenue_per_visitor_inr", 0)

    sources: list[str] = []

    # ── Conversion / buyers ───────────────────────────────────────────────
    if re.search(r"\b(conversion|convert|buyer|purchase rate)\b", q):
        sources.append("metrics")
        entry = (stages.get("stage_entry") or {}).get("count", 0)
        purchase = (stages.get("stage_purchase") or {}).get("count", 0)
        return {
            "answer": (
                f"**Conversion** for {store_id} on {date}: **{conv_pct:.2f}%** "
                f"({buyers} buyers from {visitors} unique visitors). "
                f"Funnel shows {entry:,} entries → {purchase:,} purchases. "
                f"Revenue per visitor is {_fmt_inr(rpv)}."
            ),
            "sources": sources,
            "suggestions": ["Why is conversion low?", "Show funnel drop-off"],
        }

    # ── Revenue ─────────────────────────────────────────────────────────────
    if re.search(r"\b(revenue|sales|money|inr|rupee|basket|aov)\b", q):
        sources.append("metrics")
        avg_basket = revenue / buyers if buyers else 0
        return {
            "answer": (
                f"**Revenue today**: {_fmt_inr(revenue)} from {buyers} buyers "
                f"(avg basket ~{_fmt_inr(avg_basket)}). "
                f"Revenue per visitor: {_fmt_inr(rpv)} across {visitors} visitors."
            ),
            "sources": sources,
            "suggestions": ["What is conversion?", "Which zone earns most?"],
        }

    # ── Queue / billing ─────────────────────────────────────────────────────
    if re.search(r"\b(queue|billing|checkout|counter|wait)\b", q):
        sources.extend(["metrics", "anomalies"])
        queue_alerts = [a for a in anomalies if "QUEUE" in (a.get("type") or "").upper()]
        status = (
            f"**Critical**: queue depth is **{queue}** — consider opening a second counter."
            if queue >= 6
            else f"**Queue depth is {queue}** — within normal range."
        )
        if queue_alerts:
            status += f" Active alert: {queue_alerts[0].get('suggested_action', '')}"
        return {
            "answer": status,
            "sources": sources,
            "suggestions": ["Any alerts right now?", "How many visitors?"],
        }

    # ── Visitors / footfall ─────────────────────────────────────────────────
    if re.search(r"\b(visitor|footfall|traffic|people|headcount|crowd)\b", q):
        sources.extend(["metrics", "breakdown"])
        in_store = headline.get("estimated_in_store_now")
        door_in = headline.get("door_entries_cam03")
        door_out = headline.get("door_exits_cam03")
        extra = ""
        if door_in is not None:
            extra = (
                f" CAM_03 door: **{door_in}** entries, **{door_out}** exits; "
                f"estimated **{in_store}** people in store now."
            )
        return {
            "answer": (
                f"**{visitors}** unique visitors (staff excluded, session-deduped).{extra} "
                f"Raw track IDs: {headline.get('unique_visitors_raw_tracks', '—')} "
                f"({headline.get('unique_visitors_sessions', '—')} customer sessions)."
            ),
            "sources": sources,
            "suggestions": ["What is conversion?", "Door entry count?"],
        }

    # ── Zones / dwell / heatmap ─────────────────────────────────────────────
    if re.search(r"\b(zone|dwell|heatmap|placement|brand|shelf|dead)\b", q):
        sources.append("heatmap")
        if not zones:
            return {"answer": "No zone heatmap data for this date.", "sources": sources}

        ranked = sorted(
            zones.items(),
            key=lambda x: x[1].get("normalised_score", 0),
            reverse=True,
        )
        top = ranked[0]
        gaps = []
        for name, z in ranked:
            ds = (avs.get(name) or {}).get("dwell_share", 0)
            rs = (avs.get(name) or {}).get("sales_share", 0)
            if ds - rs > 0.1:
                gaps.append(_zone_label(name))

        top_name = _zone_label(top[0])
        top_score = top[1].get("normalised_score", 0)
        gap_text = (
            f" Zones to review (high dwell, lower sales): **{', '.join(gaps[:4])}**."
            if gaps
            else " Dwell vs revenue looks balanced across zones."
        )
        return {
            "answer": (
                f"Highest traffic index: **{top_name}** (score {top_score:.2f}). "
                f"{len(zones)} zones tracked.{gap_text}"
            ),
            "sources": sources,
            "suggestions": ["Which zone underperforms?", "Show all alerts"],
        }

    # ── Funnel ──────────────────────────────────────────────────────────────
    if re.search(r"\b(funnel|drop|journey|stage|entry|billing stage)\b", q):
        sources.append("funnel")
        labels = {
            "stage_entry": "Entry",
            "stage_zone_visit": "Zone visit",
            "stage_billing": "Billing",
            "stage_purchase": "Purchase",
        }
        lines = []
        prev = None
        for key in labels:
            s = stages.get(key) or {}
            cnt = s.get("count", 0)
            drop = s.get("drop_off_from_previous_pct", 0)
            line = f"**{labels[key]}**: {cnt:,}"
            if prev is not None and drop:
                line += f" (−{drop}% from previous)"
            lines.append(line)
            prev = cnt
        return {
            "answer": "Customer journey funnel:\n" + "\n".join(f"• {ln}" for ln in lines),
            "sources": sources,
            "suggestions": ["What is conversion?", "Revenue today?"],
        }

    # ── Alerts / anomalies ──────────────────────────────────────────────────
    if re.search(r"\b(alert|anomal|issue|problem|warn|critical)\b", q):
        sources.append("anomalies")
        if not anomalies:
            return {
                "answer": "✓ No anomalies flagged for this store date. Operations look normal.",
                "sources": sources,
                "suggestions": ["Queue status?", "Zone performance"],
            }
        lines = []
        for a in anomalies[:5]:
            sev = a.get("severity", "INFO")
            typ = (a.get("type") or "").replace("_", " ")
            act = a.get("suggested_action", "")
            lines.append(f"**[{sev}]** {typ}: {act}")
        more = f"\n(+{len(anomalies) - 5} more)" if len(anomalies) > 5 else ""
        return {
            "answer": f"**{len(anomalies)} alert(s)**:\n" + "\n".join(f"• {ln}" for ln in lines) + more,
            "sources": sources,
            "suggestions": ["How to fix queue?", "Conversion rate?"],
        }

    # ── Cameras / events / breakdown ────────────────────────────────────────
    if re.search(r"\b(camera|cam_|event|track|pipeline|confidence)\b", q):
        sources.append("breakdown")
        if not breakdown:
            return {"answer": "Breakdown data unavailable.", "sources": sources}
        by_cam = breakdown.get("by_camera") or {}
        by_ev = breakdown.get("by_event_type") or {}
        cam_lines = [
            f"**{c}**: {v.get('unique_customer_tracks', 0)} customer tracks, {v.get('events_total', 0)} events"
            for c, v in sorted(by_cam.items())
        ]
        top_ev = sorted(by_ev.items(), key=lambda x: -x[1])[:3]
        ev_line = ", ".join(f"{k} ({v})" for k, v in top_ev)
        return {
            "answer": (
                f"**{headline.get('events_total', 0)}** total events.\n"
                f"Cameras:\n" + "\n".join(f"• {ln}" for ln in cam_lines) +
                f"\nTop event types: {ev_line}."
            ),
            "sources": sources,
            "suggestions": ["Visitors today?", "Any alerts?"],
        }

    # ── Door / entry ────────────────────────────────────────────────────────
    if re.search(r"\b(door|entry|exit|in store)\b", q):
        sources.append("breakdown")
        return {
            "answer": (
                f"Door line (CAM_03): **{headline.get('door_entries_cam03', 0)}** entries, "
                f"**{headline.get('door_exits_cam03', 0)}** exits → "
                f"**{headline.get('estimated_in_store_now', 0)}** estimated in store."
            ),
            "sources": sources,
            "suggestions": ["How many visitors?", "Queue length?"],
        }

    # ── Help / summary ──────────────────────────────────────────────────────
    if re.search(r"\b(help|what can|how|summary|overview|status)\b", q):
        return {"answer": _executive_summary(metrics, funnel, anomalies, store_id, date), "sources": ["metrics", "funnel", "anomalies"], "suggestions": _default_suggestions()}

    # Default: executive summary
    return {
        "answer": _executive_summary(metrics, funnel, anomalies, store_id, date),
        "sources": ["metrics", "funnel", "anomalies"],
        "suggestions": _default_suggestions(),
    }


def _executive_summary(metrics, funnel, anomalies, store_id, date) -> str:
    conv_pct = (metrics.get("conversion_rate") or 0) * 100
    visitors = metrics.get("unique_visitors", 0)
    buyers = metrics.get("buyers", 0)
    revenue = metrics.get("revenue_inr", 0)
    queue = metrics.get("current_queue_depth", 0)
    alert_n = len(anomalies)
    purchase = (funnel.get("stages") or {}).get("stage_purchase", {}).get("count", 0)
    return (
        f"**{store_id} · {date}** — {visitors} visitors, {buyers} buyers, "
        f"**{conv_pct:.2f}%** conversion, {_fmt_inr(revenue)} revenue, "
        f"queue **{queue}**, {purchase:,} funnel purchases, **{alert_n}** active alert(s)."
    )


def _default_suggestions() -> list[str]:
    return [
        "What is today's conversion?",
        "How is the billing queue?",
        "Which zones need review?",
        "Summarize alerts",
        "Explain the customer funnel",
    ]
