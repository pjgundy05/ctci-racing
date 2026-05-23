import io
import re
from datetime import datetime, date

import pandas as pd
import pdfplumber
import streamlit as st

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="🏇 CTCI Horse Racing Model", page_icon="🏇", layout="wide")
st.title("🏇 CTCI Horse Racing Model")
st.caption("Brisnet PDF analyzer — ranked by configurable scoring formula")

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
DEFAULT_PP = 100
DEFAULT_SPEED = 0
DEFAULT_STYLE = "NA"

STYLE_MAP = {"E": 7, "EP": 5, "P": 3, "S": 1}
STYLE_LABELS = {
    "E": "Early", "EP": "Early-Presser",
    "P": "Presser", "S": "Closer", "NA": "Unknown",
}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def find_first(pattern, text, flags=re.IGNORECASE, group=1, default=None):
    m = re.search(pattern, text or "", flags)
    return m.group(group) if m else default


def normalize_prog(s: str) -> str:
    s = (s or "").strip().upper()
    m = re.match(r"^(\d+)([A-Z]?)$", s)
    if not m:
        return s
    return m.group(1) + m.group(2)


# ──────────────────────────────────────────────
# Field extractors
# ──────────────────────────────────────────────
def extract_prime_power(block: str) -> int:
    v = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1)
    if v is None:
        v = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1)
    result = safe_int(v, DEFAULT_PP)
    # Sanity check: PP is typically 40–200
    return result if 40 <= result <= 200 else DEFAULT_PP


def extract_running_style(first_line: str, block: str) -> str:
    for src in (first_line, block):
        sty = find_first(r"\((E\/P|E|P|S)\s*\d*\)", src, group=1)
        if sty:
            return "EP" if sty.upper() == "E/P" else sty.upper()
    return DEFAULT_STYLE


def extract_speed(block: str) -> int:
    for pat in (
        r"Best\s+Speed\s+at\s+Dist[:\s]+(\d+)",
        r"Best\s+(?:Turf|Dirt)\s+Speed[:\s]+(\d+)",
        r"Highest\s+last\s+race\s+speed\s+rating[:\s]+(\d+)",
        r"Speed\s+Rating[:\s]+(\d+)",
        r"Last\s+Speed[:\s]+(\d+)",
        r"\bSR[:\s]+(\d+)",
    ):
        v = find_first(pat, block, group=1)
        if v is not None:
            result = safe_int(v, DEFAULT_SPEED)
            if 40 <= result <= 150:
                return result
    return DEFAULT_SPEED


def extract_jockey(block: str) -> str:
    for pat in (
        r"Jockey[:\s]+([A-Z][A-Za-z\s\.\-\']+?)(?:\s{2,}|\n|$)",
        r"\bJkt[:\s]+([A-Z][A-Za-z\s\.\-\']+?)(?:\s{2,}|\n|$)",
        r"(?:^|\n)\s*J:\s*([A-Z][A-Za-z\s\.\-\']+?)(?:\s{2,}|\n|$)",
    ):
        v = find_first(pat, block, group=1)
        if v and len(v.strip()) > 2:
            return v.strip()
    return "—"


def extract_trainer(block: str) -> str:
    for pat in (
        r"Trainer[:\s]+([A-Z][A-Za-z\s\.\-\']+?)(?:\s{2,}|\n|$)",
        r"\bTr[:\s]+([A-Z][A-Za-z\s\.\-\']+?)(?:\s{2,}|\n|$)",
    ):
        v = find_first(pat, block, group=1)
        if v and len(v.strip()) > 2:
            return v.strip()
    return "—"


def extract_morning_line(suffix: str) -> str:
    """Parse ML odds from the trailing text stripped off the horse name line."""
    if not suffix:
        return "—"
    if re.search(r"(?i)\bevt\b|even", suffix):
        return "Evn"
    m = re.search(r"\b(\d{1,2}[-/]\d{1,2}|\d{1,3})\b", suffix)
    return m.group(1) if m else "—"


def extract_days_off(block: str) -> int | None:
    """Return days since last race, or None if not found."""
    today = date.today()
    date_fmts = [
        (r"\b(\d{2}/\d{2}/\d{2})\b", "%m/%d/%y"),
        (r"\b(\d{2}/\d{2}/\d{4})\b", "%m/%d/%Y"),
        (r"\b(\d{1,2}-[A-Za-z]{3}-\d{2,4})\b", "%d-%b-%y"),
        (r"\b(\d{1,2}-[A-Za-z]{3}-\d{4})\b", "%d-%b-%Y"),
        (r"\b([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})\b", "%b %d, %Y"),
    ]
    for pat, fmt in date_fmts:
        m = re.search(pat, block)
        if m:
            try:
                dt = datetime.strptime(m.group(1), fmt).date()
                days = (today - dt).days
                if 0 <= days <= 730:
                    return days
            except ValueError:
                continue
    return None


def extract_class_label(block: str) -> str:
    """Extract broad race class: Stakes, Allowance, Claiming, Maiden, etc."""
    for pat, label in [
        (r"\b(?:G\d|Grade\s*\d|Stk|Stakes)\b", "Stakes"),
        (r"\b(?:Alw|Allowance)\b", "Allowance"),
        (r"\b(?:OC|Optional\s*Clm)\b", "Opt Clm"),
        (r"\b(?:Clm|Claiming)\b", "Claiming"),
        (r"\b(?:Mdn|Maiden)\b", "Maiden"),
    ]:
        if find_first(pat, block, group=0):
            return label
    return "—"


# ──────────────────────────────────────────────
# Horse parsing (program-line segmentation)
# ──────────────────────────────────────────────
def extract_horses_from_text(text: str) -> list[dict]:
    if not text:
        return []

    lines = text.splitlines()
    start_idxs = [
        i for i, ln in enumerate(lines)
        if re.match(r"^\s*\d+[A-Z]?\s+[A-Za-z]", ln or "")
    ]
    if not start_idxs:
        return []

    blocks = []
    for k, si in enumerate(start_idxs):
        ei = start_idxs[k + 1] if k + 1 < len(start_idxs) else len(lines)
        block = "\n".join(lines[si:ei]).strip()
        if block:
            blocks.append(block)

    horses = []
    for block in blocks:
        first_line = next((ln for ln in block.splitlines() if ln.strip()), "")

        m_prog = re.match(r"^\s*(\d+[A-Z]?)\s+", first_line or "")
        prog_raw = m_prog.group(1).strip() if m_prog else ""
        prog = normalize_prog(prog_raw)

        # Extract ML odds suffix BEFORE stripping it from the name
        name = re.sub(r"^\s*\d+[A-Z]?\s+", "", first_line or "").strip()
        ml_match = re.search(r"(\s+\d+[/\-]\d+\s*[A-Za-z]*|\s+\d+\s*[A-Za-z]*)$", name)
        ml_suffix = ml_match.group(0) if ml_match else ""
        ml = extract_morning_line(ml_suffix)

        name = re.sub(r"\s+\d+/?\d+\s*[A-Za-z]*\s*$", "", name)
        name = re.sub(r"[,\s]+$", "", name)
        if not name:
            name = f"{prog}-Horse" if prog else "Horse"

        horses.append({
            "Prog": prog if prog else "",
            "Horse": name,
            "ML": ml,
            "Style": extract_running_style(first_line, block),
            "PrimePower": extract_prime_power(block),
            "Speed": extract_speed(block),
            "DaysOff": extract_days_off(block),
            "Class": extract_class_label(block),
            "Jockey": extract_jockey(block),
            "Trainer": extract_trainer(block),
            "Block": block,
            "BlockLen": len(block),
            "NameLen": len(name),
        })

    if not horses:
        return []

    # Deduplicate within exact program number — keep richest block
    by_prog: dict = {}
    for h in horses:
        key = h["Prog"] or f"__BLANK__{id(h)}"
        prev = by_prog.get(key)
        if prev is None or (h["BlockLen"], h["NameLen"]) > (prev["BlockLen"], prev["NameLen"]):
            by_prog[key] = h

    horses = list(by_prog.values())
    df = pd.DataFrame(horses)
    if df.empty:
        return []

    # Fill blank program numbers sequentially
    if df["Prog"].eq("").any():
        c = 1
        new_progs = []
        for v in df["Prog"].tolist():
            if v:
                new_progs.append(v)
            else:
                new_progs.append(str(c))
                c += 1
        df["Prog"] = new_progs

    df = df.drop_duplicates(subset=["Prog", "Horse"])
    df["Prog"] = df["Prog"].astype(str)
    return df.drop(columns=["BlockLen", "NameLen"], errors="ignore").to_dict("records")


# ──────────────────────────────────────────────
# Race splitting (header-only strategy)
# ──────────────────────────────────────────────
def split_pdf_into_races_header_only(full_text: str) -> list[tuple[str, str]]:
    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    post_time_re = re.compile(r"(?i)\bPost\s*Time\b")
    solo_number_re = re.compile(r"^\s*\d{1,2}\s*$")

    post_idxs = [i for i, ln in enumerate(lines) if post_time_re.search(ln or "")]

    if not post_idxs:
        classic = [
            i for i, ln in enumerate(lines)
            if re.match(r"(?i)^\s*Race\s+\d{1,2}(?:\s*[-–—].*)?$", ln or "")
        ]
        if not classic:
            return [("Race 1", text)]
        starts = sorted(classic)
    else:
        starts = []
        for pt in post_idxs:
            for j in range(max(0, pt - 12), pt + 1):
                if solo_number_re.match(lines[j] or ""):
                    starts.append(j)
                    break
        if not starts:
            return [("Race 1", text)]
        starts = sorted(set(starts))

    # Dedupe near-duplicates (< 3 lines apart)
    deduped: list[int] = []
    last = -999
    for s in starts:
        if s - last > 3:
            deduped.append(s)
            last = s

    races = []
    for k, start_idx in enumerate(deduped):
        end_idx = deduped[k + 1] if k + 1 < len(deduped) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        races.append((f"Race {k + 1}", chunk))
    return races


# ──────────────────────────────────────────────
# Pace scenario
# ──────────────────────────────────────────────
def pace_scenario(style_counts: dict) -> tuple[str, str, dict]:
    """Return (label, advice, style_bonus_map) based on field pace shape."""
    e = style_counts.get("E", 0)
    ep = style_counts.get("EP", 0)
    s = style_counts.get("S", 0)
    front = e + ep

    if front >= 3:
        label, advice = "Hot Pace", "Speed duel likely — closers/stalkers favored"
        adj = {"E": -0.6, "EP": -0.3, "P": 0.2, "S": 0.6, "NA": 0.0}
    elif front == 0:
        label, advice = "No Pace", "No speed — on-pace horses may wire"
        adj = {"E": 0.0, "EP": 0.2, "P": 0.4, "S": -0.2, "NA": 0.0}
    elif e == 1 and ep == 0:
        label, advice = "Lone Speed", "Single speed horse — may wire uncontested"
        adj = {"E": 0.9, "EP": 0.3, "P": 0.0, "S": -0.2, "NA": 0.0}
    elif s >= 3:
        label, advice = "Closer-Heavy", "Many closers — front-runners may benefit"
        adj = {"E": 0.3, "EP": 0.2, "P": 0.1, "S": -0.2, "NA": 0.0}
    else:
        label, advice = "Contested Pace", "Normal pace scenario"
        adj = {"E": 0.0, "EP": 0.0, "P": 0.0, "S": 0.0, "NA": 0.0}

    return label, advice, adj


# ──────────────────────────────────────────────
# Layoff factor
# ──────────────────────────────────────────────
def layoff_factor(days_off) -> float:
    if days_off is None:
        return 1.0
    if days_off <= 14:
        return 1.02
    if days_off <= 30:
        return 1.00
    if days_off <= 60:
        return 0.97
    if days_off <= 120:
        return 0.93
    return 0.88


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────
def analyze_single_race_text(text: str, weights: dict) -> tuple:
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)
    if df.empty:
        return df, {}

    df["Prog"] = df["Prog"].astype(str)
    mask = df["Prog"].eq("") | df["Prog"].isna()
    if mask.any():
        c = 1
        new_vals = []
        for v, m in zip(df["Prog"], mask):
            if m:
                new_vals.append(str(c)); c += 1
            else:
                new_vals.append(v)
        df["Prog"] = new_vals

    df = df.fillna({"Style": DEFAULT_STYLE, "PrimePower": DEFAULT_PP, "Speed": DEFAULT_SPEED})

    prime_w = float(weights.get("prime_power", 1.0))
    speed_w = float(weights.get("speed", 1.0))
    style_w = float(weights.get("style", 0.5))
    apply_layoff = bool(weights.get("layoff", True))
    apply_pace = bool(weights.get("pace_scenario", True))

    # Pace scenario adjustments
    style_counts = df["Style"].value_counts().to_dict()
    scenario_label, scenario_advice, pace_adj_map = pace_scenario(style_counts)

    df["StyleRating"] = df["Style"].map(STYLE_MAP).fillna(0)
    df["PaceAdj"] = df["Style"].map(pace_adj_map).fillna(0.0) if apply_pace else 0.0
    df["LayoffFactor"] = df["DaysOff"].apply(layoff_factor) if apply_layoff else 1.0

    # Rating: (PP + Speed + Style + PaceAdj scaled to PP range) × LayoffFactor
    base = (
        prime_w * df["PrimePower"].astype(float)
        + speed_w * df["Speed"].astype(float)
        + style_w * df["StyleRating"]
        + df["PaceAdj"] * prime_w * 5
    )
    df["Rating"] = (base * df["LayoffFactor"]).round(1)

    df = df.sort_values(["Rating", "PrimePower"], ascending=[False, False]).reset_index(drop=True)

    meta = {
        "scenario": scenario_label,
        "advice": scenario_advice,
        "style_counts": style_counts,
        "horse_count": len(df),
        "pp_defaults": int((df["PrimePower"] == DEFAULT_PP).sum()),
    }
    return df, meta


def analyze_pdf_all(file_bytes: bytes, weights: dict) -> list:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return []

    if not full_text.strip():
        st.error(
            "No text could be extracted. This PDF may be a scanned image — "
            "only text-based Brisnet PDFs are supported."
        )
        return []

    with st.expander("Diagnostics — header scan", expanded=False):
        st.write(f"Characters extracted: {len(full_text):,}")
        st.code("\n".join(full_text.splitlines()[:20]) or "(no text)", language="text")

    races = split_pdf_into_races_header_only(full_text)

    with st.expander("Diagnostics — race chunks", expanded=False):
        st.write(f"Detected {len(races)} race(s)")
        for hdr, chunk in races[:20]:
            st.code(f"{hdr}\n" + "\n".join(chunk.splitlines()[:5]), language="text")

    results: list = []
    seen: set = set()
    for header, chunk in races:
        sig = (header.strip(), hash(chunk))
        if sig in seen:
            continue
        seen.add(sig)
        df, meta = analyze_single_race_text(chunk, weights)
        if df.empty:
            df = pd.DataFrame(columns=["Prog", "Horse", "ML", "Style", "PrimePower", "Speed", "DaysOff", "Rating"])
        results.append((header.strip(), df, meta, chunk))
    return results


# ──────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────
def styled_df(df: pd.DataFrame):
    """Apply gold/silver/bronze row highlights for top 3 (1-based index)."""
    medals = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}

    def row_style(row):
        color = medals.get(row.name)
        if color:
            return [f"background-color: {color}; font-weight: bold"] * len(row)
        return [""] * len(row)

    fmt = {}
    for col, spec in [("Rating", "{:.1f}"), ("PrimePower", "{:.0f}"), ("Speed", "{:.0f}")]:
        if col in df.columns:
            fmt[col] = spec

    s = df.style.apply(row_style, axis=1)
    return s.format(fmt) if fmt else s


def render_pace_breakdown(style_counts: dict):
    styles = ["E", "EP", "P", "S", "NA"]
    cols = st.columns(len(styles))
    for col, sty in zip(cols, styles):
        with col:
            st.metric(STYLE_LABELS[sty], style_counts.get(sty, 0))


def format_days_off(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{int(x)}d"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Scoring Weights")
    weights = {
        "prime_power": st.slider("Prime Power", 0.0, 3.0, 1.0, 0.1, key="w_pp"),
        "speed": st.slider("Speed Figure", 0.0, 3.0, 1.0, 0.1, key="w_spd"),
        "style": st.slider("Running Style", 0.0, 2.0, 0.5, 0.1, key="w_sty"),
    }
    st.divider()
    st.subheader("Adjustments")
    weights["layoff"] = st.checkbox(
        "Apply layoff penalty", value=True, key="w_layoff",
        help="Reduce score for horses off > 45 days; small bonus for recently raced horses",
    )
    weights["pace_scenario"] = st.checkbox(
        "Pace scenario adjustments", value=True, key="w_pace",
        help="Auto-adjust style ratings based on field pace composition",
    )
    st.divider()
    show_detail = st.checkbox("Show Jockey / Trainer", value=False)
    st.divider()
    st.caption(
        "**Rating formula:**\n\n"
        "`(PP_w × PrimePower`\n"
        "`+ Spd_w × Speed`\n"
        "`+ Sty_w × StyleRating`\n"
        "`+ PaceAdj) × LayoffFactor`"
    )

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Brisnet PDF (text-based, not scanned)",
    type=["pdf"],
    key="uploader_main",
)

if uploaded:
    with st.spinner("Parsing & scoring…"):
        results = analyze_pdf_all(uploaded.read(), weights)

    if not results:
        st.error("No races parsed. Open the diagnostics above to inspect what was detected.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for i, (hdr, df, meta, chunk) in enumerate(results):
            with tabs[i]:
                # ── Pace scenario banner ──────────────────────────
                if meta:
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.metric("Pace Scenario", meta.get("scenario", "—"))
                        st.caption(meta.get("advice", ""))
                    with c2:
                        st.caption("Field pace breakdown")
                        render_pace_breakdown(meta.get("style_counts", {}))

                    # Warn if extraction quality looks poor
                    pp_defaults = meta.get("pp_defaults", 0)
                    horse_count = meta.get("horse_count", 1)
                    if horse_count > 0 and pp_defaults / horse_count > 0.7:
                        st.warning(
                            f"Prime Power defaulted for {pp_defaults}/{horse_count} horses — "
                            "PDF format may not match expected Brisnet layout."
                        )
                    st.divider()

                # ── Rankings table ────────────────────────────────
                st.subheader(f"{hdr} — Rankings")

                if df.empty:
                    st.warning("No horses parsed for this race.")
                else:
                    df_show = df.copy().reset_index(drop=True)

                    base_cols = ["Prog", "Horse", "ML", "Style", "PrimePower", "Speed", "DaysOff", "Rating"]
                    extra_cols = ["Jockey", "Trainer"] if show_detail else []
                    display_cols = [c for c in base_cols + extra_cols if c in df_show.columns]

                    df_display = df_show[display_cols].copy()
                    df_display.index = range(1, len(df_display) + 1)

                    if "DaysOff" in df_display.columns:
                        df_display["DaysOff"] = df_display["DaysOff"].apply(format_days_off)

                    st.dataframe(styled_df(df_display), use_container_width=True, height=440)

                    # ── Top picks summary ─────────────────────────
                    if len(df_show) >= 1:
                        st.markdown("**Top Picks:**")
                        medals_emoji = ["🥇", "🥈", "🥉"]
                        pick_cols = st.columns(min(3, len(df_show)))
                        for j, col in enumerate(pick_cols):
                            row = df_show.iloc[j]
                            with col:
                                st.markdown(f"{medals_emoji[j]} **#{row['Prog']} {row['Horse']}**")
                                extras = []
                                if row.get("ML", "—") != "—":
                                    extras.append(f"ML: {row['ML']}")
                                extras.append(f"Style: {row['Style']}")
                                extras.append(f"Rating: {row['Rating']:.1f}")
                                st.caption(" · ".join(extras))

                    # ── Download ──────────────────────────────────
                    st.download_button(
                        label=f"⬇ Download {hdr} CSV",
                        data=df_show.to_csv(index=False).encode("utf-8"),
                        file_name=f"{hdr.replace(' ', '_')}_rankings.csv",
                        mime="text/csv",
                        key=f"dl_{i}",
                    )

                    with st.expander("Raw extracted text for this race", expanded=False):
                        st.code(chunk[:3000], language="text")
else:
    st.info("Upload a Brisnet PDF to begin.")
