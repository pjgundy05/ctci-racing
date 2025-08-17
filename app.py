import io, re, math
import pdfplumber
import pandas as pd
import streamlit as st

# =========================
# Default weights (tunable live in sidebar)
# =========================
DEF_WEIGHTS = {
    "w_prime_power": 1.00,

    # Class / Intent
    "class_drop": 18,
    "first_time_claim": 16,
    "first_off_claim": 12,
    "class_rise": -6,
    "protected_spot": 4,

    # Form cycle
    "highest_last_fig": 10,
    "last_out_2nd": 6,
    "last_out_3rd": 4,
    "hot_trn_jky": 6,
    "second_off_layoff": 8,
    "layoff_penalty": -6,   # > 60 days (heuristic via PPs text)
    "sharp_workout": 5,     # bullet-like B 1/xx or 1/xx

    # Surface / Distance fit
    "best_dist": 8,
    "fit_turf_best": 10, "fit_turf_ok": 6, "fit_turf_sire": 5, "fit_turf_damsire": 3,
    "fit_dirt_best": 10, "fit_dirt_fast": 6, "fit_dirt_mud": 4,
    "fit_aw_ok": 6,

    # Equipment / Meds
    "equip_blinkers": 6, "equip_lasix": 6, "equip_geld": 4,

    # Connections
    "conn_hi_jky": 4, "conn_hi_trn": 6, "conn_hi_jky_switch": 6, "conn_shipper": 3,

    # Penalties
    "pen_one_win": -10, "pen_poor_trn": -8, "pen_poor_jky": -6,
    "pen_always_back": -6, "pen_bad_track": -4, "pen_bad_post": -3,

    # Pace
    "pace_lone_speed": 10,
    "pace_speed_help": 5,         # soft/avg pressure favors E/EP
    "pace_closer_help": 6,        # hot pressure favors P/S
    "pace_hot_threshold": 4,      # >= this many E/EP(>=5) = hot
}

DEFAULT_PP = 100
DEFAULT_STYLE = ("P", 3)

# Allow program numbers like "1", "10", or "1A"
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+[A-Z]?\s+[A-Za-z][^\n]+?\()")

# ---------- Helpers ----------
def has_phrase(text, phrase):
    return phrase.lower() in (text or "").lower()

def regex_found(text, pat):
    return re.search(pat, text or "", flags=re.IGNORECASE) is not None

def safe_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default

# ---------- Fraction/Distance helpers ----------
UNICODE_FRACS = {
    "½": 0.5, "¼": 0.25, "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}

def parse_mixed_number(s: str) -> float:
    """
    Parse '1 1/16', '1-1/16', '1 1/8', '5 1/2', '5½', '1¾', '1/16', etc. -> float miles or furlongs
    """
    s = (s or "").strip()
    # replace unicode vulgar fractions with ascii fraction
    for uf, val in UNICODE_FRACS.items():
        if uf in s:
            # '5½' -> '5 + 1/2'
            parts = re.split(r"\s*", s.replace(uf, f" + {val}"), maxsplit=0)
            try:
                return sum(float(p) for p in "".join(parts).split("+"))
            except Exception:
                # fall through to generic parsing
                pass

    # patterns: "A B/C", "A-B/C", "B/C", "A"
    m = re.match(r"^\s*(\d+)\s*[- ]\s*(\d+)\s*/\s*(\d+)\s*$", s)  # A B/C or A-B/C
    if m:
        a, b, c = map(int, m.groups())
        return a + b / c
    m = re.match(r"^\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*$", s)
    if m:
        a, b, c = map(int, m.groups())
        return a + b / c
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)  # B/C
    if m:
        b, c = map(int, m.groups())
        return b / c
    m = re.match(r"^\s*(\d+)\s*$", s)  # A
    if m:
        return float(m.group(1))
    # fallback
    try:
        return float(s)
    except Exception:
        return None

def miles_to_furlongs(miles: float) -> float:
    return round(miles * 8.0, 2) if miles is not None else None

def furlongs_from_text(segment: str):
    """
    Return (furlongs: float, src_text: str) if we can parse "About 5 1/2 Furlongs" or "1 1/16 Miles (Turf)" etc.
    """
    seg = (segment or "")
    # About/abt + Furlongs
    m = re.search(r"(?i)\b(about|abt)\s+([\d½¼¾⅛⅜⅝⅞]+(?:\s+\d+/\d+)?)\s*furlongs?\b", seg)
    if m:
        num = parse_mixed_number(m.group(2))
        return (float(num), m.group(0)) if num is not None else (None, None)

    m = re.search(r"(?i)\b([\d½¼¾⅛⅜⅝⅞]+(?:\s+\d+/\d+)?)\s*furlongs?\b", seg)
    if m:
        num = parse_mixed_number(m.group(1))
        return (float(num), m.group(0)) if num is not None else (None, None)

    # Miles with fractions
    m = re.search(r"(?i)\b(about|abt)?\s*([\d½¼¾⅛⅜⅝⅞]+(?:\s+\d+/\d+)?)\s*miles?\b", seg)
    if m:
        num_miles = parse_mixed_number(m.group(2))
        f = miles_to_furlongs(num_miles)
        return (f, m.group(0)) if f is not None else (None, None)

    # Short forms: '1m', '1 1/16m', '7f'
    m = re.search(r"(?i)\b([\d]+(?:\s+\d+/\d+)?)\s*m\b", seg)  # miles shorthand '1m'
    if m:
        num_miles = parse_mixed_number(m.group(1))
        f = miles_to_furlongs(num_miles)
        return (f, m.group(0)) if f is not None else (None, None)
    m = re.search(r"(?i)\b([\d]+(?:\.\d+)?)\s*f\b", seg)  # '6f', '5.5f'
    if m:
        return (float(m.group(1)), m.group(0))

    return (None, None)

def surface_from_text(segment: str):
    """
    Infer surface from parenthetical or keywords near the header.
    """
    seg = (segment or "").lower()
    # Parentheses markers "(Turf)", "(Inner Turf)", "(Dirt)"
    pm = re.search(r"\(\s*(inner|outer|widener)?\s*turf\s*\)", seg)
    if pm:
        return "TURF", pm.group(0)
    if re.search(r"\(\s*([^\)]*tapeta|polytrack|synthetic|all\s*weather)\s*\)", seg):
        return "AW", "parenthetical AW"
    if re.search(r"\(\s*dirt\s*\)", seg):
        return "DIRT", "(Dirt)"

    # Keywords nearby
    if any(k in seg for k in ["inner turf", "outer turf", "widener turf", "turf course", "turf rail"]):
        return "TURF", "turf keyword"
    if any(k in seg for k in ["tapeta", "polytrack", "synthetic", "all weather", "aw "]):
        return "AW", "aw keyword"
    if any(k in seg for k in ["main track", "fast track", "sloppy", "muddy", "good (dirt)"]):
        return "DIRT", "dirt keyword"

    # If we see lots of 'turf' tokens at all
    if "turf" in seg:
        return "TURF", "turf token"
    return "DIRT", "default dirt"

# ---------- Race meta ----------
def infer_race_meta(text):
    # Look only at the top ~3000 chars of the race chunk for header info
    header = "\n".join((text or "").splitlines()[:120]).strip()
    f, f_src = furlongs_from_text(header)
    surface, s_src = surface_from_text(header)

    # Robust fallback: scan a larger window if header was unhelpful
    if f is None or surface is None:
        window = (text or "")[:4000]
        if f is None:
            f, f_src2 = furlongs_from_text(window)
            if f is not None: f_src = f_src2
        if surface is None:
            surface, s_src2 = surface_from_text(window)
            if surface is not None: s_src = s_src2

    # Final fallbacks
    if f is None:
        # assume one mile if we can't tell
        f, f_src = 8.0, "fallback 1m"
    if surface is None:
        surface, s_src = "DIRT", "fallback dirt"

    dist_type = "SPRINT" if f < 8.0 else "ROUTE"
    return surface, dist_type, f, {"distance_src": f_src, "surface_src": s_src}

# ---------- Parsing horses ----------
def parse_style(first_line):
    m = re.search(r"\((E\/P|E|P|S)\s*(\d+)?\)", first_line or "")
    if not m:
        return DEFAULT_STYLE
    style = "EP" if "E/P" in m.group(1) else m.group(1)
    rating = safe_int(m.group(2), DEFAULT_STYLE[1])
    return style, rating

def parse_program_number(first_line):
    m = re.match(r"\s*(\d+[A-Z]?)\s+", first_line or "")
    return m.group(1).strip() if m else ""

def extract_prime_power(block):
    # Forgiving patterns (OCR often mangles spacing/colon)
    m = re.search(r"Prime\s*Power[:\s]*([\d\.]+)", block or "", flags=re.IGNORECASE)
    if m and m.group(1):
        return safe_int(m.group(1), DEFAULT_PP)
    m2 = re.search(r"(?:PP|Prime\s*Powe?r)\D{0,6}([\d]{2,3})", block or "", flags=re.IGNORECASE)
    if m2 and m2.group(1):
        return safe_int(m2.group(1), DEFAULT_PP)
    return DEFAULT_PP  # fallback, never crash

def detect_sharp_work(block):
    return regex_found(block, r"\bB\s*1\/\d+") or regex_found(block, r"\b1\/\d+\b")

def extract_horses_from_text(text):
    blocks = HORSE_SPLIT.split(text or "")
    horses = []

    for raw in blocks:
        block = (raw or "").strip()
        if not block:
            continue

        lines = block.splitlines()
        first = lines[0] if lines else ""

        # Looks like a horse card?
        looks_like_card = (
            re.search(r"^\s*\d+[A-Z]?\s+[A-Za-z'’\-\. ]+\s*\(", first or "") or
            ("Prime Power" in block) or
            re.search(r"\((E\/P|E|P|S)\s*\d*", first or "", flags=re.IGNORECASE)
        )
        if not looks_like_card:
            continue

        prog = parse_program_number(first)
        nm = re.match(r"\s*\d+[A-Z]?\s+([A-Za-z'’\-\. ]+)\s*\(", first or "")
        name = nm.group(1).strip() if nm else "Unknown"
        style, style_rt = parse_style(first or "")
        pp = extract_prime_power(block) or DEFAULT_PP

        horses.append({
            "Prog": prog,
            "Horse": name,
            "Block": block,
            "FirstLine": first,
            "Style": style,
            "StyleRating": style_rt,
            "PrimePower": pp,
        })

    # Deduplicate by Program + Horse (safer than just Horse)
    if not horses:
        return []
    df = pd.DataFrame(horses).drop_duplicates(subset=["Prog","Horse"])
    return df.to_dict("records")

# ---------- Pace ----------
def build_pace_context(horses, hot_threshold=4):
    pressers = [h for h in horses if h["Style"] in ("E","EP") and (h["StyleRating"] or 0) >= 5]
    return {
        "num_pressers": len(pressers),
        "lone_speed": len(pressers) == 1,
        "hot": len(pressers) >= hot_threshold
    }

# ---------- Scoring ----------
def score_surface_distance_fit(block, surface, w):
    pts = 0
    if has_phrase(block, "Best Speed at Dist"): pts += w["best_dist"]
    if surface == "TURF":
        if has_phrase(block, "Best Turf Speed"): pts += w["fit_turf_best"]
        if has_phrase(block, "Turf starts"): pts += w["fit_turf_ok"]
        if "Sire Stats:" in (block or "") and "Turf" in (block or ""): pts += w["fit_turf_sire"]
        if "Dam'sSire" in (block or "") and "Turf" in (block or ""): pts += w["fit_turf_damsire"]
    elif surface == "DIRT":
        if has_phrase(block, "Best Dirt Speed"): pts += w["fit_dirt_best"]
        if regex_found(block, r"\bFst\s*\(\d+") or " Fast " in (block or ""): pts += w["fit_dirt_fast"]
        if regex_found(block, r"\bOff\s*\(\d+") or "Mud" in (block or ""): pts += w["fit_dirt_mud"]
    else:
        if "AW" in (block or "") or "Tapeta" in (block or "") or "Polytrack" in (block or ""):
            pts += w["fit_aw_ok"]
    return pts

def score_class_intent(block, w):
    pts = 0
    if has_phrase(block, "Drops in class"): pts += w["class_drop"]
    if has_phrase(block, "1st Time Clmg"): pts += w["first_time_claim"]
    if has_phrase(block, "First off claim") or has_phrase(block, "1st after clm"): pts += w["first_off_claim"]
    if has_phrase(block, "Class rise"): pts += w["class_rise"]
    if has_phrase(block, "Protected spot") or " OC" in (block or "") or "OC " in (block or ""): pts += w["protected_spot"]
    return pts

def score_form(block, w):
    pts = 0
    if has_phrase(block, "Highest last race speed rating"): pts += w["highest_last_fig"]
    if has_phrase(block, "Finished 2nd in last race"): pts += w["last_out_2nd"]
    if has_phrase(block, "Finished 3rd"): pts += w["last_out_3rd"]
    if has_phrase(block, "Hot Tnr/Jky combo"): pts += w["hot_trn_jky"]
    if has_phrase(block, "2nd off layoff") or has_phrase(block, "Second off layoff"): pts += w["second_off_layoff"]
    if has_phrase(block, "Has not raced for more than 2 months"): pts += w["layoff_penalty"]
    if detect_sharp_work(block): pts += w["sharp_workout"]
    return pts

def score_equipment(block, w):
    pts = 0
    if has_phrase(block, "Blinkers On") or has_phrase(block, "Blinkers on"): pts += w["equip_blinkers"]
    if has_phrase(block, "Blinkers Off") or has_phrase(block, "Blinkers off"): pts += w["equip_blinkers"]
    if has_phrase(block, "First-time Lasix") or " L1" in (block or ""): pts += w["equip_lasix"]
    if has_phrase(block, "Gelded"): pts += w["equip_geld"]
    return pts

def score_connections(block, w):
    pts = 0
    if has_phrase(block, "High % trainer"): pts += w["conn_hi_trn"]
    if has_phrase(block, "High % jockey"): pts += w["conn_hi_jky"]
    if has_phrase(block, "Switches to a high % jockey"): pts += w["conn_hi_jky_switch"]
    if has_phrase(block, "Shipper"): pts += w["conn_shipper"]
    return pts

def score_penalties(block, w):
    pts = 0
    for phrase, val in [
        ("only 1 win", w["pen_one_win"]),
        ("Poor trainer win%", w["pen_poor_trn"]),
        ("Poor jockey win%", w["pen_poor_jky"]),
        ("Always far back", w["pen_always_back"]),
        ("Poor record at this track", w["pen_bad_track"]),
    ]:
        if has_phrase(block, phrase): pts += val
    if regex_found(block, r"Post Position .* poor win%"): pts += w["pen_bad_post"]
    return pts

def score_pace(style, dist_type, surface, pace_ctx, w):
    adj = 0
    if pace_ctx["lone_speed"] and style in ("E", "EP"):
        adj += w["pace_lone_speed"]
    if dist_type == "SPRINT":
        if not pace_ctx["hot"] and style in ("E", "EP"): adj += w["pace_speed_help"]
        if pace_ctx["hot"] and style in ("P", "S"): adj += w["pace_closer_help"]
    else:
        if pace_ctx["hot"] and style in ("P", "S"): adj += w["pace_closer_help"]
        elif not pace_ctx["hot"] and style in ("E", "EP"): adj += w["pace_speed_help"]
    if surface == "TURF" and pace_ctx["hot"] and style in ("P", "S"):
        adj += 2
    return adj

# ---------- Per-race pipeline ----------
def analyze_single_race_text(text, weights):
    surface, dist_type, furlongs, meta_src = infer_race_meta(text)
    horses = extract_horses_from_text(text)
    if not horses:
        empty_df = pd.DataFrame()
        meta = {
            "surface": surface, "distance_type": dist_type, "furlongs": furlongs,
            "pace_pressers": 0, "lone_speed": False, "hot_pace": False, **meta_src
        }
        return empty_df, meta

    pace = build_pace_context(horses, weights["pace_hot_threshold"])
    rows = []
    for h in horses:
        block = h["Block"]
        score = 0.0
        score += weights["w_prime_power"] * (h["PrimePower"] or DEFAULT_PP)
        score += score_class_intent(block, weights)
        score += score_form(block, weights)
        score += score_equipment(block, weights)
        score += score_connections(block, weights)
        score += score_surface_distance_fit(block, surface, weights)
        score += score_penalties(block, weights)
        score += score_pace(h["Style"], dist_type, surface, pace, weights)
        rows.append({
            "Prog": h["Prog"],
            "Horse": h["Horse"],
            "Style": h["Style"],
            "StyleRating": h["StyleRating"],
            "PrimePower": h["PrimePower"],
            "FinalScore": round(score, 2)
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["Prog","Horse"]).sort_values(
        ["FinalScore", "PrimePower"], ascending=[False, False]
    ).reset_index(drop=True)

    meta = {
        "surface": surface, "distance_type": dist_type, "furlongs": furlongs,
        "pace_pressers": pace["num_pressers"], "lone_speed": pace["lone_speed"], "hot_pace": pace["hot"], **meta_src
    }
    return df, meta

# ---------- Ticket builder ----------
def build_tickets(df, meta, budget, risk="balanced"):
    if df.empty or budget <= 0:
        return {}, []

    if risk == "conservative":
        topN = 3; win_frac, exa_frac, tri_frac = 0.5, 0.35, 0.15
    elif risk == "aggressive":
        topN = min(6, len(df)); win_frac, exa_frac, tri_frac = 0.3, 0.35, 0.35
    else:
        topN = min(5, len(df)); win_frac, exa_frac, tri_frac = 0.4, 0.35, 0.25

    horses = df["Horse"].tolist()
    top = horses[:topN]

    sub = df[df["Horse"].isin(top)].copy()
    sub["w"] = sub["FinalScore"] - sub["FinalScore"].min() + 1.0
    sub["w"] = sub["w"] / sub["w"].sum()

    win_bank = math.floor(budget * win_frac)
    win_tickets = [(r["Horse"], round(win_bank * r["w"])) for _, r in sub.iterrows()]
    drift = win_bank - sum(x[1] for x in win_tickets)
    if drift != 0 and win_tickets:
        win_tickets[0] = (win_tickets[0][0], win_tickets[0][1] + drift)

    exa_bank = math.floor(budget * exa_frac)
    exa_unit = 1 if exa_bank >= 6 else 0.5
    A = top[:2] if risk != "aggressive" else top[:3]
    B = top[:min(5, len(top))]
    exa_combos = [(a, b) for a in A for b in B if b != a]
    exa_tickets, spent = [], 0.0
    for (a, b) in exa_combos:
        if spent + exa_unit <= exa_bank:
            exa_tickets.append((a, b, exa_unit)); spent += exa_unit
        else:
            break

    tri_bank = budget - win_bank - math.floor(spent)
    tri_unit = 0.5 if tri_bank >= 6 else 0.1
    A = top[:2] if risk == "conservative" else top[:3]
    B = top[:min(5, len(top))]
    C = top[:min(6, len(horses))] if risk != "conservative" else top[:min(5, len(horses))]
    tri_tickets, t_spent = [], 0.0
    for a in A:
        for b in B:
            if b == a: continue
            for c in C:
                if c in (a, b): continue
                if t_spent + tri_unit <= tri_bank:
                    tri_tickets.append((a, b, c, tri_unit)); t_spent += tri_unit
                else:
                    break
            if t_spent + tri_unit > tri_bank:
                break

    summary = {
        "budget": budget,
        "win_bank": win_bank,
        "exa_bank": math.floor(spent),
        "tri_bank": math.floor(t_spent),
        "risk": risk,
        "meta": meta
    }
    flat = []
    for h, amt in win_tickets:
        if amt > 0: flat.append({"Type": "WIN", "Legs": h, "Wager": amt})
    for a, b, u in exa_tickets:
        flat.append({"Type": "EXA", "Legs": f"{a}-{b}", "Wager": u})
    for a, b, c, u in tri_tickets:
        flat.append({"Type": "TRI", "Legs": f"{a}-{b}-{c}", "Wager": u})
    return summary, flat

# ---------- Robust race splitting ----------
def split_pdf_into_races_robust(full_text):
    """
    Collapses multi-page races into single chunks: find 'Race N' that’s followed
    near-term by 'Post Time' or 'Purse', then slice until next 'Race M'.
    """
    matches = list(re.finditer(r"\bRace\s+(\d+)\b", full_text or "", flags=re.IGNORECASE))
    valid = []
    for m in matches:
        num = int(m.group(1))
        if 1 <= num <= 20:
            window = (full_text or "")[m.start(): m.start() + 4000]
            if ("Post Time" in window) or ("Purse" in window) or ("Furlongs" in window) or ("Miles" in window):
                valid.append((num, m.start()))
    valid = sorted(valid, key=lambda x: x[1])

    races = []
    for i, (num, pos) in enumerate(valid):
        end = valid[i + 1][1] if i + 1 < len(valid) else len(full_text or "")
        races.append((f"Race {num}", (full_text or "")[pos:end]))
    return races

# ---------- Multi-race analyze with protection ----------
def analyze_pdf_all(file_bytes, weights):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    races = split_pdf_into_races_robust(full_text)
    results = []
    for i, (header, chunk) in enumerate(races):
        try:
            df, meta = analyze_single_race_text(chunk, weights)
            if not df.empty:
                results.append((header.strip(), df, meta))
        except Exception:
            continue

    if not results:
        try:
            df, meta = analyze_single_race_text(full_text, weights)
            if not df.empty:
                results = [("Race", df, meta)]
        except Exception:
            pass

    return results

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Thoroughbred Model + Ticket Builder", page_icon="🏇", layout="wide")
st.title("🏇 All-Surface Thoroughbred Model + Ticket Builder")
st.caption("PDF → per-race rankings for dirt/turf/AW, sprint/route. Pace-aware. Live weight sliders. Budget-aware tickets.")

# Sidebar controls
st.sidebar.header("Weights (live)")
show_debug = st.sidebar.checkbox("Show debug info", value=False, key="debug_toggle")

w = {}
for k, v in DEF_WEIGHTS.items():
    if isinstance(v, (int, float)):
        if k.startswith("w_"):
            w[k] = st.sidebar.slider(k, 0.0, 2.0, float(v), 0.05, key=f"w_{k}")
        elif "pen_" in k:
            w[k] = st.sidebar.slider(k, -20, 0, int(v), 1, key=f"w_{k}")
        elif "pace_hot_threshold" in k:
            w[k] = st.sidebar.slider(k, 2, 8, int(v), 1, key=f"w_{k}")
        else:
            maxv = 30 if v >= 10 else 10
            step = 1 if isinstance(v, int) else 0.5
            w[k] = st.sidebar.slider(k, -10 if v < 0 else 0, maxv, v, step, key=f"w_{k}")
    else:
        w[k] = v

uploaded = st.file_uploader("Upload Brisnet/Equibase PDF", type=["pdf"], key="uploader_main")

if uploaded:
    with st.spinner("Parsing & scoring races…"):
        results = analyze_pdf_all(uploaded.read(), w)

    if not results:
        st.error("No races parsed. Try another PDF (or enable debug).")
    else:
        if show_debug:
            st.info(f"Parsed {len(results)} race chunk(s) successfully.")
        tabs = st.tabs([name for name, _, _ in results])
        for i, (name, df, meta) in enumerate(results):
            with tabs[i]:
                topL, topR = st.columns([3, 2])
                with topL:
                    st.subheader(f"{name} — Rankings")
                    # Show program number & 1-based index (no '0' row)
                    df_display = df.copy()
                    df_display.index = range(1, len(df_display) + 1)  # 1-based
                    st.dataframe(
                        df_display[["Prog","Horse","Style","StyleRating","PrimePower","FinalScore"]],
                        use_container_width=True, height=420, key=f"df_rank_{i}"
                    )
                with topR:
                    st.subheader("Race Meta (inferred)")
                    st.markdown(f"- **Surface:** `{meta['surface']}`  _(source: {meta.get('surface_src','?')})_")
                    st.markdown(f"- **Distance:** `{meta['distance_type']}` (~{meta['furlongs']}f)  _(source: {meta.get('distance_src','?')})_")
                    st.markdown(f"- **Pace pressers (E/EP≥5):** `{meta['pace_pressers']}`")
                    st.markdown(f"- **Lone Speed:** `{meta['lone_speed']}`")
                    st.markdown(f"- **Hot Pace:** `{meta['hot_pace']}`")
                    if show_debug:
                        st.caption("If something looks off, adjust weights or share a sample page to tune regex.")

                # Downloads (unique keys per tab)
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ CSV (rankings)",
                        df.to_csv(index=False).encode(),
                        file_name=f"{name.replace(' ','_')}_rankings.csv",
                        mime="text/csv",
                        key=f"dl_csv_rankings_{i}"
                    )
                with dl2:
                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Rankings")
                    st.download_button(
                        "⬇️ Excel (rankings)",
                        xbuf.getvalue(),
                        file_name=f"{name.replace(' ','_')}_rankings.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_xlsx_rankings_{i}"
                    )

                st.markdown("---")
                st.subheader("🎟️ Ticket Builder")
                colA, colB, colC = st.columns([1.5, 1, 1])
                with colA:
                    budget = st.number_input(
                        "Total budget ($)", min_value=5, max_value=1000, step=1, value=24,
                        key=f"budget_{i}"
                    )
                with colB:
                    risk = st.selectbox(
                        "Risk profile", ["conservative", "balanced", "aggressive"], index=1,
                        key=f"risk_{i}"
                    )
                with colC:
                    go = st.button(f"Build tickets for {name}", key=f"go_{i}")

                if go:
                    summary, flat = build_tickets(df, meta, int(budget), risk=risk)
                    if not flat:
                        st.warning("No tickets created (check budget).")
                    else:
                        st.success(
                            f"Allocated ${summary['win_bank']} (WIN), "
                            f"${summary['exa_bank']} (EXA), ${summary['tri_bank']} (TRI)"
                        )
                        tdf = pd.DataFrame(flat)
                        tdf.index = range(1, len(tdf) + 1)  # 1-based
                        st.dataframe(tdf, use_container_width=True, key=f"df_tix_{i}")
                        st.download_button(
                            "⬇️ CSV (tickets)",
                            tdf.to_csv(index=False).encode(),
                            file_name=f"{name.replace(' ','_')}_tickets_{risk}_{budget}.csv",
                            mime="text/csv",
                            key=f"dl_csv_tickets_{i}_{risk}_{budget}"
                        )
        st.info("Tip: In Safari → Share → **Add to Home Screen** to use like an app.")
else:
    st.info("Upload a PPs PDF to begin.")