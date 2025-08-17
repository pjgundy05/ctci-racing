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
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+\s+[A-Za-z][^\n]+?\()")

# ---------- Helpers ----------
def has_phrase(text, phrase): 
    return phrase.lower() in text.lower()

def regex_found(text, pat): 
    return re.search(pat, text, flags=re.IGNORECASE) is not None

def safe_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default

# ---------- Race meta ----------
def infer_race_meta(text):
    t = text.lower()
    if "(t)" in t or "turf" in t:
        surface = "TURF"
    elif " all weather" in t or " synthetic" in t or " aw " in t:
        surface = "AW"
    else:
        surface = "DIRT"

    head = "\n".join(text.splitlines()[:150]).lower()
    furlongs = None
    mf = re.search(r"\b(\d(?:\.\d)?)\s*f\b", head)
    if mf:
        furlongs = float(mf.group(1))
    else:
        if "1 1/16" in head or "1ˆ" in head:
            furlongs = 8.5
        elif "1 1/8" in head:
            furlongs = 9.0
        elif "1 3/8" in head:
            furlongs = 11.0
        elif " 1m" in head or " 1 mile" in head:
            furlongs = 8.0
        elif any(s in head for s in ["5½", "5 1/2"]):
            furlongs = 5.5
    if furlongs is None:
        if any(tok in head for tok in ["5f","6f","7f","5½","5 1/2"]):
            furlongs = 6.0
        else:
            furlongs = 8.0

    dist_type = "SPRINT" if furlongs < 8.0 else "ROUTE"
    return surface, dist_type, furlongs

# ---------- Parsing horses ----------
def parse_style(first_line):
    m = re.search(r"\((E\/P|E|P|S)\s*(\d+)?\)", first_line)
    if not m:
        return DEFAULT_STYLE
    style = "EP" if "E/P" in m.group(1) else m.group(1)
    rating = safe_int(m.group(2), DEFAULT_STYLE[1])
    return style, rating

def extract_prime_power(block):
    # More forgiving regex variants to tolerate OCR quirks
    m = re.search(r"Prime\s*Power[:\s]*([\d\.]+)", block, flags=re.IGNORECASE)
    if m and m.group(1):
        return safe_int(m.group(1), DEFAULT_PP)
    m2 = re.search(r"(?:PP|Prime\s*Powe?r)\D{0,6}([\d]{2,3})", block, flags=re.IGNORECASE)
    if m2 and m2.group(1):
        return safe_int(m2.group(1), DEFAULT_PP)
    return DEFAULT_PP  # never crash

def detect_sharp_work(block):
    # bullet-like patterns
    return regex_found(block, r"\bB\s*1\/\d+") or regex_found(block, r"\b1\/\d+\b")

def split_pdf_into_races(text):
    chunks = re.split(r"\bRace\s+\d+\b", text, flags=re.IGNORECASE)
    headers = re.findall(r"\bRace\s+\d+\b", text, flags=re.IGNORECASE)
    races = []
    for i, chunk in enumerate(chunks[1:], start=0):
        races.append((headers[i], chunk))
    if not races:
        races = [("Race", text)]
    return races

def extract_horses_from_text(text):
    blocks = HORSE_SPLIT.split(text or "")
    horses = []

    for raw in blocks:
        block = (raw or "").strip()
        if not block:
            continue

        lines = block.splitlines()
        first = lines[0] if lines else ""

        # Heuristic: if it doesn't look like a horse card, skip it
        looks_like_card = (
            re.search(r"^\s*\d+\s+[A-Za-z'’\-\. ]+\s*\(", first) or
            ("Prime Power" in block) or
            re.search(r"\((E\/P|E|P|S)\s*\d*", first or "", flags=re.IGNORECASE)
        )
        if not looks_like_card:
            continue

        # Name
        nm = re.match(r"\s*\d+\s+([A-Za-z'’\-\. ]+)\s*\(", first)
        name = nm.group(1).strip() if nm else "Unknown"

        # Style
        style, style_rt = parse_style(first)

        # Prime Power (safe)
        pp = extract_prime_power(block)
        if pp is None:
            pp = DEFAULT_PP

        horses.append({
            "Horse": name,
            "Block": block,
            "FirstLine": first,
            "Style": style,
            "StyleRating": style_rt,
            "PrimePower": pp,
        })

    # Deduplicate by Horse name (PDF artifacts can duplicate)
    return pd.DataFrame(horses).drop_duplicates(subset=["Horse"]).to_dict("records")

# ---------- Pace ----------
def build_pace_context(horses, hot_threshold=4):
    pressers = [h for h in horses if h["Style"] in ("E","EP") and h["StyleRating"] >= 5]
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
        if "Sire Stats:" in block and "Turf" in block: pts += w["fit_turf_sire"]
        if "Dam'sSire" in block and "Turf" in block: pts += w["fit_turf_damsire"]
    elif surface == "DIRT":
        if has_phrase(block, "Best Dirt Speed"): pts += w["fit_dirt_best"]
        if regex_found(block, r"\bFst\s*\(\d+") or " Fast " in block: pts += w["fit_dirt_fast"]
        if regex_found(block, r"\bOff\s*\(\d+") or "Mud" in block: pts += w["fit_dirt_mud"]
    else:
        if "AW" in block: pts += w["fit_aw_ok"]
    return pts

def score_class_intent(block, w):
    pts = 0
    if has_phrase(block, "Drops in class"): pts += w["class_drop"]
    if has_phrase(block, "1st Time Clmg"): pts += w["first_time_claim"]
    if has_phrase(block, "First off claim") or has_phrase(block, "1st after clm"): pts += w["first_off_claim"]
    if has_phrase(block, "Class rise"): pts += w["class_rise"]
    if has_phrase(block, "Protected spot") or " OC" in block or "OC " in block: pts += w["protected_spot"]
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
    if has_phrase(block, "First-time Lasix") or " L1" in block: pts += w["equip_lasix"]
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
    surface, dist_type, furlongs = infer_race_meta(text)
    horses = extract_horses_from_text(text)
    if not horses:
        empty_df = pd.DataFrame()
        meta = {
            "surface": surface, "distance_type": dist_type, "furlongs": furlongs,
            "pace_pressers": 0, "lone_speed": False, "hot_pace": False
        }
        return empty_df, meta

    pace = build_pace_context(horses, weights["pace_hot_threshold"])
    rows = []
    for h in horses:
        block = h["Block"]
        score = 0.0
        score += weights["w_prime_power"] * h["PrimePower"]
        score += score_class_intent(block, weights)
        score += score_form(block, weights)
        score += score_equipment(block, weights)
        score += score_connections(block, weights)
        score += score_surface_distance_fit(block, surface, weights)
        score += score_penalties(block, weights)
        score += score_pace(h["Style"], dist_type, surface, pace, weights)
        rows.append({
            "Horse": h["Horse"], "Style": h["Style"], "StyleRating": h["StyleRating"],
            "PrimePower": h["PrimePower"], "FinalScore": round(score, 2)
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["Horse"]).sort_values(
        ["FinalScore", "PrimePower"], ascending=[False, False]
    ).reset_index(drop=True)

    meta = {
        "surface": surface, "distance_type": dist_type, "furlongs": furlongs,
        "pace_pressers": pace["num_pressers"], "lone_speed": pace["lone_speed"], "hot_pace": pace["hot"]
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

# ---------- Multi-race analyze with protection ----------
def analyze_pdf_all(file_bytes, weights):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    races = split_pdf_into_races(full_text)
    results = []
    for header, chunk in races:
        try:
            df, meta = analyze_single_race_text(chunk, weights)
            if not df.empty:
                results.append((header.strip(), df, meta))
        except Exception:
            # Skip malformed race chunks silently (optional: collect debug)
            continue

    if not results:
        # Fallback single pass
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
show_debug = st.sidebar.checkbox("Show debug info", value=False)

w = {}
for k, v in DEF_WEIGHTS.items():
    if isinstance(v, (int, float)):
        if k.startswith("w_"):
            w[k] = st.sidebar.slider(k, 0.0, 2.0, float(v), 0.05)
        elif "pen_" in k:
            w[k] = st.sidebar.slider(k, -20, 0, int(v), 1)
        elif "pace_hot_threshold" in k:
            w[k] = st.sidebar.slider(k, 2, 8, int(v), 1)
        else:
            maxv = 30 if v >= 10 else 10
            step = 1 if isinstance(v, int) else 0.5
            w[k] = st.sidebar.slider(k, -10 if v < 0 else 0, maxv, v, step)
    else:
        w[k] = v

uploaded = st.file_uploader("Upload Brisnet/Equibase PDF", type=["pdf"])

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
                    st.dataframe(df, use_container_width=True, height=420)
                with topR:
                    st.subheader("Race Meta (inferred)")
                    st.markdown(f"- **Surface:** `{meta['surface']}`")
                    st.markdown(f"- **Distance:** `{meta['distance_type']}` (~{meta['furlongs']}f)")
                    st.markdown(f"- **Pace pressers (E/EP≥5):** `{meta['pace_pressers']}`")
                    st.markdown(f"- **Lone Speed:** `{meta['lone_speed']}`")
                    st.markdown(f"- **Hot Pace:** `{meta['hot_pace']}`")

                    if show_debug:
                        st.caption("If something looks off, toggle weights or share a sample page to adjust regex.")

                # Downloads
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ CSV (rankings)", 
                        df.to_csv(index=False).encode(),
                        file_name=f"{name.replace(' ','_')}_rankings.csv", 
                        mime="text/csv"
                    )
                with dl2:
                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Rankings")
                    st.download_button(
                        "⬇️ Excel (rankings)", 
                        xbuf.getvalue(),
                        file_name=f"{name.replace(' ','_')}_rankings.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.markdown("---")
                st.subheader("🎟️ Ticket Builder")
                colA, colB, colC = st.columns([1.5, 1, 1])
                with colA:
                    budget = st.number_input("Total budget ($)", min_value=5, max_value=1000, step=1, value=24)
                with colB:
                    risk = st.selectbox("Risk profile", ["conservative", "balanced", "aggressive"], index=1)
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
                        st.dataframe(tdf, use_container_width=True)
                        st.download_button(
                            "⬇️ CSV (tickets)", 
                            tdf.to_csv(index=False).encode(),
                            file_name=f"{name.replace(' ','_')}_tickets_{risk}_{budget}.csv",
                            mime="text/csv"
                        )
        st.info("Tip: In Safari → Share → **Add to Home Screen** to use like an app.")
else:
    st.info("Upload a PPs PDF to begin.")