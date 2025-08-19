import streamlit as st
import pandas as pd
import pdfplumber
import re
import io

st.set_page_config(page_title="CTCI Horse Racing Model", layout="wide")

# ========== HELPER FUNCTIONS ==========

def safe_int(val, default=0):
    try:
        return int(val)
    except:
        return default

def extract_distance_surface(text):
    m = re.search(r"(\d+ (?:Furlongs|Mile))", text)
    return m.group(1) if m else "Unknown"

def extract_purse(text):
    m = re.search(r"Purse \$([0-9,]+)", text)
    return m.group(1) if m else "Unknown"

def extract_prime_power(block):
    m = re.search(r"Prime Power\s+(\d+)", block)
    return safe_int(m.group(1), 0) if m else 0

def extract_speed(block):
    m = re.search(r"Last Speed Rating[:\s]+(\d+)", block)
    return safe_int(m.group(1), 0) if m else 0

def extract_running_style(block):
    m = re.search(r"Running Style[:\s]+([A-Z]{1,2})", block)
    return m.group(1) if m else "NA"

def extract_prog(block):
    m = re.search(r"^\s*(\d+)[^\d\n]", block.strip())
    return m.group(1) if m else "?"

def extract_name(block):
    lines = block.strip().split('\n')
    if len(lines) >= 2:
        return lines[1].strip()
    return "Unknown"

def parse_race_blocks(text):
    # Detect race headers like:
    # 1\n6 Furlongs. S Mdn 90k Purse $90,000...
    race_header_pattern = re.compile(r"(?<=\n)(\d{1,2})\n((?:\d+ (?:Furlongs|Mile)).*?Purse \$\d[\d,]*)", re.DOTALL)
    headers = [(m.start(), m.group(1)) for m in race_header_pattern.finditer(text)]

    races = []
    for i in range(len(headers)):
        start = headers[i][0]
        end = headers[i+1][0] if i+1 < len(headers) else len(text)
        race_text = text[start:end].strip()
        race_number = headers[i][1]
        races.append((race_number, race_text))
    return races

def extract_horses_from_text(text):
    # Split on likely horse header blocks
    blocks = re.split(r"\n(?=\s*\d{1,2}\s)", text)
    horses = []

    for block in blocks:
        name = extract_name(block)
        if "Unknown" in name or len(name) < 2:
            continue  # Skip bad blocks
        horse = {
            "Prog": extract_prog(block),
            "Horse": name,
            "Style": extract_running_style(block),
            "PrimePower": extract_prime_power(block),
            "Speed": extract_speed(block),
        }
        horses.append(horse)
    return horses

def style_score(style):
    scores = {"E": 7, "EP": 5, "P": 3, "S": 1}
    return scores.get(style.upper(), 0)

def score_horse(h, weights):
    return (
        weights["PrimePower"] * h["PrimePower"]
        + weights["Speed"] * h["Speed"]
        + weights["Style"] * style_score(h["Style"])
    )

def analyze_single_race_text(text, weights):
    horses = extract_horses_from_text(text)
    if not horses:
        return pd.DataFrame(), {"DistanceSurface": "Unknown", "Purse": "Unknown"}

    for h in horses:
        h["StyleRating"] = style_score(h["Style"])
        h["Score"] = score_horse(h, weights)

    df = pd.DataFrame(horses)
    df.sort_values("Score", ascending=False, inplace=True)

    meta = {
        "DistanceSurface": extract_distance_surface(text),
        "Purse": extract_purse(text)
    }
    return df.reset_index(drop=True), meta

def analyze_pdf_all(pdf_data, weights):
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    race_blocks = parse_race_blocks(full_text)

    results = []
    for race_num, race_text in race_blocks:
        df, meta = analyze_single_race_text(race_text, weights)
        if not df.empty:
            results.append((f"Race {race_num}", df, meta))
    return results

# ========== STREAMLIT UI ==========

st.title("🏇 CTCI Horse Racing Model")

uploaded = st.file_uploader("Upload Brisnet PDF", type=["pdf"])

with st.expander("⚙️ Customize Weights", expanded=False):
    w_pp = st.slider("Prime Power Weight", 0, 10, 5)
    w_sp = st.slider("Speed Weight", 0, 10, 3)
    w_st = st.slider("Running Style Weight", 0, 10, 2)
    weights = {"PrimePower": w_pp, "Speed": w_sp, "Style": w_st}

if uploaded:
    with st.spinner("Analyzing races..."):
        results = analyze_pdf_all(uploaded.read(), weights)

    for race_name, df, meta in results:
        st.subheader(f"{race_name} — Rankings")
        st.caption(f"{meta['DistanceSurface']} | Purse: ${meta['Purse']}")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"{race_name.replace(' ', '_').lower()}_rankings.csv",
            mime="text/csv"
        )