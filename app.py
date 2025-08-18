import streamlit as st
import pdfplumber
import pandas as pd
import re, io

# =========================
# Utility functions
# =========================

DEFAULT_PP = 5

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

# --- Extractors ---

def extract_prime_power(block: str) -> int:
    m = re.search(r"Prime Power[:\s]+(\d+)", block)
    return safe_int(m.group(1), DEFAULT_PP)

def extract_running_style(block: str) -> str:
    m = re.search(r"Running Style[:\s]+([A-Z]+)", block)
    return m.group(1) if m else "NA"

def extract_speed(block: str) -> int:
    m = re.search(r"Speed[:\s]+(\d+)", block)
    return safe_int(m.group(1), 0)

# --- Horse parser ---

def extract_horses_from_text(text: str):
    horses = []
    blocks = text.split("\n\n")
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines: 
            continue
        first = lines[0]
        nm = re.match(r"\s*(\d+)\s+([A-Za-z'’\-\.]+(?:\s+[A-Za-z'’\-\.]+)*)", first)
        if nm:
            prog = nm.group(1)
            name = nm.group(2).strip()
        else:
            tokens = first.split()
            prog = tokens[0] if tokens and tokens[0].isdigit() else "?"
            name = tokens[1] if len(tokens) > 1 else f"{prog}-Horse"

        horses.append({
            "Prog": prog,
            "Horse": name,
            "Style": extract_running_style(block),
            "Speed": extract_speed(block),
            "PrimePower": extract_prime_power(block),
        })
    return horses

# =========================
# Race splitting
# =========================

def split_pdf_into_races_robust(full_text: str):
    """
    Improved splitter:
    - Only match 'Race N' at start of line
    - Allow trailing descriptor text
    - Filter using tokens like Post Time / Purse
    - No merging by number (prevents 'Race 12' → 'Race 1' issue)
    """
    text = full_text or ""

    header_re = r"(?im)^[ \t]*(?:Race|RACE)[ \t]+(\d{1,2})(?!\d)[ \t]*(?:[-–—].*)?$"
    headers = list(re.finditer(header_re, text))

    if not headers:
        return [("Race", text)]

    spans = []
    for i, m in enumerate(headers):
        num = int(m.group(1))
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        spans.append((num, start, end))

    def has_header_tokens(chunk: str) -> bool:
        head = chunk[:1200]
        tokens = ("Post Time", "Purse", "Furlongs", "Miles", "Surface", "Track", "Claiming", "Allowance")
        return any(t in head for t in tokens)

    races = []
    for num, start, end in spans:
        chunk = text[start:end]
        if has_header_tokens(chunk):
            races.append((f"Race {num}", chunk))

    if not races:
        races = [(f"Race {num}", text[start:end]) for (num, start, end) in spans]

    return races

# =========================
# Analysis functions
# =========================

def analyze_single_race_text(text, weights):
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)

    if df.empty:
        return df, {}

    # Ensure program numbers
    df["Prog"] = pd.to_numeric(df["Prog"], errors="coerce").fillna(0).astype(int)
    if df["Prog"].min() <= 0:
        df["Prog"] = range(1, len(df) + 1)

    # Fill missing values
    df = df.fillna({"Style": "NA", "Speed": 0, "PrimePower": DEFAULT_PP})

    # Rating
    df["Rating"] = (
        weights.get("prime_power", 1.0) * df["PrimePower"] +
        weights.get("speed", 1.0) * df["Speed"]
    )

    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)
    return df, {"count": len(df)}

def analyze_pdf_all(file_bytes, weights):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    races = split_pdf_into_races_robust(full_text)
    results = []
    seen = set()

    for header, chunk in races:
        sig = (header.strip(), hash(chunk))
        if sig in seen:
            continue
        seen.add(sig)
        df, meta = analyze_single_race_text(chunk, weights)
        if not df.empty:
            results.append((header.strip(), df, meta, chunk))

    return results

# =========================
# Streamlit App
# =========================

st.title("🏇 Horse Racing Model")

uploaded = st.file_uploader("Upload Brisnet PDF", type=["pdf"])

with st.sidebar:
    st.header("Weights")
    w = {
        "prime_power": st.slider("Prime Power weight", 0.0, 3.0, 1.0, 0.1),
        "speed": st.slider("Speed weight", 0.0, 3.0, 1.0, 0.1),
    }

if uploaded:
    results = analyze_pdf_all(uploaded.read(), w)

    if not results:
        st.error("No races found.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for (hdr, df, meta, chunk), tab in zip(results, tabs):
            with tab:
                st.subheader(f"{hdr} — Rankings")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇️ Download {hdr} rankings",
                    csv,
                    file_name=f"{hdr.replace(' ', '_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{hdr}"
                )