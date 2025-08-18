import io
import re
import pdfplumber
import pandas as pd
import streamlit as st

# =========================
# Config / defaults
# =========================
DEFAULT_PP = 100
DEFAULT_SPEED = 0
DEFAULT_STYLE = "NA"

DEFAULT_WEIGHTS = {
    "prime_power": 1.0,
    "speed": 1.0,
}

# Split horse "cards" – a new block when a line starts with a program like "1 ", "10 ", "1A "
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+[A-Z]?\s+[A-Za-z][^\n]+?\()")

# =========================
# Safe helpers
# =========================
def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def find_first(pattern, text, flags=re.IGNORECASE, group=1, default=None):
    m = re.search(pattern, text or "", flags)
    return m.group(group) if m else default

# =========================
# Field extractors (robust)
# =========================
def extract_prime_power(block: str) -> int:
    # Brisnet: "Prime Power: 125". OCR may mangle punctuation.
    val = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1, default=None)
    if val is None:
        val = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1, default=None)
    return safe_int(val, DEFAULT_PP)

def extract_running_style(first_line: str, block: str) -> str:
    # Typically "(E 5)", "(E/P 7)", "(P 3)", "(S 8)"
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", first_line, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", block, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    return DEFAULT_STYLE

def extract_speed(block: str) -> int:
    # Try multiple reasonable labels; default if none present.
    candidates = [
        r"Best\s+Speed\s+at\s+Dist[:\s]+(\d+)",
        r"Best\s+(?:Turf|Dirt)\s+Speed[:\s]+(\d+)",
        r"Highest\s+last\s+race\s+speed\s+rating[:\s]+(\d+)",
        r"Speed\s+Rating[:\s]+(\d+)",
        r"Last\s+Speed[:\s]+(\d+)",
        r"\bSR[:\s]+(\d+)",
    ]
    for pat in candidates:
        val = find_first(pat, block, group=1, default=None)
        if val is not None:
            return safe_int(val, DEFAULT_SPEED)
    return DEFAULT_SPEED

# =========================
# Horse parsing
# =========================
def parse_program_and_name(first_line: str):
    """
    Extract program number (e.g., '1', '10', '1A') and name from the first line.
    Fallbacks ensure no 'Unknown' names.
    """
    # Typical: "1 Horse Name (E 5) ..."
    m = re.match(r"\s*(\d+[A-Z]?)\s+([A-Za-z'’\-\.\d]+(?:\s+[A-Za-z'’\-\.\d]+)*)\s*\(", first_line or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Looser fallback: number then some words
    m = re.match(r"\s*(\d+[A-Z]?)\s+(.+)$", first_line or "")
    if m:
        prog = m.group(1).strip()
        tokens = m.group(2).split()
        name = " ".join(tokens[:3]).strip() if tokens else f"{prog}-Horse"
        return prog, name

    # Ultimate fallback
    tokens = (first_line or "").split()
    prog = tokens[0] if tokens and re.match(r"^\d+[A-Z]?$", tokens[0]) else ""
    name = tokens[1] if len(tokens) > 1 else (f"{prog}-Horse" if prog else "Horse")
    return (prog or ""), name

def extract_horses_from_text(text: str):
    """
    Split the race text into horse 'cards' and extract fields.
    """
    if not text:
        return []

    # Preferred custom splitter; fallback to blank-line split if nothing found
    blocks = HORSE_SPLIT.split(text)
    if len(blocks) <= 1:
        blocks = text.split("\n\n")

    horses = []
    for raw in blocks:
        block = (raw or "").strip()
        if not block:
            continue

        first_line = next((ln for ln in block.splitlines() if ln.strip()), "")

        looks_like_card = (
            re.search(r"^\s*\d+[A-Z]?\s+[A-Za-z'’\-\.\d]+", first_line or "") or
            ("Prime Power" in block) or
            re.search(r"\((E\/P|E|P|S)\s*\d*", first_line or "", flags=re.IGNORECASE)
        )
        if not looks_like_card:
            continue

        prog, name = parse_program_and_name(first_line)
        style = extract_running_style(first_line, block)
        pp = extract_prime_power(block)
        spd = extract_speed(block)

        if not prog:
            prog = ""  # will be normalized later

        horses.append({
            "Prog": prog,
            "Horse": name or "Horse",
            "Style": style or DEFAULT_STYLE,
            "PrimePower": pp if isinstance(pp, int) else DEFAULT_PP,
            "Speed": spd if isinstance(spd, int) else DEFAULT_SPEED,
            "Block": block,
            "FirstLine": first_line,
        })

    if not horses:
        return []

    df = pd.DataFrame(horses)

    # Fill blank program numbers sequentially 1..N (do not alter real ones like '1A')
    if df["Prog"].eq("").any():
        seq = []
        c = 1
        for v in df["Prog"].tolist():
            if v:
                seq.append(v)
            else:
                seq.append(str(c))
                c += 1
        df["Prog"] = seq

    # Drop duplicate OCR fragments
    df = df.drop_duplicates(subset=["Prog", "Horse"])

    return df.to_dict("records")

# =========================
# Race splitting (Brisnet-style header)
# =========================
def split_pdf_into_races_robust(full_text: str):
    """
    Split races using Brisnet header structure (like your pasted Race 1):

      "1 Mile. Alw 50000s Purse $75,000 ... Non-winners ..."
      "Post Time: (12:35)/11:35/10:35/ 9:35"

    Heuristic:
      - A header line must include 'Purse $' and a distance token (Furlong(s), Mile, Miles, 1 1/16, etc.)
      - 'Post Time' must appear on the same line OR within the next few lines.
    """

    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    def has_distance(s: str) -> bool:
        return bool(re.search(r"\b(?:Furlongs?|Furlong|Mile|Miles|1\s*\d+/\d+\s*Miles?)\b", s, re.IGNORECASE))

    header_idxs = []
    for i, line in enumerate(lines):
        if "Purse $" in line and has_distance(line):
            # Is 'Post Time' nearby (same line or next 5 lines)?
            nearby = " ".join(lines[i:min(i + 6, n)])
            if "Post Time" in nearby:
                header_idxs.append(i)

    # If none matched, fall back to entire text as one race
    if not header_idxs:
        return [("Race 1", text)]

    # Build chunks between header indices
    races = []
    for k, start_idx in enumerate(header_idxs):
        end_idx = header_idxs[k + 1] if k + 1 < len(header_idxs) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        races.append((f"Race {k + 1}", chunk))

    return races

# =========================
# Analysis
# =========================
def analyze_single_race_text(text: str, weights: dict):
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)

    if df.empty:
        return df, {}

    # Keep program numbers as strings (so '1A' stays '1A'); ensure non-empty
    df["Prog"] = df["Prog"].astype(str)
    df.loc[df["Prog"].eq("") | df["Prog"].isna(), "Prog"] = [
        str(i + 1) for i in range((df["Prog"].eq("") | df["Prog"].isna()).sum())
    ]

    # Fill missing values
    df = df.fillna({
        "Style": DEFAULT_STYLE,
        "PrimePower": DEFAULT_PP,
        "Speed": DEFAULT_SPEED,
    })

    # Simple, stable rating
    prime_w = float(weights.get("prime_power", 1.0))
    speed_w = float(weights.get("speed", 1.0))
    df["Rating"] = prime_w * df["PrimePower"].astype(int) + speed_w * df["Speed"].astype(int)

    # Sort by rating; tie-break by PrimePower
    df = df.sort_values(["Rating", "PrimePower"], ascending=[False, False]).reset_index(drop=True)
    return df, {"count": len(df)}

def analyze_pdf_all(file_bytes: bytes, weights: dict):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    races = split_pdf_into_races_robust(full_text)

    # Deduplicate exact same chunk (rare, but can happen)
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
# Streamlit UI
# =========================
st.set_page_config(page_title="🏇 Horse Racing Model", page_icon="🏇", layout="wide")
st.title("🏇 Horse Racing Model (Brisnet header splitter)")
st.caption("Splits by distance + 'Purse $' with nearby 'Post Time'. One tab per race. No ‘Unknown’ horses, no ‘horse 0’.")

with st.sidebar:
    st.header("Weights")
    weights = {
        "prime_power": st.slider("Prime Power weight", 0.0, 3.0, 1.0, 0.1, key="w_pp"),
        "speed": st.slider("Speed weight", 0.0, 3.0, 1.0, 0.1, key="w_spd"),
    }

uploaded = st.file_uploader("Upload PPs PDF", type=["pdf"], key="uploader_main")

if uploaded:
    with st.spinner("Parsing & scoring…"):
        results = analyze_pdf_all(uploaded.read(), weights)

    if not results:
        st.error("No races parsed. If your PDF format differs, share one header line and I’ll adjust the splitter.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for (hdr, df, meta, chunk), tab in zip(results, tabs):
            with tab:
                st.subheader(f"{hdr} — Rankings")
                df_show = df.copy()
                df_show.index = range(1, len(df_show) + 1)  # 1-based display
                st.dataframe(
                    df_show[["Prog", "Horse", "Style", "PrimePower", "Speed", "Rating"]],
                    use_container_width=True, height=420
                )

                st.download_button(
                    label=f"⬇️ Download {hdr} rankings (CSV)",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{hdr.replace(' ','_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{hdr}"
                )
else:
    st.info("Upload a PPs PDF to begin.")