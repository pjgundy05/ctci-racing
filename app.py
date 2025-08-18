import io
import re
import pdfplumber
import pandas as pd
import streamlit as st

# =========================
# Configuration / Defaults
# =========================
DEFAULT_PP = 100  # default Prime Power if not found
DEFAULT_SPEED = 0
DEFAULT_STYLE = "NA"

# Main weights (simple model for stability)
DEFAULT_WEIGHTS = {
    "prime_power": 1.0,
    "speed": 1.0,
}

# Split horse cards more reliably:
#   start a new block when we see a line that starts with a program number like "1 ", "10 ", "1A "
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+[A-Z]?\s+[A-Za-z][^\n]+\()")

# =========================
# Safe helpers
# =========================
def safe_int(x, default=0):
    try:
        # allow "87", "87.0", etc.
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
    """
    Brisnet PPs often show "Prime Power: 125".
    OCR can mangle spaces/colon, so we try a couple of patterns.
    """
    val = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1, default=None)
    if val is None:
        val = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1, default=None)
    return safe_int(val, DEFAULT_PP)

def extract_running_style(first_line: str, block: str) -> str:
    """
    Running style is often printed like "(E 5)" "(E/P 7)" "(P 3)" "(S 8)" next to the horse name.
    Fall back to NA if we don't see it.
    """
    # look in the first line first
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", first_line, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    # try anywhere in the block as a fallback
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", block, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    return DEFAULT_STYLE

def extract_speed(block: str) -> int:
    """
    There isn't a single canonical 'Speed:' label in many PDFs, so try several heuristics.
    If nothing matches, return DEFAULT_SPEED.
    """
    # Try a few common labels/phrases OCR may capture
    candidates = [
        r"Best\s+Speed\s+at\s+Dist[:\s]+(\d+)",
        r"Best\s+(?:Turf|Dirt)\s+Speed[:\s]+(\d+)",
        r"Highest\s+last\s+race\s+speed\s+rating[:\s]+(\d+)",
        r"Speed\s+Rating[:\s]+(\d+)",
        r"Last\s+Speed[:\s]+(\d+)",
        r"\bSR[:\s]+(\d+)",  # sometimes abbreviated
    ]
    for pat in candidates:
        val = find_first(pat, block, group=1, default=None)
        if val is not None:
            return safe_int(val, DEFAULT_SPEED)
    # Nothing matched -> default
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
        prog = m.group(1).strip()
        name = m.group(2).strip()
        return prog, name

    # Looser fallback: number then words (without requiring '(')
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

    # Use the custom splitter; if it doesn't find anything, fall back to blank-line split
    blocks = HORSE_SPLIT.split(text)
    if len(blocks) <= 1:
        blocks = text.split("\n\n")

    horses = []
    for raw in blocks:
        block = (raw or "").strip()
        if not block:
            continue

        # First non-empty line gives us program & name & likely style
        first_line = next((ln for ln in block.splitlines() if ln.strip()), "")

        # Heuristic: only keep blocks that look like horse cards
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

        # Ensure we have something sensible for program number
        if not prog:
            # sequential placeholder (we’ll normalize later)
            prog = ""

        horses.append({
            "Prog": prog,               # may be "1", "1A", "", etc.
            "Horse": name or "Horse",
            "Style": style or DEFAULT_STYLE,
            "PrimePower": pp if isinstance(pp, int) else DEFAULT_PP,
            "Speed": spd if isinstance(spd, int) else DEFAULT_SPEED,
            "Block": block,             # keep raw block (useful for debugging)
            "FirstLine": first_line,
        })

    # Normalize
    if not horses:
        return []

    df = pd.DataFrame(horses)

    # Fill missing program numbers sequentially (1..N), but do NOT overwrite real ones (like 1A)
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

    # Deduplicate by Prog+Horse to avoid repeated OCR fragments
    df = df.drop_duplicates(subset=["Prog", "Horse"])

    return df.to_dict("records")

# =========================
# Race splitting
# =========================
def split_pdf_into_races_robust(full_text: str):
    """
    Split the PDF text into race chunks.

    - Match headers like:
        "Race 1"
        "RACE 10 — Allowance (Turf)"
        "Race 7 - Starter Optional Claiming"
      at the **start of a line** (case-insensitive), allowing trailing text.

    - Build spans from each header to the next header.
    - Keep spans whose first ~1200 chars contain typical header tokens.
    - Do NOT merge by race number (prevents 'all became Race 1' if OCR drops digits).
    """
    text = full_text or ""

    # 1) Headers at start-of-line, trailing descriptor allowed
    header_re = r"(?im)^[ \t]*(?:Race|RACE)[ \t]+(\d{1,2})(?!\d)[ \t]*(?:[-–—].*)?$"
    headers = list(re.finditer(header_re, text))

    if not headers:
        return [("Race", text)]

    # 2) Build [start,end) spans to next header
    spans = []
    for i, m in enumerate(headers):
        num = int(m.group(1))
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        spans.append((num, start, end))

    # 3) Verify header with typical tokens
    def has_header_tokens(chunk: str) -> bool:
        head = chunk[:1200]
        tokens = ("Post Time", "Purse", "Furlongs", "Miles", "About", "Surface", "Track", "Claiming", "Allowance")
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
# Analysis
# =========================
def analyze_single_race_text(text: str, weights: dict):
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)

    if df.empty:
        return df, {}

    # Ensure there is never a "0 horse": we DO NOT cast program to int (so 1A stays 1A)
    # Just keep as strings, but enforce non-empty
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

    # Simple rating (stable and fast)
    prime_w = float(weights.get("prime_power", 1.0))
    speed_w = float(weights.get("speed", 1.0))
    df["Rating"] = prime_w * df["PrimePower"].astype(int) + speed_w * df["Speed"].astype(int)

    # Sort by rating, tie-breaker by PrimePower
    df = df.sort_values(["Rating", "PrimePower"], ascending=[False, False]).reset_index(drop=True)
    return df, {"count": len(df)}

def analyze_pdf_all(file_bytes: bytes, weights: dict):
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
# Streamlit UI
# =========================
st.set_page_config(page_title="🏇 Horse Racing Model", page_icon="🏇", layout="wide")
st.title("🏇 Horse Racing Model (Stable Build)")
st.caption("Upload a Brisnet/Equibase PDF → robust race split → horse rankings. No ‘Unknown’ horses, no duplicate races, no ‘horse 0’.")

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
        st.error("No races parsed. Try another PDF or share a sample page for tuning.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for (hdr, df, meta, chunk), tab in zip(results, tabs):
            with tab:
                st.subheader(f"{hdr} — Rankings")
                df_show = df.copy()
                df_show.index = range(1, len(df_show) + 1)  # 1-based display
                st.dataframe(df_show[["Prog", "Horse", "Style", "PrimePower", "Speed", "Rating"]], use_container_width=True, height=420)

                st.download_button(
                    label=f"⬇️ Download {hdr} rankings (CSV)",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{hdr.replace(' ','_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{hdr}"
                )
else:
    st.info("Upload a PPs PDF to begin.")