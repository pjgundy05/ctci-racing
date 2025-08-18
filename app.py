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

DEFAULT_WEIGHTS = {"prime_power": 1.0, "speed": 1.0}

# New horse-card splitter:
# start a new block when we see a line beginning with a program number like "1 ", "10 ", "1A " followed by name and "("
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+[A-Z]?\s+[A-Za-z][^\n]+?\()")

# =========================
# Helpers
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
    # Brisnet often: "Prime Power: 125" (OCR can mangle punctuation)
    val = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1, default=None)
    if val is None:
        val = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1, default=None)
    return safe_int(val, DEFAULT_PP)

def extract_running_style(first_line: str, block: str) -> str:
    # Typical near the name: "(E 5)" "(E/P 7)" "(P 3)" "(S 8)"
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", first_line, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", block, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    return DEFAULT_STYLE

def extract_speed(block: str) -> int:
    # Try several likely labels; default if none present
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
    Extract program number (e.g., '1', '10', '1A') and horse name from first line.
    Fallbacks ensure no 'Unknown' names and no blank program numbers left unhandled.
    """
    # Common: "1 Horse Name (E 5) ..."
    m = re.match(r"\s*(\d+[A-Z]?)\s+([A-Za-z'’\-\.\d]+(?:\s+[A-Za-z'’\-\.\d]+)*)\s*\(", first_line or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Looser: "1 Horse Name ..." (without '(' on same line)
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
    Split the race text into horse 'cards' and extract structured fields.
    """
    if not text:
        return []

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
            prog = ""  # will be normalized below

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

    # Fill blank programs sequentially 1..N; do not overwrite real ones like '1A'
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

    # Deduplicate OCR fragments
    df = df.drop_duplicates(subset=["Prog", "Horse"])
    return df.to_dict("records")

# =========================
# Splitter: ONLY scan within header blocks (DATE → Post Time)
# =========================
def split_pdf_into_races_robust(full_text: str):
    """
    Split races by scanning ONLY within header blocks.

    Header block:
      - Starts on a DATE line (e.g., "Saturday, August 16, 2025", "August 16, 2025", "08/16/25").
      - Ends at the first line containing "Post Time" (inclusive); if not found soon, stop after a small window.

    Inside each header block, we pick the first "Race <num>" line as the race start.
    If none is found in a header, that header is skipped.

    After collecting all starts, slice start → next start and keep chunks with >= 4 horse-card boundaries.
    """

    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    # Flexible date detectors
    dow_month_date_re = re.compile(
        r"(?i)\b(?:Sun|Mon|Tues?|Wed(?:nes)?|Thu(?:rs)?|Fri|Sat(?:ur)?)day\b.*\b"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\b.*\b\d{1,2},\s*\d{4}"
    )
    month_date_re = re.compile(
        r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}"
    )
    numeric_date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

    def is_date_line(s: str) -> bool:
        return bool(
            s and (dow_month_date_re.search(s) or month_date_re.search(s) or numeric_date_re.search(s))
        )

    race_header_re = re.compile(r"(?i)^\s*Race\s+(\d{1,2})(?!\d)\s*(?:[-–—].*)?$")
    post_time_re = re.compile(r"(?i)\bPost\s*Time\b")

    # 1) Build header blocks: DATE → first Post Time (or small window)
    header_blocks = []
    i = 0
    MAX_HEADER_SPAN = 24
    while i < n:
        if is_date_line(lines[i]):
            end_limit = min(i + MAX_HEADER_SPAN, n)
            post_idx = None
            for j in range(i, end_limit):
                if post_time_re.search(lines[j] or ""):
                    post_idx = j
                    break
            if post_idx is None:
                post_idx = end_limit - 1
            header_blocks.append((i, post_idx))
            i = post_idx + 1
        else:
            i += 1

    # 2) Inside each block, pick first "Race N"
    starts = []
    for h_start, h_end in header_blocks:
        picked = None
        for j in range(h_start, h_end + 1):
            if race_header_re.match(lines[j] or ""):
                picked = j
                break
        if picked is not None:
            starts.append(picked)

    # Fallback: if no "Race N" found inside headers, try bare "Race N" (rare)
    if not starts:
        for idx, ln in enumerate(lines):
            if race_header_re.match(ln or ""):
                starts.append(idx)

    if not starts:
        return [("Race 1", text)]

    # 3) Sort & dedupe near-duplicates (<3 lines apart)
    starts.sort()
    deduped = []
    last = -999
    for s in starts:
        if s - last > 3:
            deduped.append(s)
            last = s

    # 4) Slice into race chunks
    raw_chunks = []
    for k, start_idx in enumerate(deduped):
        end_idx = deduped[k + 1] if k + 1 < len(deduped) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        raw_chunks.append((k + 1, chunk))

    # 5) Keep only chunks that look like real races (>= 4 horse cards)
    def horse_card_count(txt: str) -> int:
        return len(HORSE_SPLIT.findall("\n" + (txt or "")))

    filtered = [(k, c) for (k, c) in raw_chunks if horse_card_count(c) >= 4]
    final_chunks = filtered if len(filtered) >= 2 else raw_chunks

    return [(f"Race {k}", c) for (k, c) in final_chunks]

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
    df = df.fillna({"Style": DEFAULT_STYLE, "PrimePower": DEFAULT_PP, "Speed": DEFAULT_SPEED})

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

    # Diagnostics – header detection
    with st.expander("🔎 Race-header diagnostics", expanded=False):
        st.write("Characters extracted:", len(full_text))
        # print first few lines for sanity
        st.code("\n".join(full_text.splitlines()[:15]) or "(no text extracted)", language="text")

    races = split_pdf_into_races_robust(full_text)

    # Diagnostics – detected chunks
    with st.expander("🔎 Detected race chunks", expanded=False):
        st.write("Count:", len(races))
        for hdr, chunk in races[:20]:
            first_lines = "\n".join(chunk.splitlines()[:3])
            st.code(f"{hdr}\n{first_lines}", language="text")

    # Deduplicate exact chunks (rare OCR quirk)
    results, seen = [], set()
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
st.title("🏇 Horse Racing Model — Header-Scoped Splitter")
st.caption("Splits races by scanning only within header blocks (DATE → Post Time). Picks first 'Race N' in the block. One tab per race.")

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
        st.error("No races parsed. Open the diagnostics above; if a date or 'Post Time' format differs, paste one header snippet and we’ll widen the patterns.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for i, (hdr, df, meta, chunk) in enumerate(results):
            with tabs[i]:
                st.subheader(f"{hdr} — Rankings")
                df_show = df.copy()
                df_show.index = range(1, len(df_show) + 1)
                st.dataframe(
                    df_show[["Prog", "Horse", "Style", "PrimePower", "Speed", "Rating"]],
                    use_container_width=True, height=420
                )
                st.download_button(
                    label=f"⬇️ Download {hdr} rankings (CSV)",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{hdr.replace(' ','_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{i}"
                )
else:
    st.info("Upload a PPs PDF to begin.")