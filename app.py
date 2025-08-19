import io
import re
import pdfplumber
import pandas as pd
import streamlit as st

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="🏇 CTCI Horse Racing Model", page_icon="🏇", layout="wide")
st.title("🏇 CTCI Horse Racing Model — Header-Only Splitter")
st.caption("Splits races by finding each header’s 'Post Time' line, then the standalone race number above it. One tab per race.")

# -----------------------------
# Globals / defaults
# -----------------------------
DEFAULT_PP = 100
DEFAULT_SPEED = 0
DEFAULT_STYLE = "NA"

# New horse-card splitter:
# a new card starts when a line begins with a program number (e.g., 1, 1A, 10) + a name and "("
HORSE_SPLIT = re.compile(r"\n(?=\s*\d+[A-Z]?\s+[A-Za-z][^\n]+?\()")

# -----------------------------
# Small helpers
# -----------------------------
def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def find_first(pattern, text, flags=re.IGNORECASE, group=1, default=None):
    m = re.search(pattern, text or "", flags)
    return m.group(group) if m else default

# -----------------------------
# Field extractors (robust & forgiving)
# -----------------------------
def extract_prime_power(block: str) -> int:
    v = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1, default=None)
    if v is None:
        v = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1, default=None)
    return safe_int(v, DEFAULT_PP)

def extract_running_style(first_line: str, block: str) -> str:
    sty = find_first(r"\((E\/P|E|P|S)\s*\d*\)", first_line, group=1, default=None)
    if sty:
        return "EP" if sty.upper() == "E/P" else sty.upper()
    sty = find_first(r"\((E\/P|E|P|S)\s*\d*\)", block, group=1, default=None)
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
        v = find_first(pat, block, group=1, default=None)
        if v is not None:
            return safe_int(v, DEFAULT_SPEED)
    return DEFAULT_SPEED

# -----------------------------
# Horse parsing
# -----------------------------
def parse_program_and_name(first_line: str):
    # Typical: "5 Horse Name (E 5) ..."
    m = re.match(r"\s*(\d+[A-Z]?)\s+([A-Za-z'’\-\.\d]+(?:\s+[A-Za-z'’\-\.\d]+)*)\s*\(", first_line or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Looser fallback (no '(' on first line)
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
    if not text:
        return []
    # Try the stronger 'program number + name + (' splitter first
    blocks = HORSE_SPLIT.split(text)
    if len(blocks) <= 1:
        # fallback: split on blank double newline
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
            prog = ""  # normalized below

        horses.append({
            "Prog": prog,
            "Horse": name or "Horse",
            "Style": style or DEFAULT_STYLE,
            "PrimePower": pp if isinstance(pp, int) else DEFAULT_PP,
            "Speed": spd if isinstance(spd, int) else DEFAULT_SPEED,
            "Block": block,
        })

    if not horses:
        return []

    df = pd.DataFrame(horses)

    # Fill blank programs sequentially 1..N; do not overwrite real ones like '1A'
    if df["Prog"].eq("").any():
        seq, c = [], 1
        for v in df["Prog"].tolist():
            if v:
                seq.append(v)
            else:
                seq.append(str(c)); c += 1
        df["Prog"] = seq

    # Drop duplicate OCR fragments
    df = df.drop_duplicates(subset=["Prog", "Horse"])
    return df.to_dict("records")

# -----------------------------
# HEADER-ONLY RACE SPLITTER
# -----------------------------
def split_pdf_into_races_header_only(full_text: str):
    """
    Strategy:
      1) Find every 'Post Time' line (appears once per race header).
      2) For each, scan UPWARD up to 12 lines to find a standalone race number line: r'^\s*\d{1,2}\s*$'.
      3) Use that number line index as the race START.
      4) Slice start -> next start. Drop micro-chunks (< 4 horse cards).

    This matches the screenshots you shared for Race 1 & Race 2.
    """
    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    post_time_re = re.compile(r"(?i)\bPost\s*Time\b")
    # A big, standalone race number (e.g., "1" on its own line)
    solo_number_re = re.compile(r"^\s*\d{1,2}\s*$")

    post_idxs = [i for i, ln in enumerate(lines) if post_time_re.search(ln or "")]
    if not post_idxs:
        # Fallback: try classic "Race N" at start of a line
        classic = []
        for i, ln in enumerate(lines):
            if re.match(r"(?i)^\s*Race\s+\d{1,2}(?:\s*[-–—].*)?$", ln or ""):
                classic.append(i)
        if not classic:
            return [("Race 1", text)]
        starts = sorted(classic)
    else:
        # For each Post Time line, search upward a small window for the standalone number line
        starts = []
        for pt in post_idxs:
            found = None
            for j in range(max(0, pt-12), pt+1):
                if solo_number_re.match(lines[j] or ""):
                    found = j
                    break
            if found is not None:
                starts.append(found)

        if not starts:
            return [("Race 1", text)]
        starts = sorted(set(starts))

    # Dedupe near-duplicates (< 3 lines apart)
    deduped = []
    last = -999
    for s in starts:
        if s - last > 3:
            deduped.append(s)
            last = s

    # Slice into chunks
    raw_chunks = []
    for k, start_idx in enumerate(deduped):
        end_idx = deduped[k+1] if k+1 < len(deduped) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        raw_chunks.append((k+1, chunk))

    # Keep only chunks that look like real races (>=4 horse cards)
    def horse_card_count(txt: str) -> int:
        return len(HORSE_SPLIT.findall("\n" + (txt or "")))

    filtered = [(k, c) for (k, c) in raw_chunks if horse_card_count(c) >= 4]
    final = filtered if len(filtered) >= 2 else raw_chunks

    return [(f"Race {k}", c) for (k, c) in final]

# -----------------------------
# Analysis
# -----------------------------
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
    # (Optional) small style weight
    sty_map = {"E": 7, "EP": 5, "P": 3, "S": 1}
    df["StyleRating"] = df["Style"].map(sty_map).fillna(0)
    df["Rating"] = prime_w * df["PrimePower"].astype(int) + speed_w * df["Speed"].astype(int) + 0.5 * df["StyleRating"]

    df = df.sort_values(["Rating", "PrimePower"], ascending=[False, False]).reset_index(drop=True)
    return df, {}

def analyze_pdf_all(file_bytes: bytes, weights: dict):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    # Diagnostics – show first lines for sanity
    with st.expander("🔎 Header diagnostics", expanded=False):
        st.write("Characters extracted:", len(full_text))
        st.code("\n".join(full_text.splitlines()[:20]) or "(no text extracted)", language="text")

    races = split_pdf_into_races_header_only(full_text)

    with st.expander("🔎 Detected race chunks", expanded=False):
        st.write("Count:", len(races))
        for hdr, chunk in races[:20]:
            sample = "\n".join(chunk.splitlines()[:5])
            st.code(f"{hdr}\n{sample}", language="text")

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

# -----------------------------
# Sidebar weights
# -----------------------------
with st.sidebar:
    st.header("Weights")
    weights = {
        "prime_power": st.slider("Prime Power weight", 0.0, 3.0, 1.0, 0.1, key="w_pp"),
        "speed": st.slider("Speed weight", 0.0, 3.0, 1.0, 0.1, key="w_spd"),
    }

# -----------------------------
# Main UI
# -----------------------------
uploaded = st.file_uploader("Upload Brisnet PDF (text-based, not scanned)", type=["pdf"], key="uploader_main")

if uploaded:
    with st.spinner("Parsing & scoring…"):
        results = analyze_pdf_all(uploaded.read(), weights)

    if not results:
        st.error("No races parsed. Open the diagnostics above to see what header lines were detected.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for i, (hdr, df, meta, chunk) in enumerate(results):
            with tabs[i]:
                st.subheader(f"{hdr} — Rankings")
                df_show = df.copy()
                df_show.index = range(1, len(df_show) + 1)
                st.dataframe(
                    df_show[["Prog", "Horse", "Style", "StyleRating", "PrimePower", "Speed", "Rating"]],
                    use_container_width=True,
                    height=440,
                )
                st.download_button(
                    label=f"⬇️ Download {hdr} rankings (CSV)",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{hdr.replace(' ','_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{i}"
                )
else:
    st.info("Upload a Brisnet PDF to begin.")