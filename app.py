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
st.caption("Splits races by finding each header’s 'Post Time' line, then the standalone race number above it. Program-line horse parsing. Keeps ALL races. Handles 1/1A, 2/2X, etc.")

# -----------------------------
# Globals / defaults
# -----------------------------
DEFAULT_PP = 100
DEFAULT_SPEED = 0
DEFAULT_STYLE = "NA"

# Program line starts a card: "<digits>[A-Z]?  <letters...>"
PROG_LINE_RE = re.compile(r"(?m)^\s*\d+[A-Z]?\s+[A-Za-z]")

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

def normalize_prog(s: str) -> str:
    """Normalize program numbers like '1', '1A', '2X'. Keep letter suffix uppercased; don't merge 1 vs 1A."""
    s = (s or "").strip().upper()
    m = re.match(r"^(\d+)([A-Z]?)$", s)
    if not m:
        return s
    num, suf = m.group(1), m.group(2)
    return num + suf  # exact identity preserved

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
# Horse parsing (program-line segmentation)
# -----------------------------
def extract_horses_from_text(text: str):
    """
    - A card starts at any line that begins with a program number (e.g., 1, 1A, 10) + letters (horse name).
    - Each card runs until the next program line.
    - Strip trailing odds/flags from the name (e.g., '5/2 L').
    """
    if not text:
        return []

    lines = text.splitlines()

    # indices of program lines
    start_idxs = []
    for i, ln in enumerate(lines):
        if re.match(r"^\s*\d+[A-Z]?\s+[A-Za-z]", ln or ""):
            start_idxs.append(i)

    if not start_idxs:
        return []

    # Build blocks
    blocks = []
    for k, si in enumerate(start_idxs):
        ei = start_idxs[k+1] if k+1 < len(start_idxs) else len(lines)
        block = "\n".join(lines[si:ei]).strip()
        if block:
            blocks.append(block)

    horses = []
    for block in blocks:
        first_line = next((ln for ln in block.splitlines() if ln.strip()), "")

        # Program number (allow A/X suffix)
        m_prog = re.match(r"^\s*(\d+[A-Z]?)\s+", first_line or "")
        prog_raw = m_prog.group(1).strip() if m_prog else ""
        prog = normalize_prog(prog_raw)

        # Name = everything after program number on the first line
        name = re.sub(r"^\s*\d+[A-Z]?\s+", "", first_line or "").strip()
        # Strip trailing odds/flags like "5/2 L", "12/1", etc.
        name = re.sub(r"\s+\d+/?\d+\s*[A-Za-z]*\s*$", "", name)
        name = re.sub(r"[,\s]+$","", name)
        if not name:
            name = f"{prog}-Horse" if prog else "Horse"

        style = extract_running_style(first_line, block)
        pp    = extract_prime_power(block)
        spd   = extract_speed(block)

        horses.append({
            "Prog": prog if prog else "",
            "Horse": name,
            "Style": style or DEFAULT_STYLE,
            "PrimePower": pp if isinstance(pp, int) else DEFAULT_PP,
            "Speed": spd if isinstance(spd, int) else DEFAULT_SPEED,
            "Block": block,
            "NameLen": len(name),
            "BlockLen": len(block),
        })

    if not horses:
        return []

    # Deduplicate ONLY within exact same program (keep best block/name)
    by_prog = {}
    for h in horses:
        key = h["Prog"]  # exact identity, so 1 vs 1A vs 2X are distinct
        if not key:
            # stash blank-prog items under a synthetic key to dedupe later
            key = f"__BLANK__{id(h)}"
        prev = by_prog.get(key)
        if prev is None:
            by_prog[key] = h
        else:
            # choose the richer entry: longer block first, then longer name
            cand = h
            score_prev = (prev["BlockLen"], prev["NameLen"])
            score_cand = (cand["BlockLen"], cand["NameLen"])
            if score_cand > score_prev:
                by_prog[key] = cand

    horses = list(by_prog.values())

    # Build DataFrame and tidy
    df = pd.DataFrame(horses)
    if df.empty:
        return []

    # Fill any blank programs sequentially 1..N; do not overwrite real ones like '1A' or '2X'
    if df["Prog"].eq("").any():
        seq, c = [], 1
        for v in df["Prog"].tolist():
            if v:
                seq.append(v)
            else:
                seq.append(str(c)); c += 1
        df["Prog"] = seq

    # Drop duplicate (Prog, Horse) pairs (extra safety)
    df = df.drop_duplicates(subset=["Prog", "Horse"])
    # Ensure string type
    df["Prog"] = df["Prog"].astype(str)

    # Return records without helper columns
    return df.drop(columns=["NameLen", "BlockLen"], errors="ignore").to_dict("records")

# -----------------------------
# HEADER-ONLY RACE SPLITTER
# -----------------------------
def split_pdf_into_races_header_only(full_text: str):
    """
    Strategy:
      1) Find every 'Post Time' line (appears once per race header).
      2) For each, scan UPWARD up to 12 lines to find a standalone race number line: r'^\s*\d{1,2}\s*$'.
      3) Use that number line index as the race START.
      4) Slice start -> next start.
      5) KEEP ALL CHUNKS (no filtering by horse-card count).
    """
    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    post_time_re = re.compile(r"(?i)\bPost\s*Time\b")
    solo_number_re = re.compile(r"^\s*\d{1,2}\s*$")  # big race number on its own line

    post_idxs = [i for i, ln in enumerate(lines) if post_time_re.search(ln or "")]
    if not post_idxs:
        # Fallback: try classic "Race N" header
        classic = []
        for i, ln in enumerate(lines):
            if re.match(r"(?i)^\s*Race\s+\d{1,2}(?:\s*[-–—].*)?$", ln or ""):
                classic.append(i)
        if not classic:
            return [("Race 1", text)]
        starts = sorted(classic)
    else:
        # For each Post Time line, search upward a small window for the standalone number
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

    # Slice into chunks and KEEP ALL of them
    races = []
    for k, start_idx in enumerate(deduped):
        end_idx = deduped[k+1] if k+1 < len(deduped) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        races.append((f"Race {k+1}", chunk))

    return races

# -----------------------------
# Analysis
# -----------------------------
def analyze_single_race_text(text: str, weights: dict):
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)
    if df.empty:
        return df, {}

    # Ensure non-empty Prog strings
    df["Prog"] = df["Prog"].astype(str)
    df.loc[df["Prog"].eq("") | df["Prog"].isna(), "Prog"] = [
        str(i + 1) for i in range((df["Prog"].eq("") | df["Prog"].isna()).sum())
    ]

    # Fill missing values
    df = df.fillna({"Style": DEFAULT_STYLE, "PrimePower": DEFAULT_PP, "Speed": DEFAULT_SPEED})

    # Simple, stable rating
    prime_w = float(weights.get("prime_power", 1.0))
    speed_w = float(weights.get("speed", 1.0))
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

    # Produce results; keep empty races as empty tables so you still see the tab
    results, seen = [], set()
    for header, chunk in races:
        sig = (header.strip(), hash(chunk))
        if sig in seen:
            continue
        seen.add(sig)
        df, meta = analyze_single_race_text(chunk, weights)
        if df.empty:
            df = pd.DataFrame(columns=["Prog","Horse","Style","StyleRating","PrimePower","Speed","Rating"])
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
                if df.empty:
                    st.warning("No horses parsed for this race (OCR/text extraction issue). The race was kept as requested.")
                else:
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