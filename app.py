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

# Horse card splitter: new block when a line starts with "1 ", "10 ", "1A ", etc., followed by a name and "("
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
    val = find_first(r"Prime\s*Power[:\s]*([\d\.]+)", block, group=1, default=None)
    if val is None:
        val = find_first(r"(?:PP|Prime\s*Powe?r)\D{0,6}(\d{2,3})", block, group=1, default=None)
    return safe_int(val, DEFAULT_PP)

def extract_running_style(first_line: str, block: str) -> str:
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", first_line, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    style = find_first(r"\((E\/P|E|P|S)\s*\d*\)", block, group=1, default=None)
    if style:
        return "EP" if style.upper() == "E/P" else style.upper()
    return DEFAULT_STYLE

def extract_speed(block: str) -> int:
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
    m = re.match(r"\s*(\d+[A-Z]?)\s+([A-Za-z'’\-\.\d]+(?:\s+[A-Za-z'’\-\.\d]+)*)\s*\(", first_line or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"\s*(\d+[A-Z]?)\s+(.+)$", first_line or "")
    if m:
        prog = m.group(1).strip()
        tokens = m.group(2).split()
        name = " ".join(tokens[:3]).strip() if tokens else f"{prog}-Horse"
        return prog, name
    tokens = (first_line or "").split()
    prog = tokens[0] if tokens and re.match(r"^\d+[A-Z]?$", tokens[0]) else ""
    name = tokens[1] if len(tokens) > 1 else (f"{prog}-Horse" if prog else "Horse")
    return (prog or ""), name

def extract_horses_from_text(text: str):
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
            re.search(r"^\s*\d+[A-Z]?\s+[A-Za-z'’\-\.\d]+", first_line or "")
            or ("Prime Power" in block)
            or re.search(r"\((E\/P|E|P|S)\s*\d*", first_line or "", flags=re.IGNORECASE)
        )
        if not looks_like_card:
            continue

        prog, name = parse_program_and_name(first_line)
        style = extract_running_style(first_line, block)
        pp = extract_prime_power(block)
        spd = extract_speed(block)

        if not prog:
            prog = ""  # normalized later

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

    # Fill blank programs sequentially (1..N) without overwriting real ones like '1A'
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

    df = df.drop_duplicates(subset=["Prog", "Horse"])
    return df.to_dict("records")

# =========================
# Multi-heuristic race splitter
# =========================
def split_pdf_into_races_robust(full_text: str):
    """
    Find race starts using three anchors:
      A) Lines starting with 'Ultimate PP' (repeated at each race in your file)
      B) Lines containing distance + 'Purse $' with 'Post Time' nearby (0..8 lines)
      C) Classic 'Race N' at start-of-line (fallback)
    Then dedupe near-duplicates and slice start->next start.
    Also require each resulting chunk to contain >= 4 horse cards; otherwise drop it.
    """
    text = full_text or ""
    lines = text.splitlines()
    n = len(lines)

    def has_distance(s: str) -> bool:
        return bool(re.search(r"\b(?:Furlongs?|Furlong|Mile|Miles|1\s*\d+/\d+\s*Miles?)\b", s, re.IGNORECASE))

    starts = set()

    # A) 'Ultimate PP' anchor
    for idx, line in enumerate(lines):
        if re.match(r"(?i)^\s*Ultimate\s*PP", line):
            starts.add(idx)

    # B) Distance + Purse + Post Time nearby
    i = 0
    while i < n:
        line = lines[i]
        if "Purse $" in line and has_distance(line):
            window = " ".join(lines[i:min(i+8, n)])
            if re.search(r"(?i)\bPost\s*Time\b", window):
                starts.add(i)
                i += 1
                continue
        i += 1

    # C) Classic "Race N"
    for m in re.finditer(r"(?im)^[ \t]*(?:Race|RACE)[ \t]+(\d{1,2})(?!\d)[ \t]*(?:[-–—].*)?$", text):
        pos = m.start()
        line_idx = text[:pos].count("\n")
        starts.add(line_idx)

    # If no starts, fallback to one big chunk
    if not starts:
        return [("Race 1", text)]

    # Sort & dedupe near-duplicates (<3 lines apart)
    starts_sorted = sorted(starts)
    deduped = []
    last = -999
    for s in starts_sorted:
        if s - last > 3:
            deduped.append(s)
            last = s

    # Slice into chunks
    raw_chunks = []
    for k, start_idx in enumerate(deduped):
        end_idx = deduped[k + 1] if k + 1 < len(deduped) else n
        chunk = "\n".join(lines[start_idx:end_idx]).strip()
        raw_chunks.append((k + 1, chunk))

    # Filter: keep only chunks that have at least 4 horse-card boundaries
    def horse_card_count(txt: str) -> int:
        return len(HORSE_SPLIT.findall("\n" + txt))  # leading \n to help regex see boundary

    filtered = []
    for k, chunk in raw_chunks:
        if horse_card_count(chunk) >= 4:
            filtered.append((k, chunk))

    # If filtering removed too many, fall back to raw_chunks
    if len(filtered) >= 2:
        final = filtered
    else:
        final = raw_chunks

    races = [(f"Race {k}", chunk) for k, chunk in final]
    return races

# =========================
# Analysis
# =========================
def analyze_single_race_text(text: str, weights: dict):
    horses = extract_horses_from_text(text)
    df = pd.DataFrame(horses)
    if df.empty:
        return df, {}

    df["Prog"] = df["Prog"].astype(str)
    df.loc[df["Prog"].eq("") | df["Prog"].isna(), "Prog"] = [
        str(i + 1) for i in range((df["Prog"].eq("") | df["Prog"].isna()).sum())
    ]
    df = df.fillna({"Style": DEFAULT_STYLE, "PrimePower": DEFAULT_PP, "Speed": DEFAULT_SPEED})

    prime_w = float(weights.get("prime_power", 1.0))
    speed_w = float(weights.get("speed", 1.0))
    df["Rating"] = prime_w * df["PrimePower"].astype(int) + speed_w * df["Speed"].astype(int)

    df = df.sort_values(["Rating", "PrimePower"], ascending=[False, False]).reset_index(drop=True)
    return df, {"count": len(df)}

def analyze_pdf_all(file_bytes: bytes, weights: dict):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

    # Diagnostics BEFORE splitting
    with st.expander("🔎 Diagnostics (race starts & first lines)", expanded=False):
        st.write("Characters extracted:", len(full_text))
        preview = "\n".join(full_text.splitlines()[:12])
        st.code(preview or "(no text extracted)", language="text")

    races = split_pdf_into_races_robust(full_text)

    # Diagnostics AFTER splitting
    with st.expander("🔎 Diagnostics (detected chunks)", expanded=False):
        st.write("Detected race chunks:", len(races))
        for hdr, chunk in races[:20]:
            first_lines = "\n".join(chunk.splitlines()[:3])
            st.code(f"{hdr}\n{first_lines}", language="text")

    # Deduplicate exact chunks
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
st.title("🏇 Horse Racing Model — Robust Race Splitter")
st.caption("Anchors: 'Ultimate PP' | distance + 'Purse $' + nearby 'Post Time' | classic 'Race N'. Drops tiny chunks. One tab per race.")

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
        st.error("No races parsed. Expand diagnostics to see what the splitter found; paste a header snippet and I’ll tune the anchors.")
    else:
        tabs = st.tabs([hdr for hdr, _, _, _ in results])
        for i, (hdr, df, meta, chunk) in enumerate(results):
            with tabs[i]:
                st.subheader(f"{hdr} — Rankings")
                df_show = df.copy()
                df_show.index = range(1, len(df_show) + 1)
                st.dataframe(df_show[["Prog", "Horse", "Style", "PrimePower", "Speed", "Rating"]],
                             use_container_width=True, height=420)
                st.download_button(
                    label=f"⬇️ Download {hdr} rankings (CSV)",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{hdr.replace(' ','_')}_rankings.csv",
                    mime="text/csv",
                    key=f"dl_{i}"
                )
else:
    st.info("Upload a PPs PDF to begin.")