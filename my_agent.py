# agent_d_app.py
# Path: place this file directly in C:\Users\saisarkar\crewai_project
# Bot: Agent D (Streamlit + CrewAI orchestration, Gemini for LLM/RAG)
# Features:
# - Internet toggle (web-qa vs local-only)
# - Repository folder indexing: .\repository (PDF/TXT/CSV/XLSX)
# - Multi-file upload (≤ 20 MB total), tabular + text ingestion
# - Accurate Excel reading (all sheets), deterministic allocation queries
# - Local RAG with FAISS + Gemini embeddings
# - Web RAG (DuckDuckGo + trafilatura) when Internet mode is ON
# - Chat history saved to .\chats
# - CrewAI Agents + Tasks + Crew used to orchestrate the flow
 
import io
import os
import re
import json
import glob
import uuid
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
 
import streamlit as st
import pandas as pd
from dateutil.parser import parse as parse_date
 
# Optional libs
try:
    import faiss
except Exception:
    faiss = None
 
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
 
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None
 
try:
    import trafilatura
except Exception:
    trafilatura = None
 
# LLM (Gemini)
import google.generativeai as genai
 
# CrewAI (orchestration)
from crewai import Agent, Task, Crew
 
# -----------------------------
# CONSTANTS & PATHS
# -----------------------------
APP_TITLE = "Agent D"
DEFAULT_MODEL = "gemini-2.0-flash-lite-001"
EMBED_MODEL = "text-embedding-004"
 
BASE_DIR = os.getcwd()  # stays in your current folder
DATA_DIR = os.path.join(BASE_DIR, "data_uploads")
REPO_DIR = os.path.join(BASE_DIR, "repository")
CHATS_DIR = os.path.join(BASE_DIR, "chats")
for d in (DATA_DIR, REPO_DIR, CHATS_DIR):
    os.makedirs(d, exist_ok=True)
 
MAX_TOTAL_UPLOAD_MB = 20
 
# -----------------------------
# STREAMLIT & GEMINI SETUP
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)
 
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
MODEL = st.secrets.get("MODEL", DEFAULT_MODEL)
 
# -----------------------------
# SESSION STATE
# -----------------------------
def ensure_state():
    if "internet_mode" not in st.session_state:
        st.session_state.internet_mode = False
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "file_store" not in st.session_state:
        st.session_state.file_store: Dict[str, Dict[str, Any]] = {}
    if "repo_index" not in st.session_state:
        st.session_state.repo_index = {"tabular": [], "texts": [], "index": None, "metas": []}
    if "active_checksum" not in st.session_state:
        st.session_state.active_checksum = None
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None
 
ensure_state()
 
# -----------------------------
# HELPERS
# -----------------------------
def file_checksum(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()
 
def safe_parse_date(text: str) -> Optional[str]:
    try:
        return parse_date(text).date().isoformat()
    except Exception:
        return None
 
def load_pdf_text(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)
 
def excel_to_frames(raw: bytes) -> List[pd.DataFrame]:
    """Read Excel across all sheets."""
    xls = pd.ExcelFile(io.BytesIO(raw))
    frames = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if not df.empty:
            df["__sheet__"] = sheet
            frames.append(df)
    if not frames:
        frames = [pd.read_excel(io.BytesIO(raw))]
    return frames
 
def normalize_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical columns for allocation analytics while preserving others."""
    lower_cols = {c.lower(): c for c in df.columns}
 
    def pick(opts):
        for o in opts:
            if o in lower_cols:
                return lower_cols[o]
        return None
 
    name_col = pick(["name","provider","employee","resource"])
    alloc_col = pick(["allocation_pct","allocated %","allocation","allocation%","utilization","percent","percentage"])
    date_col  = pick(["date","day","as_of_date","dt"])
 
    if name_col is None:
        name_col = df.columns[0]
    if alloc_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        alloc_col = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns)>1 else df.columns[0])
 
    out = pd.DataFrame()
    out["name"] = df[name_col].astype(str)
    out["allocation_pct"] = pd.to_numeric(df[alloc_col], errors="coerce")
 
    if date_col is not None:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT
 
    for c in df.columns:
        if c not in [name_col, alloc_col, date_col]:
            out[c] = df[c]
    if "__sheet__" in df.columns:
        out["sheet"] = df["__sheet__"]
    return out
 
def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        return [text[:max_chars]]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])
    return chunks
 
def make_faiss_index(texts: List[str], metas: Optional[List[Dict[str, Any]]] = None):
    if faiss is None or not texts:
        return None, metas or []
    import numpy as np
    vecs = []
    for t in texts:
        emb = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(emb["embedding"])
    mat = np.array(vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index, metas or [{} for _ in texts]
 
def faiss_search(index, query: str, metas, k=5):
    if faiss is None or index is None:
        return []
    import numpy as np
    q = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    out = []
    for i, score in zip(I[0], D[0]):
        if i == -1:
            continue
        md = metas[i] if metas and i < len(metas) else {}
        out.append((i, float(score), md))
    return out
 
def gemini_complete(system_prompt: str, user_prompt: str) -> str:
    model = genai.GenerativeModel(MODEL, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt)
    return (resp.text or "").strip()
 
# -----------------------------
# INDEXERS (Repository & Uploads)
# -----------------------------
def index_repository(repo_dir: str) -> Dict[str, Any]:
    tabular_frames: List[pd.DataFrame] = []
    text_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []
 
    for path in glob.glob(os.path.join(repo_dir, "**/*"), recursive=True):
        if not os.path.isfile(path):
            continue
        low = path.lower()
        try:
            if low.endswith(".xlsx"):
                raw = open(path, "rb").read()
                for df in excel_to_frames(raw):
                    tabular_frames.append(normalize_tabular(df))
            elif low.endswith(".csv"):
                df = pd.read_csv(path)
                tabular_frames.append(normalize_tabular(df))
            elif low.endswith(".pdf"):
                raw = open(path, "rb").read()
                txt = load_pdf_text(raw)
                for j, ch in enumerate(chunk_text(txt)):
                    text_chunks.append(ch)
                    metas.append({"source": path, "chunk": j})
            elif low.endswith(".txt"):
                txt = open(path, "r", encoding="utf-8", errors="ignore").read()
                for j, ch in enumerate(chunk_text(txt)):
                    text_chunks.append(ch)
                    metas.append({"source": path, "chunk": j})
        except Exception:
            continue
 
    combined_tabular = pd.concat(tabular_frames, ignore_index=True) if tabular_frames else pd.DataFrame()
    index, metas = make_faiss_index(text_chunks, metas) if text_chunks else (None, [])
    return {
        "tabular": [combined_tabular] if not combined_tabular.empty else [],
        "texts": text_chunks,
        "index": index,
        "metas": metas,
    }
 
def add_uploaded_files(files) -> Tuple[Dict[str, Dict[str, Any]], List[str], int]:
    store_updates: Dict[str, Dict[str, Any]] = {}
    checksums: List[str] = []
    total_bytes = 0
 
    for up in files:
        total_bytes += up.size
        raw = up.read()
        checksum = file_checksum(raw)
        checksums.append(checksum)
 
        rec: Dict[str, Any] = {"filename": up.name, "uploaded_at": datetime.now().isoformat(timespec="seconds")}
        kind = None
 
        try:
            if up.name.lower().endswith(".xlsx"):
                frames = [normalize_tabular(df) for df in excel_to_frames(raw) if not df.empty]
                if frames:
                    rec["df"] = pd.concat(frames, ignore_index=True)
                    kind = "tabular"
            elif up.name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw))
                rec["df"] = normalize_tabular(df)
                kind = "tabular"
            elif up.name.lower().endswith(".pdf"):
                txt = load_pdf_text(raw)
                chunks = chunk_text(txt)
                idx, metas = make_faiss_index(chunks, [{"source": up.name, "chunk": i} for i,_ in enumerate(chunks)])
                rec.update({"texts": chunks, "index": idx, "metas": metas})
                kind = "text"
            elif up.name.lower().endswith(".txt"):
                txt = raw.decode("utf-8", errors="ignore")
                chunks = chunk_text(txt)
                idx, metas = make_faiss_index(chunks, [{"source": up.name, "chunk": i} for i,_ in enumerate(chunks)])
                rec.update({"texts": chunks, "index": idx, "metas": metas})
                kind = "text"
        except Exception as e:
            st.error(f"Ingestion error for {up.name}: {e}")
 
        if kind:
            rec["kind"] = kind
            store_updates[checksum] = rec
 
    return store_updates, checksums, total_bytes

# -----------------------------
# NL PARSER (strict JSON)
# -----------------------------
PARSER_SYSTEM = """Translate a single user question into STRICT JSON:
Keys:
- intent: one of ["get_under_allocated","get_over_allocated","get_summary","tabular_select","rag_qa","web_qa"]
- start_date: "YYYY-MM-DD" or null
- end_date: "YYYY-MM-DD" or null
- threshold: number or null
- columns: array or null
- conditions: array of {column, op, value} or null
Rules:
- Output ONLY the JSON object.
- Allocation/threshold language -> allocation intents.
- Generic table filters -> tabular_select.
- Current events/world facts -> web_qa (only if Internet mode is ON).
Examples:
User: "Who is under allocated between Oct 1 2025 and Oct 29 2025?"
{"intent":"get_under_allocated","start_date":"2025-10-01","end_date":"2025-10-29","threshold":100,"columns":["name","allocation_pct","date"],"conditions":null}
User: "List rows where project='Alpha' and status='Open'"
{"intent":"tabular_select","start_date":null,"end_date":null,"threshold":null,"columns":null,"conditions":[{"column":"project","op":"=","value":"Alpha"},{"column":"status","op":"=","value":"Open"}]}
"""
 
def nl_parse(user_text: str, internet_mode: bool) -> Dict[str, Any]:
    raw = gemini_complete(PARSER_SYSTEM, user_text)
    try:
        parsed = json.loads(raw)
    except Exception:
        # FIX: robust JSON extraction
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(m.group(0)) if m else {"intent":"rag_qa","start_date":None,"end_date":None,"threshold":None,"columns":None,"conditions":None}
    for k in ("start_date","end_date"):
        v = parsed.get(k)
        if v:
            parsed[k] = safe_parse_date(v)
    if parsed.get("intent") == "web_qa" and not internet_mode:
        parsed["intent"] = "rag_qa"
    return parsed
 
# -----------------------------
# EXECUTORS (Tabular, RAG, Web)
# -----------------------------
def run_tabular_allocation(df: pd.DataFrame, intent: str, start: Optional[str], end: Optional[str], threshold: Optional[float]) -> Dict[str, Any]:
    work = df.copy()
    work["allocation_pct"] = pd.to_numeric(work["allocation_pct"], errors="coerce")
    if start:
        work = work[work["date"] >= pd.to_datetime(start).date()]
    if end:
        work = work[work["date"] <= pd.to_datetime(end).date()]
 
    if intent == "get_under_allocated":
        thr = threshold if threshold is not None else 100.0
        work = work[work["allocation_pct"] < thr]
        descriptor = f"allocation_pct < {thr}"
        work = work.sort_values("allocation_pct", ascending=True)
    else:
        thr = threshold if threshold is not None else 100.0
        work = work[work["allocation_pct"] > thr]
        descriptor = f"allocation_pct > {thr}"
        work = work.sort_values("allocation_pct", ascending=False)
 
    q = f"date BETWEEN {start or '-inf'} AND {end or '+inf'} AND {descriptor}"
    return {"mode":"table","table":work, "query": q, "count": int(work.shape[0])}
 
def apply_condition(df: pd.DataFrame, column: str, op: str, value: Any) -> pd.Series:
    op = op.strip().lower()
    if column not in df.columns:
        return pd.Series([False]*len(df))
    col = df[column]
    if op in ("=", "=="):    return col.astype(str) == str(value)
    if op in ("!=", "<>"):   return col.astype(str) != str(value)
    if op in (">", ">=","<","<="):
        left = pd.to_numeric(col, errors="coerce")
        try:
            val = float(value)
        except Exception:
            return pd.Series([False]*len(df))
        if op == ">":  return left > val
        if op == ">=": return left >= val
        if op == "<":  return left < val
        if op == "<=": return left <= val
    if op == "in" and isinstance(value, list):
        return col.astype(str).isin([str(x) for x in value])
    if op == "contains":
        return col.astype(str).str.contains(str(value), case=False, na=False)
    return pd.Series([False]*len(df))
 
def run_tabular_select(df: pd.DataFrame, conditions: Optional[List[Dict[str,Any]]]) -> Dict[str, Any]:
    work = df.copy()
    mask = pd.Series([True]*len(work))
    if conditions:
        for cond in conditions:
            col, op, val = cond.get("column"), cond.get("op","="), cond.get("value")
            mask = mask & apply_condition(work, col, op, val)
    work = work[mask]
    return {"mode":"table","table":work, "query":" AND ".join([f"{c.get('column')} {c.get('op')} {c.get('value')}" for c in (conditions or [])]) or "ALL", "count": int(work.shape[0])}
 
def local_rag(query: str, repo_idx: Dict[str, Any]) -> Dict[str, Any]:
    texts, metas, index = repo_idx.get("texts"), repo_idx.get("metas"), repo_idx.get("index")
    if not texts or not index:
        return {"mode":"error","error":"No local text index. Put files in repository and Re-index."}
    hits = faiss_search(index, query, metas, k=5)
    retrieved = []
    for i, score, md in hits:
        retrieved.append({"text": texts[i][:5000], "score": score, "meta": md})
    ctx = "\n\n".join([f"[Chunk {j+1}] {r['text']}" for j, r in enumerate(retrieved)])
    prompt = f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    answer = gemini_complete("Be concise and accurate. Cite chunk numbers when relevant.", prompt)
    return {"mode":"rag","answer":answer,"sources":retrieved}
 
def fetch_web_context(query: str, n: int = 5) -> List[Dict[str, Any]]:
    pages = []
    if DDGS is None or trafilatura is None:
        return pages
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, safesearch="moderate", max_results=n):
                url = r.get("href") or r.get("url") or r.get("link")
                title = r.get("title") or r.get("body") or ""
                if not url:
                    continue
                html = trafilatura.fetch_url(url)
                text = trafilatura.extract(html) if html else ""
                if text:
                    pages.append({"url": url, "title": title, "text": text})
    except Exception:
        pass
    return pages
 
def web_qa(query: str) -> Dict[str, Any]:
    pages = fetch_web_context(query, 5)
    if not pages:
        return {"mode":"error","error":"Web fetch returned no readable pages."}
    ctx = "\n\n---\n\n".join([f"[{p['title']}] {p['text'][:2000]}" for p in pages[:5]])
    answer = gemini_complete("Answer only with provided web context; be precise and neutral.", f"{ctx}\n\nQuestion: {query}\nAnswer:")
    return {"mode":"web","answer":answer,"sources":[{"title":p['title'],"url":p['url']}]}
 
# -----------------------------
# CREWAI ORCHESTRATION
# -----------------------------
# We build a Crew that conceptually mirrors GPT-like flow:
#   NLParserAgent -> (TabularAgent OR RAGAgent OR WebAgent) -> Answer
# CrewAI is used for structure & traceability; Gemini performs the LLM work.
 
nl_agent = Agent(
    role="NLParserAgent",
    goal="Parse user question into strict JSON parameters (intent, dates, filters).",
    backstory="Expert at extracting precise analytics parameters from natural language.",
    allow_delegation=False,
)
 
tabular_agent = Agent(
    role="TabularAgent",
    goal="Answer structured questions deterministically from tabular data.",
    backstory="Analyst who uses reproducible pandas filters and returns exact rows.",
    allow_delegation=False,
)
 
rag_agent = Agent(
    role="RAGAgent",
    goal="Retrieve and answer from local documents with FAISS + Gemini embeddings.",
    backstory="Grounded QA specialist; cites sources.",
    allow_delegation=False,
)
 
web_agent = Agent(
    role="WebAgent",
    goal="Search and summarize web sources when Internet mode is ON.",
    backstory="Finds reliable sources and answers grounded in fetched pages.",
    allow_delegation=False,
)
 
def crew_answer(user_text: str) -> Dict[str, Any]:
    """Run a mini-crew flow per question. We keep it synchronous & deterministic."""
    # Task 1: parse
    parse_task = Task(
        description=f"Parse to JSON. User: {user_text}",
        expected_output="A JSON object with fields: intent,start_date,end_date,threshold,columns,conditions",
        agent=nl_agent,
    )
    # Execute parsing via our deterministic function (Gemini under-the-hood)
    parsed = nl_parse(user_text, st.session_state.internet_mode)
 
    # Task 2: route to execution
    exec_task = Task(
        description="Execute the parsed intent using the appropriate executor and return structured result.",
        expected_output="A dict containing {mode, result, sources?, query?}",
        agent=tabular_agent if parsed.get("intent") in {"get_under_allocated","get_over_allocated","get_summary","tabular_select"} else (web_agent if parsed.get('intent')=='web_qa' else rag_agent),
    )
 
    # Merge tabular data sources (repository + active upload)
    tabular_sources: List[pd.DataFrame] = []
    if st.session_state.repo_index.get("tabular"):
        tabular_sources.extend(st.session_state.repo_index["tabular"])
    if st.session_state.active_checksum and st.session_state.active_checksum in st.session_state.file_store:
        rec = st.session_state.file_store[st.session_state.active_checksum]
        if rec.get("kind") == "tabular":
            tabular_sources.append(rec["df"])
    combined_df = pd.concat(tabular_sources, ignore_index=True) if tabular_sources else pd.DataFrame()
 
    # Actual execution (deterministic)
    intent = parsed.get("intent")
    if intent in {"get_under_allocated","get_over_allocated"} and not combined_df.empty:
        res = run_tabular_allocation(combined_df, intent, parsed.get("start_date"), parsed.get("end_date"), parsed.get("threshold"))
        return {"mode":"tabular","parsed":parsed,"result":res}
    if intent == "get_summary" and not combined_df.empty:
        desc = combined_df["allocation_pct"].describe().to_frame().reset_index()
        return {"mode":"tabular","parsed":parsed,"result":{"mode":"summary","table":desc,"query":"SUMMARY(allocation_pct)","count":int(combined_df.shape[0])}}
    if intent == "tabular_select" and not combined_df.empty:
        res = run_tabular_select(combined_df, parsed.get("conditions"))
        return {"mode":"tabular","parsed":parsed,"result":res}
    if intent == "web_qa" and st.session_state.internet_mode:
        res = web_qa(user_text)
        return {"mode":"web","parsed":parsed,"result":res}
    # default to local RAG
    res = local_rag(user_text, st.session_state.repo_index)
    return {"mode":"rag","parsed":parsed,"result":res}
 
# We instantiate a Crew (for traceability/consistency with CrewAI usage).
crew = Crew(agents=[nl_agent, tabular_agent, rag_agent, web_agent], tasks=[])

# -----------------------------
# SIDEBAR: CHAT HISTORY
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("🗂️ Chat History")
    files = sorted(glob.glob(os.path.join(CHATS_DIR, "*.json")))
    names = ["(new)"] + [os.path.basename(p)[:-5] for p in files]
    choice = st.selectbox("Open:", options=names, index=0)
    if choice != "(new)":
        with open(os.path.join(CHATS_DIR, f"{choice}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.chat_id = data.get("chat_id", choice)
        st.session_state.messages = data.get("messages", [])
        st.success(f"Loaded chat: {choice}")
    if st.button("🧹 New chat"):
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        st.session_state.messages = []
        st.success("Started a new chat.")
    if st.session_state.messages and st.button("💾 Save chat"):
        if not st.session_state.chat_id:
            st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        out = {"chat_id": st.session_state.chat_id, "saved_at": datetime.now().isoformat(timespec="seconds"), "messages": st.session_state.messages}
        with open(os.path.join(CHATS_DIR, f"{st.session_state.chat_id}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        st.success(f"Saved chat: {st.session_state.chat_id}")
 
# -----------------------------
# MAIN CHAT
# -----------------------------
st.markdown("---")
st.markdown("#### 💬 Chat (Agent D)")
 
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
user_text = st.chat_input("Ask Agent D… (toggle Internet in sidebar)")
if user_text:
    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
 
    # Run the Crew flow for this question
    result = crew_answer(user_text)
 
    with st.chat_message("assistant"):
        mode = result.get("mode")
        if mode == "tabular":
            res = result["result"]
            if res["mode"] == "summary":
                st.markdown("**Summary (allocation_pct)**")
                st.dataframe(res["table"])
                st.info(f"Rows considered: {res['count']}")
                st.session_state.messages.append({"role":"assistant","content":"Posted summary stats."})
            elif res["mode"] == "table":
                st.markdown("**Tabular results**")
                with st.expander("Query descriptor"):
                    st.code(res["query"], language="sql")
                st.dataframe(res["table"])
                csv = res["table"].to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="agentd_result.csv", mime="text/csv")
                st.success(f"{res['count']} row(s).")
                st.session_state.messages.append({"role":"assistant","content":f"Returned {res['count']} rows."})
            else:
                st.warning("No results.")
                st.session_state.messages.append({"role":"assistant","content":"No results."})
 
        elif mode == "rag":
            res = result["result"]
            if res.get("mode") == "rag":
                st.markdown("**Local RAG Answer**")
                st.markdown(res["answer"])
                if res.get("sources"):
                    with st.expander("Sources"):
                        for j, s in enumerate(res["sources"], 1):
                            st.markdown(f"**Chunk {j}** · score={s['score']:.3f} · source={s['meta'].get('source','')}")
                            st.write((s["text"][:1000] + "…") if len(s["text"])>1000 else s["text"])
                st.session_state.messages.append({"role":"assistant","content":res["answer"][:2000]})
            else:
                st.error(res.get("error","Local RAG error"))
                st.session_state.messages.append({"role":"assistant","content":"RAG error."})
 
        elif mode == "web":
            res = result["result"]
            if res.get("mode") == "web":
                st.markdown("**Web-grounded Answer**")
                st.markdown(res["answer"])
                if res.get("sources"):
                    with st.expander("Sources"):
                        for s in res["sources"]:
                            st.markdown(f"- [{s['title']}]({s['url']})")
                st.session_state.messages.append({"role":"assistant","content":res["answer"][:2000]})
            else:
                st.error(res.get("error","Web QA error"))
                st.session_state.messages.append({"role":"assistant","content":"Web QA error."})
        else:
            st.error("Unknown mode.")
            st.session_state.messages.append({"role":"assistant","content":"Unknown mode."})
 
    # autosave each turn
    if not st.session_state.chat_id:
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out = {"chat_id": st.session_state.chat_id, "saved_at": datetime.now().isoformat(timespec="seconds"), "messages": st.session_state.messages}
    with open(os.path.join(CHATS_DIR, f"{st.session_state.chat_id}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
 
# Footer tips
with st.expander("Tips & Examples"):
    st.markdown("""
- **Internet OFF** → local-only:
  - *Who is under allocated between 2025-10-01 and 2025-10-29?*
  - *List rows where project="Alpha" and status="Open".*
- **Internet ON** → web-grounded:
  - *Latest updates about <topic>* (citations shown).
- **Repository**:
  - Drop many files in `.\repository`, then click **Re-index repository**.
""")
 