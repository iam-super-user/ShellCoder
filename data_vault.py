import sys, os, re, json, yaml, logging, hashlib, threading, time, argparse
import chromadb, ollama, pytesseract, docx
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIG ---
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR      = os.path.join(BASE_DIR, "vault_data")
STATE_FILE     = os.path.join(VAULT_DIR, ".vault_state.json")
DB_PATH        = os.path.join(VAULT_DIR, "chroma_db")
os.makedirs(VAULT_DIR, exist_ok=True)

# --- CONSTANTS (no more magic numbers) ---
CHUNK_SIZE          = 2000
CHUNK_OVERLAP       = 200
MIN_CHUNK_LENGTH    = 100
QUERY_TOP_K         = 7
RELEVANCE_THRESHOLD = 1.2   # ChromaDB L2 distance — lower = better match
MAX_WORKERS         = os.cpu_count() or 4
TESS_CONFIG         = r'--psm 6 -c preserve_interword_spaces=1'

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(VAULT_DIR, "vault_operations.log")),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# --- LAZY DB INIT (not at module import time) ---
_db              = None
_collection      = None
_chat_collection = None

def get_db() -> chromadb.PersistentClient:
    global _db
    if _db is None:
        _db = chromadb.PersistentClient(path=DB_PATH)
    return _db

def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_db().get_or_create_collection(name="arsenal_docs")
    return _collection

def get_chat_collection() -> chromadb.Collection:
    """Separate collection for Q&A history — keeps it out of the source knowledge base."""
    global _chat_collection
    if _chat_collection is None:
        _chat_collection = get_db().get_or_create_collection(name="chat_history")
    return _chat_collection

db_lock    = threading.RLock()
state_lock = threading.RLock()

# --- STATE ---
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state: dict) -> None:
    with state_lock:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

# --- HASHING (SHA-256, not MD5) ---
def get_file_hash(file_path: str) -> str | None:
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(8192), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception:
        return None

# --- TEXT CLEANING ---
def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse excessive blank lines
    text = re.sub(r'[ \t]+', ' ', text)       # normalize horizontal whitespace
    text = re.sub(r'\x00', '', text)          # strip null bytes
    return text.strip()

# --- SMART CHUNKING (file-type-aware, newline-respecting) ---
def chunk_content(content: str, file_ext: str) -> list[str]:
    content = clean_text(content)

    # Choose split strategy based on file type
    asm_exts  = {'.asm', '.s'}
    py_exts   = {'.py'}
    c_exts    = {'.c', '.cpp', '.h', '.hpp', '.rs'}

    if file_ext in asm_exts:
        # Split on ASM labels / section boundaries
        segments = re.split(r'(?=\n[a-zA-Z_][a-zA-Z0-9_]*:|\nsection\s|\nproc\s)', content)
    elif file_ext in py_exts:
        # Split on top-level def / class
        segments = re.split(r'(?=\ndef\s|\nclass\s)', content)
    elif file_ext in c_exts:
        # Split after closing brace at column 0
        segments = re.split(r'(?<=\n}\n)', content)
    else:
        # Generic: paragraph boundaries for docs, logs, text
        segments = re.split(r'\n\n+', content)

    # Aggregate segments into CHUNK_SIZE chunks with overlap
    chunks: list[str] = []
    current = ""
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(current) + len(seg) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + seg).strip()
        else:
            if current and len(current) >= MIN_CHUNK_LENGTH:
                chunks.append(current)
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = (overlap + "\n\n" + seg).strip()

    if current and len(current) >= MIN_CHUNK_LENGTH:
        chunks.append(current)

    # Fallback: hard character split with newline-boundary awareness
    if not chunks and content:
        i = 0
        n = len(content)
        while i < n:
            end = min(i + CHUNK_SIZE, n)
            if end < n:
                boundary = content.rfind('\n', i, end)
                if boundary > i:            # only snap back to a newline past i
                    end = boundary
            chunk = content[i:end].strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            if end >= n:
                break
            i = max(end - CHUNK_OVERLAP, i + 1)  # always move forward — no neg index/loop

    return chunks

# --- TEXT EXTRACTION ---
def extract_text_from_pdf(file_path: str) -> str | None:
    text = ""
    try:
        reader = PdfReader(file_path)
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        text = "\n\n".join(pages)  # page-separated, not blindly joined
    except Exception as e:
        logger.error(f"PDF read failed {file_path}: {e}")

    if len(text.strip()) < 50:
        logger.info(f"[*] Image-PDF detected. Running OCR on {os.path.basename(file_path)}...")
        try:
            ocr_pages = [
                pytesseract.image_to_string(img, config=TESS_CONFIG)
                for img in convert_from_path(file_path)
            ]
            return "\n\n".join(ocr_pages)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None
    return text

def get_text_from_file(file_path: str) -> str | None:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in {".lib", ".dll", ".exe", ".so", ".a", ".obj", ".bin", ".sys"}:
        return None
    if ext in {".idb", ".i64", ".gzf", ".gpr"}:
        logger.warning(f"[!] Skipped {ext} — export from Ghidra/IDA to .c or .asm first.")
        return None

    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)

        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            with Image.open(file_path) as img:
                return pytesseract.image_to_string(img, config=TESS_CONFIG)

        if ext == ".docx":
            doc = docx.Document(file_path)
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract tables (missed entirely in old code)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        parts.append(row_text)
            return "\n\n".join(parts)

        if ext == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)

        if ext in {".yaml", ".yml"}:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.dump(yaml.safe_load(f))

        text_exts = {
            ".c", ".cpp", ".h", ".hpp", ".rs", ".py", ".asm", ".s",
            ".sh", ".ps1", ".txt", ".md", ".csv", ".xml", ".html",
            ".log", ".hex", ".ihx", ".rtf", ""
        }
        if ext in text_exts:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-16', errors='ignore') as f:
                    return f.read()

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return None

# --- INGEST ---
def ingest_file(file_path: str) -> None:
    file_path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.isfile(file_path):
        return

    file_hash = get_file_hash(file_path)
    if file_hash is None:
        logger.warning(f"[!] Could not hash {file_path}, skipping.")
        return

    # Check hash under lock to prevent race condition between threads
    with state_lock:
        state = load_state()
        if state.get(file_path) == file_hash:
            return  # Unchanged — skip

    content = get_text_from_file(file_path)
    if not (content and content.strip()):
        return

    ext    = os.path.splitext(file_path)[1].lower()
    chunks = chunk_content(content, ext)

    if not chunks:
        logger.warning(f"[!] No usable chunks from {os.path.basename(file_path)}, skipping.")
        return

    collection = get_collection()
    with db_lock:
        collection.delete(where={"source": file_path})
        collection.add(
            ids=[f"{os.path.basename(file_path)}_{file_hash}_{idx}" for idx in range(len(chunks))],
            documents=chunks,
            metadatas=[{
                "source":       file_path,
                "filename":     os.path.basename(file_path),
                "type":         ext,
                "chunk_index":  idx,
                "total_chunks": len(chunks),
            } for idx in range(len(chunks))]
        )

    with state_lock:
        state = load_state()
        state[file_path] = file_hash
        save_state(state)

    logger.info(f"[+] Ingested: {os.path.basename(file_path)} ({len(chunks)} chunks)")

def ingest_directory(dir_path: str) -> None:
    files = [
        os.path.join(root, fname)
        for root, _, fnames in os.walk(os.path.expanduser(dir_path))
        for fname in fnames
    ]
    logger.info(f"[*] Starting concurrent ingestion of {len(files)} files...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(ingest_file, files)
    logger.info("[+] Bulk ingestion complete.")

def purge_deleted_files() -> None:
    """Remove chunks and state entries for files that no longer exist on disk."""
    state      = load_state()
    collection = get_collection()
    removed    = [path for path in state if not os.path.exists(path)]

    if not removed:
        logger.info("[*] No stale entries to purge.")
        return

    for path in removed:
        with db_lock:
            collection.delete(where={"source": path})
        del state[path]
        logger.info(f"[-] Purged: {os.path.basename(path)}")

    save_state(state)
    logger.info(f"[+] Purge complete — removed {len(removed)} stale file(s).")

# --- WATCHER ---
class VaultEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            ingest_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            ingest_file(event.src_path)

    def on_deleted(self, event):
        """Remove chunks when a file is deleted from the watched directory."""
        if not event.is_directory:
            file_path  = os.path.abspath(event.src_path)
            collection = get_collection()
            with db_lock:
                collection.delete(where={"source": file_path})
            with state_lock:
                state = load_state()
                state.pop(file_path, None)
                save_state(state)
            logger.info(f"[-] Removed chunks for deleted file: {os.path.basename(file_path)}")

def watch_directory(dir_path: str) -> None:
    logger.info(f"[*] Live watch started on {dir_path}...")
    observer = Observer()
    observer.schedule(VaultEventHandler(), os.path.expanduser(dir_path), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- QUERY ---
def query(user_input: str) -> None:
    logger.info(f"[*] Searching Vault for: '{user_input}'...")
    collection = get_collection()

    results = collection.query(
        query_texts=[user_input],
        n_results=min(QUERY_TOP_K + 3, 10),  # fetch extras, then threshold-filter
        include=["documents", "distances", "metadatas"]
    )

    docs      = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    # Filter by relevance threshold — don't feed noise to the LLM
    filtered = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist < RELEVANCE_THRESHOLD
    ][:QUERY_TOP_K]

    if not filtered:
        print("\n[SHELLCODER]: No relevant data found in vault for your query.")
        return

    # Build context with per-chunk source attribution
    context_parts = []
    for doc, dist, meta in filtered:
        filename  = meta.get('filename', os.path.basename(meta.get('source', 'unknown')))
        ftype     = meta.get('type', '')
        chunk_idx = meta.get('chunk_index', '?')
        total     = meta.get('total_chunks', '?')
        relevance = round(max(0.0, 1.0 - dist), 2)
        header    = f"[SOURCE: {filename} ({ftype}) | Chunk {chunk_idx}/{total} | Relevance: {relevance}]"
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n--- [VAULT CHUNK] ---\n\n".join(context_parts)

    messages = [
        {
            'role': 'system',
            'content': (
                "You are Shellcoder, an expert in exploit development, vulnerability research, "
                "and malware analysis. Answer based STRICTLY on the provided CONTEXT. "
                "If the context does not contain enough information, say so explicitly. "
                "Always preserve code formatting exactly as shown.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {'role': 'user', 'content': user_input},
    ]

    # Stream output token by token — no more blank terminal wait
    print("\n[SHELLCODER]:")
    full_response = ""
    try:
        for chunk in ollama.chat(model='Shellcoder', messages=messages, stream=True):
            token = chunk['message']['content']
            print(token, end='', flush=True)
            full_response += token
        print()
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return

    # Store Q&A in SEPARATE chat_history collection — keeps arsenal_docs clean
    chat_id = f"chat_{abs(hash(user_input + str(time.time())))}"
    with db_lock:
        get_chat_collection().add(
            ids=[chat_id],
            documents=[f"Q: {user_input}\nA: {full_response}"],
            metadatas=[{"source": "chat", "timestamp": str(time.time())}]
        )

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="data_vault",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
======================================================================
    DATA VAULT: Exploit Dev & Malware Analysis Brain
======================================================================
A multimodal RAG pipeline for vulnerability researchers.
Automatically ingests code, PDFs, memory dumps, and images (OCR) into
a local vector database and queries them using the Shellcoder LLM.
        """,
        epilog="""
COMMAND EXAMPLES:
  Ingest a single file:
    python data_vault.py --ingest ./shellcode.c

  Bulk ingest an entire folder (Recursive):
    python data_vault.py --ingest-dir ./CVE-2026-0001/

  Start Live Monitor (Auto-ingests when a file is saved/modified):
    python data_vault.py --watch ./Reverse_Engineering/

  Ask the AI a question against your loaded vault data:
    python data_vault.py --query "What is the SEH offset mentioned in the PDF?"

  Purge stale chunks for files no longer on disk:
    python data_vault.py --purge
        """
    )
    parser.add_argument("--ingest",     type=str,            metavar="FILE", help="Path to a single file to ingest")
    parser.add_argument("--ingest-dir", type=str,            metavar="DIR",  help="Path to a folder to bulk-ingest (recursive)")
    parser.add_argument("--watch",      type=str,            metavar="DIR",  help="Path to a folder to monitor live")
    parser.add_argument("--query",      type=str,            metavar="TEXT", help="Ask a question against your vault")
    parser.add_argument("--purge",      action="store_true",                 help="Remove chunks for files no longer on disk")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.ingest:
        ingest_file(args.ingest)
    elif args.ingest_dir:
        ingest_directory(args.ingest_dir)
    elif args.watch:
        watch_directory(args.watch)
    elif args.query:
        query(args.query)
    elif args.purge:
        purge_deleted_files()
