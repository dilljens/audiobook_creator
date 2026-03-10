"""
gui_proper_noun_player.py
──────────────────────────
GUI for auditing proper noun pronunciations — supports multiple books.

Each book's data is isolated in its own subdirectory:
  output_proper_nouns/<book_slug>/manifest.json
  output_proper_nouns/<book_slug>/correct_words.json
  output_proper_nouns/<book_slug>/pronunciation_fixes.json
  proper_nouns_audio/<book_slug>/<word>.wav
  proper_nouns_audio/<book_slug>/replacements_cache/<phonetic>.wav

Three columns (all persisted as JSON per book):
  • Review   – words not yet audited
  • Correct  – words that already pronounce fine
  • Fixes    – linked list: original word → phonetic replacement
               e.g.  "Nephi" → "Kneephi"

Hotkeys (always active):
  Space          – replay current word
  s              – stop audio
  Escape         – reset fix entry to original word, refocus review list

On the Review list:
  ↑ / ↓          – navigate
  Click / Enter  – play word AND focus fix entry

On the fix entry (bottom bar, right of the word label):
  Start typing to overwrite the pre-filled word.
  Enter  →  if text == original word  →  mark Correct, advance to next
            if text differs           →  add as Fix, advance to next
  Escape →  reset text to original word, return focus to review list

On the Correct list:
  Delete / BackSpace – move selected word back to Review

On the Fixes list:
  Delete / BackSpace – move selected fix back to Review

"Apply Fixes to Text" writes a TTS-ready copy of the source file with all
substitutions applied (case-sensitive whole-word replace).

Run:
    .venv/bin/python gui_proper_noun_player.py
"""

import json
import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import NamedTuple

# Model is already cached locally — skip all HuggingFace Hub network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sounddevice as sd
import soundfile as sf

VOICE       = "am_michael"
SAMPLE_RATE = 24000

# ── Book source ────────────────────────────────────────────────────────────────

class BookSource(NamedTuple):
    label: str          # Display name shown in the UI
    slug: str           # Filesystem-safe identifier used for subdirectory names
    source_paths: list  # list[Path] — one or more source .txt files
    fixed_out: Path     # Where "Apply Fixes to Text" writes the TTS-ready copy


def _book_slug(text: str) -> str:
    """Convert a display name to a lowercase filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")[:60].lower()


def discover_books(root: Path = Path(".")) -> list[BookSource]:
    """Scan the workspace root for candidate source text files and directories."""
    books: list[BookSource] = []
    EXCLUDE = {"tts fixed", "proper_noun", "table of contents", "columns pdf"}

    # Root-level .txt files (single-file books)
    for f in sorted(root.glob("*.txt")):
        name_lower = f.stem.lower()
        if any(kw in name_lower for kw in EXCLUDE):
            continue
        if f.stat().st_size < 10_000:   # skip tiny/metadata files
            continue
        slug = _book_slug(f.stem)
        fixed_out = f.parent / f"{f.stem} (TTS Fixed).txt"
        books.append(BookSource(label=f.stem, slug=slug,
                                source_paths=[f], fixed_out=fixed_out))

    # Sub-directories containing .txt files (multi-file books, e.g. Lightbringer)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(("output_", "proper_noun", "__", ".")):
            continue
        txts = sorted(d.glob("*.txt"))
        if not txts:
            continue
        slug = _book_slug(d.name)
        fixed_out = d / f"{d.name} (TTS Fixed).txt"
        books.append(BookSource(label=d.name, slug=slug,
                                source_paths=list(txts), fixed_out=fixed_out))

    return books

# ── Colours ────────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
BG2     = "#181825"
BG3     = "#313244"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"
GREEN   = "#a6e3a1"
BLUE    = "#89b4fa"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
MAUVE   = "#cba6f7"

# ── Audio ──────────────────────────────────────────────────────────────────────

def play_async(path: Path) -> None:
    sd.stop()
    def _play():
        try:
            data, sr = sf.read(str(path), dtype="float32")
            sd.play(data, sr)
        except Exception as exc:
            print(f"[audio] playback error: {exc}")
    threading.Thread(target=_play, daemon=True).start()


def _slug(text: str) -> str:
    """Safe filename from arbitrary text."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")[:80]


# Lazy KPipeline singleton — only imported+loaded on first synthesis request
_pipeline = None
_pipeline_lock = threading.Lock()

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                import warnings
                from kokoro import KPipeline  # type: ignore
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", message=".*unauthenticated.*")
                    _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def synth_and_play(text: str, replacements_dir: Path, on_ready=None) -> None:
    """Synthesise *text* with Kokoro (cached to *replacements_dir*) and play it.
    Runs entirely on a daemon thread so the GUI never blocks.
    *on_ready(path)* is called on the same thread once the file is written.
    """
    def _run():
        try:
            path = _synth_to_cache(text, replacements_dir)
            if path:
                if on_ready:
                    on_ready(path)
                play_async(path)
        except Exception as exc:
            print(f"[synth] error synthesising '{text}': {exc}")

    threading.Thread(target=_run, daemon=True).start()


def _synth_to_cache(text: str, replacements_dir: Path) -> "Path | None":
    """Synthesise *text* to a cached WAV and return its path (or None on failure).
    Skips synthesis if the file already exists.  Safe to call from any thread.
    """
    replacements_dir.mkdir(parents=True, exist_ok=True)
    cache_path = replacements_dir / f"{_slug(text)}.wav"
    if not cache_path.exists():
        import warnings
        import numpy as np
        pipeline = _get_pipeline()
        chunks = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for _, _, audio in pipeline(text, voice=VOICE):
                if audio is not None:
                    chunks.append(audio)
        if chunks:
            combined = np.concatenate(chunks)
            sf.write(str(cache_path), combined, SAMPLE_RATE)
    return cache_path if cache_path.exists() else None


# ── Persistence helpers ────────────────────────────────────────────────────────

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Styled widget helpers ──────────────────────────────────────────────────────

def make_listbox(parent) -> tuple[tk.Listbox, tk.Frame]:
    frame = tk.Frame(parent, bg=BG2, bd=0)
    sb = ttk.Scrollbar(frame, orient="vertical")
    sb.pack(side="right", fill="y")
    lb = tk.Listbox(
        frame,
        yscrollcommand=sb.set,
        font=("Helvetica", 11),
        bg=BG2, fg=FG,
        selectbackground=BLUE, selectforeground=BG,
        activestyle="none", bd=0, highlightthickness=0, relief="flat",
        exportselection=False,
    )
    lb.pack(side="left", fill="both", expand=True)
    sb.config(command=lb.yview)
    return lb, frame

def styled_btn(parent, text, command, color=FG, bg=BG3, **kw):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=color, activebackground=BG2, activeforeground=color,
        font=("Helvetica", 10, "bold"), relief="flat", bd=0,
        padx=10, pady=5, cursor="hand2", **kw
    )

def section_label(parent, text):
    return tk.Label(parent, text=text, bg=BG, fg=FG_DIM,
                    font=("Helvetica", 9, "bold"), anchor="w")


# ── Main app ───────────────────────────────────────────────────────────────────

class ProperNounAuditor(tk.Tk):

    # tracks which word is currently loaded into the fix entry
    _fix_entry_word: str = ""

    def __init__(self, books: list[BookSource]) -> None:
        super().__init__()
        self.title("Proper Noun Pronunciation Auditor")
        self.geometry("1020x760")
        self.minsize(800, 560)
        self.configure(bg=BG)

        self.books: list[BookSource] = books
        self.book: BookSource | None = None

        # Loaded per-book via _load_book()
        self.manifest: dict[str, str] = {}
        self.all_words: list[str] = []
        self.correct: list[str] = []
        self.fixes: dict[str, str] = {}

        self._build_ui()
        self._alive = True
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Window-level hotkeys
        self.bind("<space>",  lambda e: self._replay())
        self.bind("s",        lambda e: sd.stop())
        self.bind("r",        lambda e: self._regen_current()
                  if self.focus_get() is not self._fix_entry else None)
        self.bind("<Escape>", lambda e: self._reset_fix_entry())

        # Auto-load first book that already has a manifest; otherwise select first
        for book in books:
            if (Path("output_proper_nouns") / book.slug / "manifest.json").exists():
                self._load_book(book)
                break
        else:
            if books:
                self._book_var.set(books[0].label)
                self._on_book_change()

    # ── Per-book path properties ─────────────────────────────────────────────────

    @property
    def _data_dir(self) -> Path:
        return Path("output_proper_nouns") / self.book.slug

    @property
    def _audio_dir(self) -> Path:
        return Path("proper_nouns_audio") / self.book.slug

    @property
    def _manifest_file(self) -> Path:
        return self._data_dir / "manifest.json"

    @property
    def _replacements_dir(self) -> Path:
        return self._audio_dir / "replacements_cache"

    @property
    def _correct_file(self) -> Path:
        return self._data_dir / "correct_words.json"

    @property
    def _fixes_file(self) -> Path:
        return self._data_dir / "pronunciation_fixes.json"

    # ── Book loading / switching ──────────────────────────────────────────────────

    def _load_book(self, book: BookSource) -> None:
        """Switch to *book* — reload all state from its per-book data files."""
        sd.stop()
        self.book = book
        self._book_var.set(book.label)

        if self._manifest_file.exists():
            self.manifest = load_json(self._manifest_file, {})
        else:
            self.manifest = {}

        self.all_words = sorted(self.manifest.keys(), key=str.casefold)
        self.correct   = load_json(self._correct_file, [])
        self.fixes     = load_json(self._fixes_file, {})

        n = len(self.manifest)
        if n:
            status = f"{n} words loaded  ·  {len(self.correct)} correct  ·  {len(self.fixes)} fixes"
        else:
            status = "No manifest yet — click  ⚙ Extract & Generate Audio  to create one"
        self._book_status_var.set(status)

        self._refresh_all()
        self.fix_var.set("")
        self._fix_entry_word = ""
        self.now_playing_var.set("—")

    def _on_book_change(self, event=None) -> None:
        label = self._book_var.get()
        book = next((b for b in self.books if b.label == label), None)
        if book:
            self._load_book(book)

    def _on_close(self) -> None:
        self._alive = False
        sd.stop()
        self.destroy()

    def _safe_after(self, ms: int, func) -> None:
        """Schedule func on the Tk thread; silently no-ops if window is gone."""
        if self._alive:
            try:
                self.after(ms, func)
            except RuntimeError:
                pass

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        PAD = 8

        # ── Book selector bar ──────────────────────────────────────────────────────
        book_bar = tk.Frame(self, bg=BG2, pady=7)
        book_bar.pack(fill="x")

        tk.Label(book_bar, text="  Book:", bg=BG2, fg=FG_DIM,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=(10, 4))

        self._book_var = tk.StringVar()
        book_menu = ttk.Combobox(
            book_bar, textvariable=self._book_var,
            values=[b.label for b in self.books],
            state="readonly", font=("Helvetica", 10), width=44)
        book_menu.pack(side="left", padx=(0, 8))
        book_menu.bind("<<ComboboxSelected>>", self._on_book_change)

        self._extract_btn = styled_btn(
            book_bar, "⚙ Extract & Generate Audio",
            self._extract_and_generate, color=GREEN, bg=BG3)
        self._extract_btn.pack(side="left", padx=4)

        styled_btn(book_bar, "⇄ Apply Fixes to Text",
                   self._apply_fixes, color=YELLOW, bg=BG3).pack(side="left", padx=4)
        styled_btn(book_bar, "⬇ Export Remaining",
                   self._export_remaining, color=BLUE, bg=BG3).pack(side="left", padx=4)

        self._book_status_var = tk.StringVar(value="Select a book above")
        tk.Label(book_bar, textvariable=self._book_status_var,
                 bg=BG2, fg=FG_DIM, font=("Helvetica", 9),
                 anchor="w").pack(side="left", padx=(10, 10))

        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=BG, pady=6)
        title_bar.pack(fill="x", padx=PAD)
        tk.Label(title_bar, text="Proper Noun Pronunciation Auditor",
                 font=("Helvetica", 15, "bold"), bg=BG, fg=FG).pack(side="left")
        hint = "Space=replay  r=regen  s=stop  Esc=reset fix  Del=remove from list  Enter=correct|fix"
        tk.Label(title_bar, text=hint,
                 font=("Helvetica", 8), bg=BG, fg=FG_DIM).pack(side="left", padx=14)

        # Three-column body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=PAD, pady=(0, PAD))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # ── Column 0: Review list ──────────────────────────────────────────────
        col0 = tk.Frame(body, bg=BG)
        col0.grid(row=0, column=0, sticky="nsew", padx=(0, PAD))

        filter_row = tk.Frame(col0, bg=BG)
        filter_row.pack(fill="x", pady=(0, 4))
        tk.Label(filter_row, text="Filter:", bg=BG, fg=FG,
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_review())
        self._filter_entry = tk.Entry(
            filter_row, textvariable=self.search_var,
            font=("Helvetica", 11), bg=BG3, fg=FG,
            insertbackground=FG, relief="flat", bd=4)
        self._filter_entry.pack(side="left", fill="x", expand=True)
        self._filter_entry.focus_set()
        styled_btn(filter_row, "✕", lambda: self.search_var.set(""),
                   color=RED, bg=BG3).pack(side="left", padx=(3, 0))

        hdr0 = tk.Frame(col0, bg=BG)
        hdr0.pack(fill="x")
        section_label(hdr0, "TO REVIEW").pack(side="left")
        self.review_count_var = tk.StringVar()
        tk.Label(hdr0, textvariable=self.review_count_var, bg=BG, fg=FG_DIM,
                 font=("Helvetica", 9)).pack(side="right")

        self.review_lb, review_frame = make_listbox(col0)
        review_frame.pack(fill="both", expand=True)
        self.review_lb.bind("<<ListboxSelect>>", self._on_review_select)
        self.review_lb.bind("<Return>", self._on_review_select)

        # ── Column 1: Correct list ─────────────────────────────────────────────
        col1 = tk.Frame(body, bg=BG)
        col1.grid(row=0, column=1, sticky="nsew", padx=(0, PAD))

        hdr1 = tk.Frame(col1, bg=BG)
        hdr1.pack(fill="x")
        section_label(hdr1, "✓ CORRECT  [Del=remove]").pack(side="left")
        self.correct_count_var = tk.StringVar()
        tk.Label(hdr1, textvariable=self.correct_count_var, bg=BG, fg=FG_DIM,
                 font=("Helvetica", 9)).pack(side="right")

        self.correct_lb, correct_frame = make_listbox(col1)
        correct_frame.pack(fill="both", expand=True)
        self.correct_lb.bind("<<ListboxSelect>>",
                             lambda e: self._on_side_select(self.correct_lb))
        self.correct_lb.bind("<Delete>",
                             lambda e: self._move_back(self.correct_lb, is_dict=False))
        self.correct_lb.bind("<BackSpace>",
                             lambda e: self._move_back(self.correct_lb, is_dict=False))

        styled_btn(col1, "← Back to Review  [Del]",
                   lambda: self._move_back(self.correct_lb, is_dict=False),
                   color=YELLOW).pack(fill="x", pady=(4, 0))

        # ── Column 2: Fixes list ───────────────────────────────────────────────
        col2 = tk.Frame(body, bg=BG)
        col2.grid(row=0, column=2, sticky="nsew")

        hdr2 = tk.Frame(col2, bg=BG)
        hdr2.pack(fill="x")
        section_label(hdr2, "⇄ FIXES  (original → phonetic)").pack(side="left")
        self.fixes_count_var = tk.StringVar()
        tk.Label(hdr2, textvariable=self.fixes_count_var, bg=BG, fg=FG_DIM,
                 font=("Helvetica", 9)).pack(side="right")

        self.fixes_lb, fixes_frame = make_listbox(col2)
        fixes_frame.pack(fill="both", expand=True)
        self.fixes_lb.bind("<<ListboxSelect>>",
                           lambda e: self._on_side_select(self.fixes_lb))
        self.fixes_lb.bind("<Delete>",
                           lambda e: self._move_back(self.fixes_lb, is_dict=True))
        self.fixes_lb.bind("<BackSpace>",
                           lambda e: self._move_back(self.fixes_lb, is_dict=True))

        styled_btn(col2, "← Back to Review  [Del]",
                   lambda: self._move_back(self.fixes_lb, is_dict=True),
                   color=YELLOW).pack(fill="x", pady=(4, 0))

        # ── Bottom action bar ──────────────────────────────────────────────────
        action_bar = tk.Frame(self, bg=BG3, pady=8)
        action_bar.pack(fill="x")

        # Now-playing word label
        tk.Label(action_bar, text="▶", bg=BG3, fg=GREEN,
                 font=("Helvetica", 11)).pack(side="left", padx=(10, 2))
        self.now_playing_var = tk.StringVar(value="—")
        tk.Label(action_bar, textvariable=self.now_playing_var,
                 bg=BG3, fg=GREEN, font=("Helvetica", 11, "bold"),
                 width=20, anchor="w").pack(side="left")

        # Inline fix entry — right next to the word, auto-focused on word click
        tk.Label(action_bar, text="→", bg=BG3, fg=MAUVE,
                 font=("Helvetica", 13, "bold")).pack(side="left", padx=(6, 3))
        self.fix_var = tk.StringVar()
        self._fix_entry = tk.Entry(
            action_bar, textvariable=self.fix_var,
            font=("Helvetica", 11), bg=BG2, fg=MAUVE,
            insertbackground=MAUVE, relief="flat", bd=4, width=22)
        self._fix_entry.pack(side="left")
        self._fix_entry.bind("<Return>", lambda e: self._enter_action())
        self._fix_entry.bind("<Escape>", lambda e: self._reset_fix_entry())
        self._fix_entry.bind("<Up>",   lambda e: (self._navigate_review(-1), "break")[1])
        self._fix_entry.bind("<Down>", lambda e: (self._navigate_review(+1), "break")[1])

        tk.Label(action_bar, text="Enter=correct  (edit first for fix)  Esc=reset",
                 bg=BG3, fg=FG_DIM, font=("Helvetica", 8)).pack(side="left", padx=(5, 10))

        tk.Label(action_bar, text="│", bg=BG3, fg=FG_DIM).pack(side="left", padx=4)
        styled_btn(action_bar, "■ Stop  [s]", sd.stop,
                   color=RED).pack(side="left", padx=4)
        styled_btn(action_bar, "↺ Replay  [Space]", self._replay,
                   color=BLUE).pack(side="left", padx=2)
        styled_btn(action_bar, "↻ Regen  [r]", self._regen_current,
                   color=GREEN).pack(side="left", padx=2)

        tk.Label(action_bar, text="│", bg=BG3, fg=FG_DIM).pack(side="left", padx=4)
        self._pregen_btn = styled_btn(
            action_bar, "↻ Pre-gen Fix Audio",
            self._pregen_all_fix_audio, color=MAUVE, bg=BG2)
        self._pregen_btn.pack(side="left", padx=4)
        self._pregen_status_var = tk.StringVar(value="")
        tk.Label(action_bar, textvariable=self._pregen_status_var,
                 bg=BG3, fg=FG_DIM, font=("Helvetica", 8),
                 width=28, anchor="w").pack(side="left", padx=(4, 10))

    # ── Refresh helpers ────────────────────────────────────────────────────────

    def _review_words(self) -> list[str]:
        excluded = set(self.correct) | set(self.fixes.keys())
        q = self.search_var.get().strip().casefold()
        words = [w for w in self.all_words if w not in excluded]
        if q:
            words = [w for w in words if q in w.casefold()]
        return words

    def _refresh_review(self) -> None:
        words = self._review_words()
        self.review_lb.delete(0, "end")
        for w in words:
            self.review_lb.insert("end", f"  {w}")
        self.review_count_var.set(f"{len(words)}")

    def _refresh_correct(self) -> None:
        self.correct_lb.delete(0, "end")
        for w in self.correct:  # already newest-first
            self.correct_lb.insert("end", f"  {w}")
        self.correct_count_var.set(f"{len(self.correct)}")

    def _refresh_fixes(self) -> None:
        self.fixes_lb.delete(0, "end")
        for orig, rep in reversed(list(self.fixes.items())):  # newest-first
            self.fixes_lb.insert("end", f"  {orig}  →  {rep}")
        self.fixes_count_var.set(f"{len(self.fixes)}")

    def _refresh_all(self) -> None:
        self._refresh_review()
        self._refresh_correct()
        self._refresh_fixes()

    # ── Playback ───────────────────────────────────────────────────────────────

    def _play_word(self, word: str) -> None:
        if not self.book:
            return
        wav_name = self.manifest.get(word)
        if not wav_name:
            return
        wav_path = self._audio_dir / wav_name
        if not wav_path.exists():
            messagebox.showwarning("Missing audio",
                                   f"No audio file for '{word}'.\n"
                                   "Click '⚙ Extract & Generate Audio' first.")
            return
        self.now_playing_var.set(word)
        play_async(wav_path)

    # ── Selection callbacks ────────────────────────────────────────────────────

    def _on_review_select(self, event=None) -> None:
        sel = self.review_lb.curselection()
        if not sel:
            return
        word = self.review_lb.get(sel[0]).strip()
        self._fix_entry_word = word
        self.fix_var.set(word)              # pre-fill fix entry with the word
        self._fix_entry.selection_range(0, "end")
        self._fix_entry.icursor("end")
        # Defer focus so the listbox doesn't reclaim it after the click event settles
        self.after(0, self._fix_entry.focus_set)
        self._play_word(word)

    def _on_side_select(self, listbox: tk.Listbox) -> None:
        if not self.book:
            return
        sel = listbox.curselection()
        if not sel:
            return
        row = listbox.get(sel[0]).strip()
        parts = row.split("  →  ")
        original = parts[0].strip()

        if listbox is self.fixes_lb and len(parts) == 2:
            replacement = parts[1].strip()
            self._fix_entry_word = original
            self.fix_var.set(replacement)
            self.now_playing_var.set(f"… {replacement}")
            rdir = self._replacements_dir
            def _on_ready(_path):
                self._safe_after(0, lambda: self.now_playing_var.set(replacement))
            synth_and_play(replacement, rdir, on_ready=_on_ready)
        else:
            self._fix_entry_word = original
            self.fix_var.set(original)
            self._play_word(original)

    # ── Actions ────────────────────────────────────────────────────────────────

    def _selected_review_word(self) -> str | None:
        sel = self.review_lb.curselection()
        if not sel:
            return None
        return self.review_lb.get(sel[0]).strip()

    def _enter_action(self) -> None:
        """Smart Enter handler for the fix entry.

        If the entry text matches the original word  → mark Correct.
        If the entry text differs from the original  → add as Fix.
        """
        word = self._fix_entry_word or self._selected_review_word()
        if not word:
            return
        text = self.fix_var.get().strip()
        if not text or text == word:
            self._mark_correct_word(word)
        else:
            self._add_fix_for_word(word, text)

    def _reset_fix_entry(self) -> None:
        """Escape: reset fix entry to the original word, refocus the review list."""
        self.fix_var.set(self._fix_entry_word)
        self.review_lb.focus_set()

    def _replay(self) -> None:
        if self._fix_entry_word:
            self._play_word(self._fix_entry_word)

    def _regen_current(self) -> None:
        """Delete the cached WAV for the current word/replacement and re-synthesise."""
        word = self._fix_entry_word
        if not word:
            return

        # Determine which file to delete based on context
        fix_text = self.fix_var.get().strip()
        # If the fix box contains something different from the word, regen that text
        is_fix_replacement = bool(fix_text and fix_text != word)

        if not self.book:
            return
        if is_fix_replacement:
            target = self._replacements_dir / f"{_slug(fix_text)}.wav"
            if target.exists():
                target.unlink()
            self.now_playing_var.set(f"… regen {fix_text}")
            rdir = self._replacements_dir
            def _on_ready(_p):
                self._safe_after(0, lambda: self.now_playing_var.set(fix_text))
            synth_and_play(fix_text, rdir, on_ready=_on_ready)
        else:
            wav_name = self.manifest.get(word)
            if not wav_name:
                return
            wav_path = self._audio_dir / wav_name
            if wav_path.exists():
                wav_path.unlink()
            self.now_playing_var.set(f"… regen {word}")

            def _regen():
                try:
                    import warnings, numpy as np
                    pipeline = _get_pipeline()
                    chunks = []
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        for _, _, audio in pipeline(word, voice=VOICE):
                            if audio is not None:
                                chunks.append(audio)
                    if chunks:
                        sf.write(str(wav_path), np.concatenate(chunks), SAMPLE_RATE)
                        self._safe_after(0, lambda: self.now_playing_var.set(word))
                        play_async(wav_path)
                except Exception as exc:
                    print(f"[regen] error for '{word}': {exc}")

            threading.Thread(target=_regen, daemon=True).start()

    def _navigate_review(self, delta: int) -> None:
        """Move the review list selection up (delta=-1) or down (delta=+1)."""
        size = self.review_lb.size()
        if size == 0:
            return
        sel = self.review_lb.curselection()
        current = sel[0] if sel else -1
        new_idx = max(0, min(size - 1, current + delta))
        if new_idx == current:
            return
        self.review_lb.selection_clear(0, "end")
        self.review_lb.selection_set(new_idx)
        self.review_lb.see(new_idx)
        self.review_lb.event_generate("<<ListboxSelect>>")

    def _advance_review(self, from_idx: int = 0) -> None:
        """Select the item at from_idx (clamped), positioned in the upper portion
        of the viewport so the word doesn't end up in the bottom half unless
        the list can't scroll any further down."""
        size = self.review_lb.size()
        if size == 0:
            return
        target = min(from_idx, size - 1)
        self.review_lb.selection_clear(0, "end")
        self.review_lb.selection_set(target)

        # First call see() to let tk calculate the viewport, then reposition.
        self.review_lb.see(target)
        self.review_lb.update_idletasks()

        first, last = self.review_lb.yview()
        visible_count = max(1, round((last - first) * size))

        # Ideal top-of-viewport: put target ~1/4 down from the top
        ideal_top = target - visible_count // 4
        ideal_top = max(0, ideal_top)

        self.review_lb.yview_moveto(ideal_top / size)
        self.review_lb.event_generate("<<ListboxSelect>>")

    def _mark_correct_word(self, word: str) -> None:
        idx = self.review_lb.curselection()
        from_idx = idx[0] if idx else 0
        if word not in self.correct:
            self.correct.insert(0, word)
        save_json(self._correct_file, self.correct)
        self._fix_entry_word = ""
        self.fix_var.set("")
        self.now_playing_var.set("—")
        self._refresh_all()
        self._advance_review(from_idx)

    def _add_fix_for_word(self, word: str, replacement: str) -> None:
        idx = self.review_lb.curselection()
        from_idx = idx[0] if idx else 0
        self.fixes.pop(word, None)
        self.fixes[word] = replacement
        save_json(self._fixes_file, self.fixes)
        self._fix_entry_word = ""
        self.fix_var.set("")
        self.now_playing_var.set("—")
        self._refresh_all()
        self._advance_review(from_idx)

    def _move_back(self, listbox: tk.Listbox, is_dict: bool) -> None:
        sel = listbox.curselection()
        if not sel:
            return
        raw = listbox.get(sel[0]).strip().split("  →  ")[0].strip()
        if is_dict:
            self.fixes.pop(raw, None)
            save_json(self._fixes_file, self.fixes)
            if raw in self.correct:
                self.correct.remove(raw)
                save_json(self._correct_file, self.correct)
        else:
            if raw in self.correct:
                self.correct.remove(raw)
            save_json(self._correct_file, self.correct)
        self._refresh_all()

    # ── Extract & Generate ─────────────────────────────────────────────────────────────

    def _extract_and_generate(self) -> None:
        """Extract proper nouns from the selected book’s source text, then
        generate a TTS audio clip for each one.  Runs in a background thread.
        """
        if not self.book:
            messagebox.showinfo("No book selected", "Please select a book first.")
            return

        missing = [p for p in self.book.source_paths if not p.exists()]
        if missing:
            messagebox.showerror(
                "Source file(s) not found",
                "Could not find:\n" + "\n".join(str(p) for p in missing))
            return

        self._extract_btn.config(state="disabled")
        self._book_status_var.set("Loading spaCy NLP model…")
        book = self.book   # capture for the thread

        def _run():
            try:
                self._safe_after(0, lambda: self._book_status_var.set(
                    "Running NLP extraction (may take a minute)…"))
                words = _extract_nouns_from_paths(book.source_paths)
                n_extracted = len(words)
                self._safe_after(0, lambda: self._book_status_var.set(
                    f"Extracted {n_extracted} nouns — generating audio…"))

                data_dir  = Path("output_proper_nouns") / book.slug
                audio_dir = Path("proper_nouns_audio")  / book.slug
                data_dir.mkdir(parents=True, exist_ok=True)
                audio_dir.mkdir(parents=True, exist_ok=True)

                manifest_path = data_dir / "manifest.json"
                manifest: dict = load_json(manifest_path, {})

                pipeline = _get_pipeline()
                done = failed = 0

                for i, word in enumerate(sorted(words, key=str.casefold)):
                    word_slug = re.sub(r"[^a-z0-9]+", "_", word.lower()).strip("_")
                    wav_name  = f"{word_slug}.wav"
                    wav_path  = audio_dir / wav_name

                    if word in manifest and wav_path.exists():
                        continue

                    try:
                        import warnings, numpy as np
                        chunks = []
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            for _, _, audio in pipeline(word, voice=VOICE):
                                if audio is not None:
                                    chunks.append(audio)
                        if chunks:
                            sf.write(str(wav_path), np.concatenate(chunks), SAMPLE_RATE)
                            manifest[word] = wav_name
                            done += 1
                        else:
                            failed += 1
                    except Exception as exc:
                        print(f"[gen] failed for '{word}': {exc}")
                        failed += 1

                    if i % 10 == 0:
                        remaining = n_extracted - i
                        self._safe_after(0, lambda r=remaining:
                            self._book_status_var.set(f"Generating audio… {r} remaining"))

                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2))
                self._safe_after(0, lambda:
                    self._finish_extract(book, manifest, done, failed))

            except ImportError as exc:
                msg = (f"Missing dependency: {exc}\n\n"
                       "Install with:  pip install spacy wordfreq\n"
                       "Then:          python -m spacy download en_core_web_sm")
                self._safe_after(0, lambda m=msg: messagebox.showerror(
                    "Missing package", m))
                self._safe_after(0, lambda: self._book_status_var.set("Error — see popup"))
                self._safe_after(0, lambda: self._extract_btn.config(state="normal"))
            except Exception as exc:
                err = str(exc)
                self._safe_after(0, lambda e=err: self._book_status_var.set(f"Error: {e}"))
                self._safe_after(0, lambda: self._extract_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _finish_extract(self, book: BookSource, manifest: dict,
                        done: int, failed: int) -> None:
        self._extract_btn.config(state="normal")
        self._book_status_var.set(
            f"Done — {len(manifest)} words total  ({done} new, {failed} failed)")
        if self.book and self.book.slug == book.slug:
            self._load_book(book)

    def _pregen_all_fix_audio(self) -> None:
        if not self.book:
            return
        if not self.fixes:
            messagebox.showinfo("No fixes", "The Fixes list is empty.")
            return

        replacements = list(self.fixes.values())
        total = len(replacements)
        rdir = self._replacements_dir
        already = sum(1 for r in replacements if (rdir / f"{_slug(r)}.wav").exists())
        new_count = total - already
        if new_count == 0:
            messagebox.showinfo("Already done",
                                f"All {total} replacement clips already exist.")
            return

        self._pregen_btn.config(state="disabled")
        self._pregen_status_var.set(f"0 / {new_count} new  ({already} cached)")

        def _run():
            try:
                done = 0
                for rep in replacements:
                    if not (rdir / f"{_slug(rep)}.wav").exists():
                        _synth_to_cache(rep, rdir)
                        done += 1
                        self._safe_after(0, lambda d=done, t=new_count:
                            self._pregen_status_var.set(f"{d} / {t} synthesised…"))
                self._safe_after(0, lambda:
                    self._pregen_status_var.set(f"Done — {total} clips ready"))
            except Exception as exc:
                print(f"[pregen] error: {exc}")
            finally:
                self._safe_after(0, lambda: self._pregen_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _export_remaining(self) -> None:
        if not self.book:
            return
        words = self._review_words()
        if not words:
            messagebox.showinfo("Nothing to export", "No words left to review.")
            return
        out = self._data_dir / "remaining_review.txt"
        out.write_text("\n".join(words), encoding="utf-8")
        messagebox.showinfo("Exported", f"{len(words)} words written to:\n{out}")

    def _apply_fixes(self) -> None:
        if not self.book:
            return
        if not self.fixes:
            messagebox.showinfo("No fixes", "The Fixes list is empty.")
            return

        parts = []
        for p in self.book.source_paths:
            if not p.exists():
                messagebox.showerror("Source not found", f"Cannot find:\n{p}")
                return
            parts.append(p.read_text(encoding="utf-8"))
        text = "\n\n".join(parts)

        count_total = 0
        for original, replacement in self.fixes.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
            if n:
                text = new_text
                count_total += n

        text, n_caps = re.subn(
            r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',
            lambda m: m.group(0).title(),
            text,
        )

        self.book.fixed_out.write_text(text, encoding="utf-8")
        messagebox.showinfo(
            "Done",
            f"Applied {len(self.fixes)} fix rules ({count_total} replacements).\n"
            f"Converted {n_caps} ALL-CAPS words to Title Case.\n\n"
            f"Saved to:\n{self.book.fixed_out}"
        )


# ── Standalone NLP extraction (lazy-imports spaCy) ─────────────────────────────────

def _extract_nouns_from_paths(source_paths: list) -> set[str]:
    """Run spaCy NER + PROPN pass over all *source_paths* and return a set of
    unique proper-noun strings, noise-filtered.
    Raises ImportError if spaCy or wordfreq are not installed.
    """
    import spacy                        # lazy — only loaded when button is clicked
    from wordfreq import top_n_list

    TOP_10K: frozenset[str] = frozenset(top_n_list("en", 10_000))
    WHITELIST: frozenset[str] = frozenset({
        "aaron","abel","abraham","adam","cain","eden","egypt",
        "elijah","ephraim","eve","gad","ham","isaac","israel",
        "jacob","james","jehovah","john","joseph","judah",
        "laban","lehi","levi","micah","michael","moses","noah",
        "peter","pharaoh","samuel","sarah","sarai","seth","simeon",
        "timothy","zion",
        "alma","ether","gideon","limhi","mormon","moroni","mulek",
        "mosiah","nephi","satan","sidon",
    })
    STOP_WORDS: set[str] = {
        "A","AN","AND","AS","AT","BE","BUT","BY","DO","DID","DOTH","EVEN",
        "FOR","FROM","HAD","HAS","HAVE","HATH","HE","HER","HIS","HOW","I",
        "IN","IS","IT","ITS","MAY","ME","MORE","MY","NAY","NO","NOT","NOW",
        "OF","OR","OUR","SHALL","SHE","SO","SOME","THAT","THE","THEE",
        "THEIR","THEN","THERE","THESE","THEY","THIS","THOSE","THOU","THUS",
        "THY","TO","UP","UPON","US","WAS","WE","WHEN","WHERE","WHICH","WHO",
        "WILL","WITH","YE","YEA","YET","YOU","YOUR",
        "BEHOLD","CHAPTER","CHRIST","GOD","GHOST","HOLY","LORD","VERSE",
        "CITY","DAYS","DAY","GREAT","LAND","MAN","MEN","NEW","PEOPLE","SON","TIME",
    }

    def _is_noise(t: str) -> bool:
        t = t.strip()
        if len(t) <= 1: return True
        if t.isupper() and len(t) > 4: return True
        if t.upper() in STOP_WORDS: return True
        if re.search(r"[^a-zA-Z\-']", t): return True
        if "-" not in t and t.lower() in TOP_10K and t.lower() not in WHITELIST:
            return True
        return False

    def _canonical(text: str) -> str:
        return " ".join(text.split()).title()

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 4_000_000

    PERSON = {"PERSON"}
    PLACE  = {"GPE", "LOC", "FAC"}
    ORG    = {"ORG", "NORP"}
    OTHER  = {"EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE"}

    found: set[str] = set()

    for path in source_paths:
        raw = path.read_text(encoding="utf-8")
        doc = nlp(raw)

        for ent in doc.ents:
            if ent.label_ not in (PERSON | PLACE | ORG | OTHER):
                continue
            for word in _canonical(ent.text).split():
                if not _is_noise(word):
                    found.add(word)

        for token in doc:
            if token.pos_ != "PROPN":
                continue
            t = token.text.strip()
            if not t[0].isupper() or t.isupper():
                continue
            if token.i == token.sent.start:
                continue
            word = _canonical(t)
            if not _is_noise(word) and word not in found:
                found.add(word)

    return found


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    books = discover_books()
    if not books:
        print("No source text files found in the current directory.")
        raise SystemExit(1)

    print(f"Discovered {len(books)} book source(s):")
    for b in books:
        print(f"  [{b.slug}]  {b.label}  ({len(b.source_paths)} file(s))")

    app = ProperNounAuditor(books)
    app.mainloop()


if __name__ == "__main__":
    main()
