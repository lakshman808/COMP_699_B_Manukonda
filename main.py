import json
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========= YOUR DATASET PATH =========
# IMPORTANT: point to the enriched file you just downloaded
DATASET_PATH = r"C:\Users\Lakshman\Desktop\Homeworks\Fall 2025\Lakshmi Narayana Manukonda\Professional Seminar Riabov\Final Code\books_enriched.csv"
# ====================================

APP_STATE_PATH = "book_reco_state.json"

def load_state() -> dict:
    if os.path.exists(APP_STATE_PATH):
        try:
            with open(APP_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    try:
        with open(APP_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save state:", e)

def normalize_list_cell(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if not s: return []
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

# ---------- Scrollable Frame for the left sidebar ----------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel (Windows/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

class BookData:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.feature_text: Optional[pd.Series] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.colmap = {
            "title": None,        # required
            "authors": None,      # required
            "genres": None,       # optional (maps to 'genre')
            "language": None,     # optional (maps to 'language_code' if present)
            "description": None,  # optional
            "rating": None,       # optional (maps to 'average_rating' if present)
            "difficulty": None    # optional (maps to 'difficulty')
        }

    def load_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        if df.empty:
            raise ValueError("Empty dataset.")
        self.df = df
        return df

    def set_column_map(self, mapping: Dict[str, Optional[str]]):
        for k in self.colmap:
            self.colmap[k] = mapping.get(k) or None

        for req in ["title", "authors"]:
            if self.colmap[req] is None or self.colmap[req] not in self.df.columns:
                raise ValueError(f"Column for '{req}' not set or not found.")

        df = self.df.copy()

        # Title + fallback
        df["_title_main"] = df[self.colmap["title"]].astype(str)
        if "original_title" in df.columns:
            mask_blank = df["_title_main"].isna() | (df["_title_main"].str.strip() == "")
            df.loc[mask_blank, "_title_main"] = df.loc[mask_blank, "original_title"].astype(str)

        df["_authors"] = df[self.colmap["authors"]].astype(str).fillna("")

        # Genres
        if self.colmap["genres"] and self.colmap["genres"] in df.columns:
            df["_genres_list"] = df[self.colmap["genres"]].apply(normalize_list_cell)
            df["_genres"] = df["_genres_list"].apply(lambda xs: " ".join([g.replace(" ", "_") for g in xs]))
        else:
            df["_genres_list"] = [[] for _ in range(len(df))]
            df["_genres"] = ""

        # Language (or fallback)
        if self.colmap["language"] and self.colmap["language"] in df.columns:
            df["_language"] = df[self.colmap["language"]].astype(str).fillna("")
        else:
            if "language_code" in df.columns:
                df["_language"] = df["language_code"].astype(str).fillna("")
            else:
                df["_language"] = ""

        # Description
        if self.colmap["description"] and self.colmap["description"] in df.columns:
            df["_description"] = df[self.colmap["description"]].astype(str).fillna("")
        else:
            df["_description"] = ""

        # Rating
        if self.colmap["rating"] and self.colmap["rating"] in df.columns:
            df["_rating"] = pd.to_numeric(df[self.colmap["rating"]], errors="coerce")
        else:
            df["_rating"] = pd.NA

        # Difficulty
        if self.colmap["difficulty"] and self.colmap["difficulty"] in df.columns:
            df["_difficulty"] = df[self.colmap["difficulty"]].astype(str)
        else:
            df["_difficulty"] = pd.NA

        # Text features for TF-IDF
        # Keep description in the mix so semantic similarity still works when strict match is empty
        self.feature_text = (
            df["_title_main"].str.lower().fillna("") + " " +
            df["_authors"].str.lower().str.replace(r"\s+", "_", regex=True).fillna("") + " " +
            df["_genres"].str.lower().fillna("") + " " +
            df["_description"].str.lower().fillna("")
        )

        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english",
            max_features=50000, ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.feature_text.fillna(""))

    def get_languages(self) -> List[str]:
        if self.df is None: return []
        langs = [x for x in self.df["_language"].dropna().astype(str).unique() if x]
        return sorted(langs) if langs else ["Any"]

    def get_genres(self) -> List[str]:
        if self.df is None: return []
        all_g = set()
        for xs in self.df["_genres_list"]:
            for g in xs:
                if g:
                    all_g.add(g)
        return sorted(all_g)

    def _apply_filters(self, df: pd.DataFrame, title_kw: str, author_kw: str,
                       language: str, genre: str, difficulty: str) -> pd.DataFrame:
        m = pd.Series(True, index=df.index)
        if title_kw:
            m &= df["_title_main"].str.contains(title_kw, case=False, na=False)
        if author_kw:
            m &= df["_authors"].str.contains(author_kw, case=False, na=False)
        if language and language != "Any":
            m &= df["_language"].str.lower().eq(language.lower())
        if genre and genre != "Any":
            m &= df["_genres_list"].apply(lambda xs: genre in xs if isinstance(xs, list) else False)
        if difficulty and difficulty != "Any":
            m &= df["_difficulty"].fillna("").str.lower().eq(difficulty.lower())
        return df.loc[m].copy()

    def search(self, title_kw: str, author_kw: str, language: str, genre: str, difficulty: str) -> pd.DataFrame:
        df = self._apply_filters(self.df, title_kw, author_kw, language, genre, difficulty)
        out = df.loc[:, ["_title_main", "_authors", "_language", "_rating", "_difficulty"]].copy()
        out.rename(columns={"_title_main": "title", "_authors": "authors", "_language": "language",
                            "_rating": "rating", "_difficulty": "difficulty"}, inplace=True)
        return out

    # ---- similarity helpers ----
    def _strict_desc_subset(self, base: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Return subset of base whose _description contains ANY of the tokens in query.
        """
        tokens = [t.strip() for t in re.split(r"[,\s]+", query) if t.strip()]
        if not tokens:
            return base.iloc[0:0]
        pat = "|".join(re.escape(t) for t in tokens)
        mask = base["_description"].str.contains(pat, case=False, na=False)
        return base.loc[mask].copy()

    def recommend_for_profile(self, profile_query: str, k: int = 20,
                              allow_language: Optional[str] = None,
                              genre_filter: Optional[str] = None,
                              difficulty: Optional[str] = None,
                              title_kw: str = "", author_kw: str = "") -> pd.DataFrame:
        if not profile_query.strip():
            return pd.DataFrame()
        base = self._apply_filters(self.df, title_kw, author_kw,
                                   allow_language or "Any",
                                   genre_filter or "Any",
                                   difficulty or "Any")
        if base.empty:
            return pd.DataFrame()

        vec = self.vectorizer.transform([profile_query.lower()])
        sims = cosine_similarity(self.tfidf_matrix[base.index], vec).ravel()
        base = base.copy()
        base["score"] = sims
        base = base.sort_values("score", ascending=False).head(k)
        return base[["_title_main", "_authors", "_language", "_rating", "_difficulty", "score"]].rename(
            columns={"_title_main": "title", "_authors": "authors", "_language": "language",
                     "_rating": "rating", "_difficulty": "difficulty"}
        )

    def recommend_like_title(self, like_title: str, k: int = 20,
                             allow_language: Optional[str] = None,
                             genre_filter: Optional[str] = None,
                             difficulty: Optional[str] = None,
                             title_kw: str = "", author_kw: str = "") -> pd.DataFrame:
        if not like_title.strip():
            return pd.DataFrame()

        base = self._apply_filters(self.df, title_kw, author_kw,
                                   allow_language or "Any",
                                   genre_filter or "Any",
                                   difficulty or "Any")
        if base.empty:
            return pd.DataFrame()

        # First try strict description match using the title text (helps steer to true topic)
        strict = self._strict_desc_subset(base, like_title)
        pool = strict if not strict.empty else base

        # Find a good anchor vector
        t_norm = like_title.strip().casefold()
        exact_idx = pool.index[pool["_title_main"].str.casefold() == t_norm]
        if len(exact_idx) == 0:
            contains_idx = self.df.index[self.df["_title_main"].str.contains(re.escape(like_title), case=False, na=False)]
            anchor_idx = contains_idx[0] if len(contains_idx) else None
        else:
            anchor_idx = exact_idx[0]

        anchor_vec = self.tfidf_matrix[anchor_idx] if anchor_idx is not None else self.vectorizer.transform([like_title.lower()])

        sims = cosine_similarity(self.tfidf_matrix[pool.index], anchor_vec).ravel()
        res = pool.copy()
        res["score"] = sims
        res = res[res["_title_main"].str.casefold() != t_norm]
        res = res.sort_values("score", ascending=False).head(k)
        return res[["_title_main", "_authors", "_language", "_rating", "_difficulty", "score"]].rename(
            columns={"_title_main": "title", "_authors": "authors", "_language": "language",
                     "_rating": "rating", "_difficulty": "difficulty"}
        )

    def recommend_by_keyword(self, keyword_query: str, k: int = 20,
                             allow_language: Optional[str] = None,
                             genre_filter: Optional[str] = None,
                             difficulty: Optional[str] = None,
                             title_kw: str = "", author_kw: str = "") -> pd.DataFrame:
        if not keyword_query.strip():
            return pd.DataFrame()
        base = self._apply_filters(self.df, title_kw, author_kw,
                                   allow_language or "Any",
                                   genre_filter or "Any",
                                   difficulty or "Any")
        if base.empty:
            return pd.DataFrame()

        # STRICT pass: require that description contains ANY of the tokens
        strict = self._strict_desc_subset(base, keyword_query)
        pool = strict if not strict.empty else base

        qvec = self.vectorizer.transform([keyword_query.lower()])
        sims = cosine_similarity(self.tfidf_matrix[pool.index], qvec).ravel()
        res = pool.copy()
        res["score"] = sims
        res = res.sort_values("score", ascending=False).head(k)
        return res[["_title_main", "_authors", "_language", "_rating", "_difficulty", "score"]].rename(
            columns={"_title_main": "title", "_authors": "authors", "_language": "language",
                     "_rating": "rating", "_difficulty": "difficulty"}
        )

class BookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Book Recommendation Platform")
        self.geometry("1180x760")

        self.state = load_state()
        self.data = BookData()

        self.to_read: List[str] = self.state.get("to_read", [])
        self.completed: List[str] = self.state.get("completed", [])

        # One shared IntVar used by all spinboxes (kept in sync)
        self.max_results_var = tk.IntVar(value=300)

        self._build_menu()
        self._build_layout()
        self._auto_load_dataset()

    def _build_menu(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Load dataset (CSV/XLSX)...", command=self.on_load_dataset)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)
        self.config(menu=menubar)

    def _build_layout(self):
        # Left: scrollable sidebar
        left_wrap = ScrollableFrame(self)
        left_wrap.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        left = left_wrap.inner

        # --- Search / Filters ---
        ttk.Label(left, text="Search / Filters", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,6))
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="Any")
        self.genre_var = tk.StringVar(value="Any")
        self.diff_var = tk.StringVar(value="Any")

        ttk.Label(left, text="Title contains:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.title_var, width=28).pack(anchor="w", pady=2)

        ttk.Label(left, text="Author contains:").pack(anchor="w", pady=(8,0))
        ttk.Entry(left, textvariable=self.author_var, width=28).pack(anchor="w", pady=2)

        ttk.Label(left, text="Language:").pack(anchor="w", pady=(8,0))
        self.lang_cb = ttk.Combobox(left, textvariable=self.lang_var, width=25, state="readonly", values=["Any"])
        self.lang_cb.pack(anchor="w", pady=2)

        ttk.Label(left, text="Genre:").pack(anchor="w", pady=(8,0))
        self.genre_cb = ttk.Combobox(left, textvariable=self.genre_var, width=25, state="readonly", values=["Any"])
        self.genre_cb.pack(anchor="w", pady=2)

        ttk.Label(left, text="Difficulty:").pack(anchor="w", pady=(8,0))
        self.diff_cb = ttk.Combobox(left, textvariable=self.diff_var, width=25, state="readonly",
                                    values=["Any","Beginner","Intermediate","Advanced"])
        self.diff_cb.pack(anchor="w", pady=2)

        # Max results (Search/Filters section)
        ttk.Label(left, text="Max results to show:").pack(anchor="w", pady=(10,0))
        ttk.Spinbox(left, from_=10, to=2000, increment=10, textvariable=self.max_results_var, width=8).pack(anchor="w", pady=2)

        ttk.Button(left, text="Search", command=self.on_search).pack(anchor="w", pady=(10,4))
        ttk.Button(left, text="Clear Filters", command=self.on_clear_filters).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=12)

        # --- Personalize ---
        ttk.Label(left, text="Personalize", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,6))
        self.pref_author = tk.StringVar()
        self.pref_genre = tk.StringVar()
        ttk.Label(left, text="Preferred author(s):").pack(anchor="w")
        ttk.Entry(left, textvariable=self.pref_author, width=28).pack(anchor="w", pady=2)
        ttk.Label(left, text="Preferred genre(s):").pack(anchor="w")
        ttk.Entry(left, textvariable=self.pref_genre, width=28).pack(anchor="w", pady=2)

        # Max results (Personalize section)
        ttk.Label(left, text="Max results to show:").pack(anchor="w", pady=(8,0))
        ttk.Spinbox(left, from_=10, to=2000, increment=10, textvariable=self.max_results_var, width=8).pack(anchor="w", pady=2)

        ttk.Button(left, text="Get Recommendations", command=self.on_recommend).pack(anchor="w", pady=8)

        # --- Recommend by Title ---
        ttk.Label(left, text="I like this title:").pack(anchor="w", pady=(12,0))
        self.like_title_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.like_title_var, width=28).pack(anchor="w", pady=2)

        # Max results (Title section)
        ttk.Label(left, text="Max results to show:").pack(anchor="w", pady=(4,0))
        ttk.Spinbox(left, from_=10, to=2000, increment=10, textvariable=self.max_results_var, width=8).pack(anchor="w", pady=2)

        ttk.Button(left, text="Recommend by Title", command=self.on_recommend_by_title).pack(anchor="w", pady=4)

        # --- Recommend by Keyword ---
        ttk.Label(left, text="Keyword / topic:").pack(anchor="w", pady=(12,0))
        self.keyword_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.keyword_var, width=28).pack(anchor="w", pady=2)

        # Max results (Keyword section)
        ttk.Label(left, text="Max results to show:").pack(anchor="w", pady=(4,0))
        ttk.Spinbox(left, from_=10, to=2000, increment=10, textvariable=self.max_results_var, width=8).pack(anchor="w", pady=2)

        ttk.Button(left, text="Recommend by Keyword", command=self.on_recommend_by_keyword).pack(anchor="w", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=12)

        ttk.Button(left, text="Add selected to To-Read", command=self.add_selected_to_read).pack(anchor="w", pady=4)
        ttk.Button(left, text="Mark selected as Completed", command=self.add_selected_completed).pack(anchor="w", pady=2)

        # Right: notebook with scrollable results + other tabs
        right = ttk.Notebook(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_results = ttk.Frame(right)
        self.tab_toread = ttk.Frame(right)
        self.tab_completed = ttk.Frame(right)
        self.tab_dashboard = ttk.Frame(right)

        right.add(self.tab_results, text="Results / Recommendations")
        right.add(self.tab_toread, text="To-Read")
        right.add(self.tab_completed, text="Completed")
        right.add(self.tab_dashboard, text="Dashboard")

        self._build_results_tab(self.tab_results)
        self._build_to_read_tab()
        self._build_completed_tab()
        self._build_dashboard_tab()

    def _build_results_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        cols = ("title","authors","language","rating","difficulty","score")
        self.tree = ttk.Treeview(container, columns=cols, show="headings")

        # Scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for c in cols:
            self.tree.heading(c, text=c.title())
            w = 260 if c in ("title","authors") else 110 if c == "language" else 90
            self.tree.column(c, width=w, anchor="w")

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        bottom = ttk.Frame(parent)
        bottom.pack(fill="x", pady=6)
        ttk.Button(bottom, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT)
        ttk.Label(parent, text="Tip: Click a row, then use the buttons on the left to add to lists.").pack(anchor="w", pady=4)

    def clear_results(self):
        for tr in self.tree.get_children():
            self.tree.delete(tr)

    def _build_to_read_tab(self):
        self.tree_tr = ttk.Treeview(self.tab_toread, columns=("title","authors"), show="headings")
        for c in ("title","authors"):
            self.tree_tr.heading(c, text=c.title())
            self.tree_tr.column(c, width=420, anchor="w")
        self.tree_tr.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(self.tab_toread)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Remove from To-Read", command=self.remove_from_to_read).pack(side=tk.LEFT, padx=4)

    def _build_completed_tab(self):
        self.tree_c = ttk.Treeview(self.tab_completed, columns=("title","authors"), show="headings")
        for c in ("title","authors"):
            self.tree_c.heading(c, text=c.title())
            self.tree_c.column(c, width=420, anchor="w")
        self.tree_c.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(self.tab_completed)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Remove from Completed", command=self.remove_from_completed).pack(side=tk.LEFT, padx=4)

    def _build_dashboard_tab(self):
        self.fig = Figure(figsize=(5,3), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_dashboard)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(self.tab_dashboard, text="Refresh Dashboard", command=self.refresh_dashboard).pack(pady=6)

    # ---------- Dataset loading ----------
    def _auto_load_dataset(self):
        if not os.path.exists(DATASET_PATH):
            messagebox.showwarning("Dataset not found",
                                   f"Could not find dataset at:\n{DATASET_PATH}\n\nUse File → Load dataset to pick your CSV/XLSX.")
            return
        try:
            df = self.data.load_file(DATASET_PATH)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        mapping = {
            "title": "title" if "title" in df.columns else ("original_title" if "original_title" in df.columns else None),
            "authors": "authors" if "authors" in df.columns else None,
            "genres": "genre" if "genre" in df.columns else None,
            "language": "language_code" if "language_code" in df.columns else None,
            "description": "description" if "description" in df.columns else None,
            "rating": "average_rating" if "average_rating" in df.columns else None,
            "difficulty": "difficulty" if "difficulty" in df.columns else None
        }
        try:
            self.data.set_column_map(mapping)
        except Exception as e:
            messagebox.showerror("Mapping error", f"{e}\n\nOpen File → Load dataset and map manually.")
            return

        self._populate_filters()
        messagebox.showinfo("Dataset ready",
                            f"Loaded and indexed:\n{DATASET_PATH}\n\nYou can now Search or Get Recommendations.")

    def on_load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select books dataset (CSV or Excel)",
            filetypes=[("CSV","*.csv"),("Excel","*.xlsx *.xls"),("All files","*.*")]
        )
        if not path:
            return
        try:
            df = self.data.load_file(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        mapping = {
            "title": "title" if "title" in df.columns else ("original_title" if "original_title" in df.columns else None),
            "authors": "authors" if "authors" in df.columns else None,
            "genres": "genre" if "genre" in df.columns else None,
            "language": "language_code" if "language_code" in df.columns else None,
            "description": "description" if "description" in df.columns else None,
            "rating": "average_rating" if "average_rating" in df.columns else None,
            "difficulty": "difficulty" if "difficulty" in df.columns else None
        }
        try:
            self.data.set_column_map(mapping)
        except Exception:
            self.ask_column_map(df)
            return

        self._populate_filters()
        messagebox.showinfo("Dataset ready", "Dataset loaded and indexed. You can now search or get recommendations.")

    def _populate_filters(self):
        langs = ["Any"] + self.data.get_languages()
        self.lang_cb.configure(values=langs)
        self.lang_var.set("Any")

        genres = ["Any"] + (self.data.get_genres() or [])
        self.genre_cb.configure(values=genres if genres else ["Any"])
        self.genre_var.set("Any")

    def ask_column_map(self, df: pd.DataFrame):
        top = tk.Toplevel(self)
        top.title("Map Columns")
        top.grab_set()
        cols = ["(none)"] + list(df.columns)

        mapping_vars = {}
        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Map your file's columns (only Title & Authors are required)").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))

        fields = [
            ("title","Title *"),
            ("authors","Authors *"),
            ("genres","Genres (optional)"),
            ("language","Language (optional)"),
            ("description","Description (optional)"),
            ("rating","Rating (optional)"),
            ("difficulty","Difficulty (optional)")
        ]
        for i,(key,label) in enumerate(fields, start=1):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="e", padx=(0,8), pady=3)
            var = tk.StringVar(value="(none)")
            cb = ttk.Combobox(frm, textvariable=var, values=cols, state="readonly", width=40)
            if key == "genres":
                guess = [c for c in df.columns if c.lower() == "genre"]
                if guess: var.set(guess[0])
            if key == "difficulty":
                guess = [c for c in df.columns if c.lower() == "difficulty"]
                if guess: var.set(guess[0])
            if key == "description":
                guess = [c for c in df.columns if c.lower() == "description"]
                if guess: var.set(guess[0])
            cb.grid(row=i, column=1, sticky="w", pady=3)
            mapping_vars[key] = var

        btns = ttk.Frame(frm)
        btns.grid(row=len(fields)+1, column=0, columnspan=2, pady=(10,0))
        ttk.Button(btns, text="Confirm", command=lambda: self._confirm_map(top, mapping_vars)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=5)

    def _confirm_map(self, dialog: tk.Toplevel, mapping_vars: Dict[str, tk.StringVar]):
        mapping = {k:(v.get() if v.get()!="(none)" else None) for k,v in mapping_vars.items()}
        try:
            self.data.set_column_map(mapping)
        except Exception as e:
            messagebox.showerror("Mapping error", str(e))
            return

        self._populate_filters()
        dialog.destroy()
        messagebox.showinfo("Dataset ready", "Dataset loaded and indexed. You can now search or get recommendations.")

    # ---------- Actions ----------
    def on_search(self):
        if self.data.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first (File → Load dataset).")
            return
        df = self.data.search(
            self.title_var.get().strip(),
            self.author_var.get().strip(),
            self.lang_var.get(),
            self.genre_var.get(),
            self.diff_var.get()
        )
        df = df.head(max(1, int(self.max_results_var.get())))
        self._fill_tree(df)

    def on_clear_filters(self):
        self.title_var.set("")
        self.author_var.set("")
        self.lang_var.set("Any")
        self.genre_var.set("Any")
        self.diff_var.set("Any")
        self.clear_results()

    def on_recommend(self):
        if self.data.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first (File → Load dataset).")
            return
        profile = " ".join([
            self.pref_author.get().strip(),
            self.pref_genre.get().strip()
        ])
        if not profile.strip():
            messagebox.showinfo("Enter preferences", "Provide preferred author(s) and/or genre(s).")
            return
        recs = self.data.recommend_for_profile(
            profile_query=profile,
            k=max(1, int(self.max_results_var.get())),
            allow_language=self.lang_var.get(),
            genre_filter=self.genre_var.get(),
            difficulty=self.diff_var.get(),
            title_kw=self.title_var.get().strip(),
            author_kw=self.author_var.get().strip()
        )
        self._fill_tree(recs)

    def on_recommend_by_title(self):
        if self.data.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first (File → Load dataset).")
            return
        like_title = self.like_title_var.get().strip()
        if not like_title:
            messagebox.showinfo("Type a title", "Enter a title you like (e.g., Harry Potter).")
            return
        recs = self.data.recommend_like_title(
            like_title=like_title,
            k=max(1, int(self.max_results_var.get())),
            allow_language=self.lang_var.get(),
            genre_filter=self.genre_var.get(),
            difficulty=self.diff_var.get(),
            title_kw=self.title_var.get().strip(),
            author_kw=self.author_var.get().strip()
        )
        if recs is None or recs.empty:
            messagebox.showinfo("No results", "No similar titles found with current filters. Try relaxing filters (Genre/Language/Difficulty = Any).")
            return
        self._fill_tree(recs)

    def on_recommend_by_keyword(self):
        if self.data.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first (File → Load dataset).")
            return
        q = self.keyword_var.get().strip()
        if not q:
            messagebox.showinfo("Type a keyword", "Enter a keyword or topic (e.g., money, investing, magic, wizard school).")
            return
        recs = self.data.recommend_by_keyword(
            keyword_query=q,
            k=max(1, int(self.max_results_var.get())),
            allow_language=self.lang_var.get(),
            genre_filter=self.genre_var.get(),
            difficulty=self.diff_var.get(),
            title_kw=self.title_var.get().strip(),
            author_kw=self.author_var.get().strip()
        )
        if recs is None or recs.empty:
            messagebox.showinfo("No results", "No matches for that keyword with current filters. Try relaxing Genre/Language/Difficulty or clear Title/Author contains.")
            return
        self._fill_tree(recs)

    def _fill_tree(self, df: pd.DataFrame):
        self.clear_results()
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            row = (
                r.get("title",""),
                r.get("authors",""),
                r.get("language",""),
                "" if pd.isna(r.get("rating")) else r.get("rating"),
                "" if pd.isna(r.get("difficulty")) else r.get("difficulty"),
                "" if "score" not in r else f"{r['score']:.3f}"
            )
            self.tree.insert("", tk.END, values=row)

        self.refresh_lists()
        self.refresh_dashboard()

    def get_selected_titles_from_results(self) -> List[str]:
        sel = []
        for item in self.tree.selection():
            vals = self.tree.item(item, "values")
            if vals:
                sel.append(vals[0])
        return sel

    def add_selected_to_read(self):
        titles = self.get_selected_titles_from_results()
        if not titles:
            return
        for t in titles:
            if t not in self.to_read and t not in self.completed:
                self.to_read.append(t)
        self.persist_lists()
        self.refresh_lists()

    def add_selected_completed(self):
        titles = self.get_selected_titles_from_results()
        if not titles:
            return
        for t in titles:
            if t in self.to_read:
                self.to_read.remove(t)
            if t not in self.completed:
                self.completed.append(t)
        self.persist_lists()
        self.refresh_lists()

    def remove_from_to_read(self):
        items = self.tree_tr.selection()
        for it in items:
            vals = self.tree_tr.item(it, "values")
            if vals and vals[0] in self.to_read:
                self.to_read.remove(vals[0])
        self.persist_lists()
        self.refresh_lists()

    def remove_from_completed(self):
        items = self.tree_c.selection()
        for it in items:
            vals = self.tree_c.item(it, "values")
            if vals and vals[0] in self.completed:
                self.completed.remove(vals[0])
        self.persist_lists()
        self.refresh_lists()

    def persist_lists(self):
        self.state["to_read"] = self.to_read
        self.state["completed"] = self.completed
        save_state(self.state)

    def refresh_lists(self):
        for tr in self.tree_tr.get_children():
            self.tree_tr.delete(tr)
        for t in self.to_read:
            a = ""
            if self.data.df is not None:
                m = self.data.df["_title_main"] == t
                if m.any():
                    a = self.data.df.loc[m, "_authors"].iloc[0]
            self.tree_tr.insert("", tk.END, values=(t, a))

        for tr in self.tree_c.get_children():
            self.tree_c.delete(tr)
        for t in self.completed:
            a = ""
            if self.data.df is not None:
                m = self.data.df["_title_main"] == t
                if m.any():
                    a = self.data.df.loc[m, "_authors"].iloc[0]
            self.tree_c.insert("", tk.END, values=(t, a))

    def refresh_dashboard(self):
        self.ax1.clear()
        total = len(self.to_read) + len(self.completed)
        values = [len(self.to_read), len(self.completed)]
        if sum(values) == 0:
            self.ax1.text(0.5, 0.5, "No data yet.\nAdd books to lists.",
                          ha="center", va="center", fontsize=12)
        else:
            self.ax1.bar(["To-Read","Completed"], values)
            self.ax1.set_title(f"Reading Progress (Total tracked: {total})")
            self.ax1.set_ylabel("Count")
        self.canvas.draw()

if __name__ == "__main__":
    app = BookApp()
    app.mainloop()

