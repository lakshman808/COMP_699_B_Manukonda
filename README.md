# Book Recommendation Platform

## Overview

The **Book Recommendation Platform** is a standalone Python desktop application built with **Tkinter** that helps readers quickly discover and organize books.  
It loads an enriched books dataset (CSV or Excel), lets users search by **title, author, language, genre, and difficulty**, and uses a **TF-IDF + cosine similarity** engine (via `pandas` and `scikit-learn`) to generate content-based recommendations.  
Users can save titles to **To-Read** and **Completed** lists, which are persisted in a local JSON file and visualized in a simple **matplotlib** dashboard showing reading progress.

---

## Key Features

- **Search & Filter**
  - Filter by title substring, author substring, language, genre, and difficulty.
  - Limit the maximum number of results shown in the main results grid.

- **Content-Based Recommendations**
  - **Personalize** by preferred authors/genres (profile-based query).
  - **Recommend by Title**: find books similar to a chosen title.
  - **Recommend by Keyword/Topic**: use free-text keywords (e.g., “wizard school”, “investing”) to locate similar books.

- **Reading List Management**
  - Add selected books from Results/Recommendations to **To-Read**.
  - Mark selected books as **Completed** (automatically removed from To-Read).
  - Remove books from To-Read or Completed via dedicated tabs.

- **Persistent Local State**
  - Stores **To-Read** and **Completed** lists in a JSON state file (`book_reco_state.json`).
  - Reading lists are automatically **restored on startup**.

- **Dashboard**
  - A **bar chart** (To-Read vs Completed) built with `matplotlib`.
  - Shows overall reading progress with the total number of tracked books.

- **Dataset Flexibility**
  - Reads from a configurable `DATASET_PATH` (CSV/XLSX).
  - Automatic column mapping where possible; manual mapping dialog when needed.

---

## Architecture at a Glance

- **`BookApp` (Tkinter GUI controller)**
  - Builds the main window, menu, sidebar, and tabbed interface.
  - Handles user actions such as **search**, **recommend**, and **list updates**.
  - Owns the `BookData` instance and the in-memory reading lists.

- **`BookData` (model / recommendation engine)**
  - Loads the dataset from CSV/XLSX and applies a **column map**.
  - Prepares internal columns and builds a **TF-IDF matrix** over titles, authors, genres, and descriptions.
  - Exposes `search`, `recommend_for_profile`, `recommend_like_title`, and `recommend_by_keyword`.

- **App State / Persistence**
  - `load_state()` and `save_state()` read/write `book_reco_state.json`.
  - Persists `to_read` and `completed` lists between sessions.

- **UI Helpers**
  - `ScrollableFrame` provides the scrollable left sidebar used for filters and recommendation controls.
  - `Treeview` widgets are used to render **Results/Recommendations**, **To-Read**, and **Completed** tables.

---

## Requirements

- **Python**
  - Python **3.9+** (tested with Python 3.10 / 3.11).
- **Standard Library**
  - `tkinter`, `json`, `os`, `re`, `pathlib`, `typing`.
- **Third-Party Libraries**
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

You can install dependencies with:

```bash
pip install pandas scikit-learn matplotlib



## How to Run the Book Recommendation Platform

Follow these steps to run the application:

1. **Open a terminal / command prompt**

2. **Change into the project folder**

   cd <your-project-folder>
3. Run the application

   python main.py



