# arXiv Paper Searcher

This project is a tool for paper searching on arXiv. It allows searching by keywords in specific categories, filtering by date and sorting, making it easier to collect papers for research.

## Core Logic (`arxiv_searcher.py`)

### Query Construction (`build_arxiv_query`)
The system builds a logical search that looks into both the **title** (`ti`) and the **abstract** (`abs`) for each provided keyword.

- **Field-Specific Search**: For a keyword like "transformer", it generates `(ti:transformer OR abs:transformer)`.
- **Exact Phrases**: Terms with more than one word (e.g., "large language models") are automatically quoted to ensure an exact phrase search.
- **Categories**: Automatically includes all subcategories for the chosen field (e.g., `cs.AI`, `cs.LG` for Computer Science).

### Search Function (`search`)
The `search` function coordinates the process:
1. **Validation**: Ensures no keyword overload (maximum 12) and validates the date range.
2. **Date Filtering**: Uses the submission date to restrict results to the requested period.
3. **Sorting**: Allows sorting by **Relevance** or **Submission Date**.
4. **Data Modeling**: Transforms results into a list of `Paper` objects for easy frontend use.

```python
@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str] 
    abstract: str
    published: datetime
    updated: datetime
    link: str
    pdf_link: str
    main_category: str
    categories: List[str]
```

## Web Interface (`streamlit_app.py`)

The project includes a Streamlit-based web interface for easier interaction.

- **Default Keywords**: If the search bar is left empty, the system automatically uses a default set of keywords: `large language models, multi-agent systems` (tailored for the Computer Science category).
- **Search Caching**: To improve performance and avoid redundant API calls, results are cached for 1 hour (or up to 1000 unique searches).

---

## How to use

### 1. Prerequisites
Install [uv](https://github.com/astral-sh/uv). The project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

### 2. Web Interface (Streamlit)
To use the visual interface:

```bash
uv run streamlit run streamlit_app.py
```

---

## Configuration and Logs
- **`config.json`**: Where you define the log folder.
- **Logs**: Users' searches are saved in log files

## Supported Categories
The project covers the main arXiv areas:
- **Computer Science** (cs.*)
- **Mathematics** (math.*)
- **Physics**
- **Statistics** (stat.*)
- **Electrical Engineering & Systems Science** (eess.*)
- **Quantitative Biology** (q-bio.*)
- **Quantitative Finance** (q-fin.*)
- **Economics** (econ.*)