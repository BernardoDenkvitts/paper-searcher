import logging
import logging
from pathlib import Path
from typing import List
import streamlit as st
from datetime import date, timedelta

from arxiv_searcher import Paper, search, ARXIV_CATEGORIES
from preprocessing import preprocess_and_vectorize, get_2d_coordinates
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Paper Explorer", page_icon="📚", layout="centered"
)


def load_css(file_path: str):
    """Load CSS from external file."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = Path(__file__).parent / "styles.css"
load_css(css_path)

# header link
st.markdown(
    '<div class="header-bar"><a href="https://ai4society.github.io/" target="_blank" class="header-link">AI4Society</a></div>',
    unsafe_allow_html=True,
)


@st.cache_data(ttl=timedelta(hours=1), max_entries=1000, show_spinner=False)
def search_papers(
    keywords: str, start_date: date, end_date: date, sort_opt: str, category_option: str
) -> List[Paper]:
    return search(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_opt,
        category=category_option,
    )


@st.cache_data(show_spinner=False)
def get_paper_clusters(papers: List[Paper]):
    """
    Perform clustering on a list of papers.
    """
    if not papers:
        return {}
        
    try:
        X = preprocess_and_vectorize(papers)
        # 5 clusters, but at least 2 if possible, and not more than len(papers)
        n_clusters = min(5, len(papers))
        if n_clusters < 2 and len(papers) > 1:
            n_clusters = 2
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_
        else:
            labels = np.zeros(len(papers), dtype=int)

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(papers[idx])
        
        # Sort clusters by label keys to ensure consistent order
        return dict(sorted(clusters.items()))
    except Exception as e:
        # Log error in console but return a fallback
        print(f"Clustering error: {e}")
        logging.error(f"Clustering error: {e}")
        return {0: papers}


ORDER_BY_OPTIONS = {
    "Relevance": "relevance",
    "Submitted Date": "submitted",
}

DEFAULT_KEYWORDS = "large language models, multi-agent systems"

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state["search_results"] = {"papers": [], "searched": False, "clusters": {}}

st.title("Paper Discovery", anchor=False)

st.markdown(
    '<div class="page-subtitle">Discover and explore research papers across the arXiv using keyword-based search</div>',
    unsafe_allow_html=True,
)

col_keywords, col_category = st.columns([2.5, 1])

with col_keywords:
    keywords = st.text_input(
        "Search Keywords or Phrases",
        placeholder=f"e.g., {DEFAULT_KEYWORDS} ...",
        help=(
            "Enter one or more keywords or phrases separated by commas. "
            "Results include only papers where all keywords are present. "
            f"If left empty, the default keywords - {DEFAULT_KEYWORDS} - will be used."
        ),
    )

with col_category:
    category_option = st.selectbox(
        "Category",
        ARXIV_CATEGORIES.keys(),
        index=0,
        help="Focus your search on a specific research field",
    )

col_start_date, col_end_date, col_order_by, col_search_bt = st.columns(
    [1, 1, 1.5, 1], vertical_alignment="center"
)

with col_start_date:
    # Default -> Starts from last year
    start_date = st.date_input(
        "Start Date",
        value=date(date.today().year - 1, date.today().month, date.today().day),
    )

with col_end_date:
    end_date = st.date_input("End Date", value="today")

with col_order_by:
    sort_option = st.selectbox(
        "Order By", ORDER_BY_OPTIONS.keys(), help="Order applied to ArXiv search"
    )

with col_search_bt:
    # Keeps button's vertical alignment
    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    search_button = st.button("Search")

# On click
if search_button:
    st.session_state["search_results"]["searched"] = False
    with st.spinner("🔎 Searching papers..."):
        try:
            if not keywords:
                keywords = DEFAULT_KEYWORDS

            papers = search_papers(
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                sort_opt=sort_option.lower(),
                category_option=category_option,
            )
            st.session_state["search_results"]["papers"] = papers

            if papers:
                st.session_state["search_results"]["clusters"] = get_paper_clusters(papers)

            st.session_state["search_results"]["searched"] = True
        except Exception as e:
            if str(e) != "Error fetching papers":
                st.error(f"⚠️ An error occurred")
                logging.error(f"An error occurred: {e}")
            else:
                st.warning("We couldn't fetch papers right now. Please try again in a few minutes.")

if st.session_state["search_results"]["searched"] and st.session_state["search_results"]["papers"]:
    col_results = st.columns(1)[0]
    with col_results:
        # Show number of results
        st.markdown(
            f"<div class='results-count'>{len(st.session_state['search_results']['papers'])} Papers Found</div>",
            unsafe_allow_html=True,
        )

    # Show each paper
    clusters = st.session_state["search_results"].get("clusters")
    
    is_first_paper = True
    # Flatten the clusters to match the dataframe indices
    ordered_papers = []
    all_labels = []
    
    for label, p_list in clusters.items():
        for p in p_list:
            ordered_papers.append(p)
            all_labels.append(str(label))
    
    show_viz = st.toggle("Show Cluster Visualization", value=True)
    if show_viz:  
        if len(ordered_papers) > 2:
            try:
                # TODO cache this
                X_ordered = preprocess_and_vectorize(ordered_papers)
                coords = get_2d_coordinates(X_ordered)

                if coords is not None and len(coords) == len(ordered_papers):
                    df_viz = pd.DataFrame({
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "Title": [p.title for p in ordered_papers],
                        "Cluster": all_labels,
                    })
                    
                    fig = px.scatter(
                        df_viz, 
                        x="x", y="y", 
                        color="Cluster", 
                        hover_data={"Title": True, "x": False, "y": False, "Cluster": True},
                        title="Paper Clusters"
                    )
                    
                    fig.update_layout(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Not enough variance to visualize clusters in 2D.")
            except Exception as e:
                st.warning(f"Visualization failed")
                logging.error(f"Visualization failed: {e}")
        else:
            st.info("Not enough papers to visualize clusters.")

    # Show papers by cluster
    tabs = st.tabs([f"Cluster {k}" for k in clusters.keys()])
    for i, tab in enumerate(tabs):
        with tab:
            cluster_id = list(clusters.keys())[i]
            for paper in clusters[cluster_id]:
                paper_date = paper.published.strftime("%d/%m/%Y")
                other_cats_html = "".join(f'<div class="paper-other-categories">{cat}</div>' for cat in paper.categories[1:])


                paper_html = f"""
                <div class="paper-card">
                    <div class="paper-title">{paper.title}</div>
                    <div class="paper-categories">
                        <div class="paper-main-category tooltip">{paper.main_category}<span class="tooltiptext">Primary Category</span></div>{other_cats_html}
                    </div>
                    <div class="paper-metadata">
                        <span class="metadata-item">📅 {paper_date}</span>
                        <span class="metadata-item">🔗 <a href="{paper.link}">View on arXiv</a></span>
                    </div>
                </div>
                """
                st.markdown(paper_html, unsafe_allow_html=True)

                with st.expander(f"View details"):
                    authors_str = ", ".join(paper.authors)
                    st.markdown(
                        f"""
                        <div class='paper-authors'><strong>Authors:</strong> {authors_str}</div>
                        [📄 <a href='{paper.pdf_link}'>PDF</a>]
                        <div class='paper-abstract'><strong>Abstract</strong><br>{paper.abstract}</div>
                    """,
                        unsafe_allow_html=True,
                    )

# Empty state
elif st.session_state["search_results"]["searched"] and not st.session_state["search_results"]["papers"]:
    st.info("🔍 No papers found. Try adjusting your keywords or search period.")
