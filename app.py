import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# ---------- CONFIG ----------

DATA_PATH = Path("EV Data Explorer 2025.xlsx")  # put the file in the same folder
SHEET_NAME = "EV sales countries"

BLUE = "#1f4e79"
LIGHT_BLUE = "#a7c4e0"
LIGHT_GREY = "#e5e5e5"
DARK_GREY = "#444444"


# ---------- DATA LOADING & CLEANING ----------

@st.cache_data
def load_ev_sales(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load EV sales from IEA 'EV sales countries' sheet and reshape to long format:
    columns: region_country, Year, EV_Sales
    """
    # Row 7 (0-based) is the header row with: region_country, 2010, 2011, ...
    df = pd.read_excel(path, sheet_name=sheet_name, header=7)

    # Keep region_country + year columns
    year_cols = []
    for c in df.columns:
        if isinstance(c, (int, float)):
            year_cols.append(c)
        elif isinstance(c, str) and c.strip().isdigit():
            year_cols.append(c)

    if "region_country" not in df.columns:
        raise ValueError("Expected a 'region_country' column in the sheet header row.")

    df_wide = df[["region_country"] + year_cols]

    # Melt to long format
    df_long = df_wide.melt(
        id_vars=["region_country"],
        value_vars=year_cols,
        var_name="Year",
        value_name="EV_Sales",
    )

    # Drop rows with no data
    df_long = df_long.dropna(subset=["EV_Sales"])
    df_long = df_long[df_long["region_country"].notna()]

    # Fix types
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["EV_Sales"] = df_long["EV_Sales"].astype(float)

    # Simple clean: trim whitespace
    df_long["region_country"] = df_long["region_country"].astype(str).str.strip()

    return df_long


# ---------- KPI HELPERS ----------

def compute_global_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate global EV sales by year."""
    return (
        df_long.groupby("Year", as_index=False)["EV_Sales"]
        .sum()
        .sort_values("Year")
    )


def compute_kpis(df_long: pd.DataFrame, year_selected: int):
    global_by_year = compute_global_summary(df_long)

    # Total sales selected year
    total_selected = (
        global_by_year.loc[global_by_year["Year"] == year_selected, "EV_Sales"]
        .sum()
    )

    # YoY growth (vs previous year) – if previous year exists
    prev_year = year_selected - 1
    prev_sales = (
        global_by_year.loc[global_by_year["Year"] == prev_year, "EV_Sales"]
        .sum()
    )

    yoy_growth = None
    if prev_sales > 0:
        yoy_growth = (total_selected - prev_sales) / prev_sales * 100.0

    # Top country for selected year
    df_year = df_long[df_long["Year"] == year_selected]
    top_row = df_year.sort_values("EV_Sales", ascending=False).head(1)

    if not top_row.empty:
        top_country = top_row["region_country"].iloc[0]
        top_sales = top_row["EV_Sales"].iloc[0]
    else:
        top_country, top_sales = None, None

    num_countries = df_year["region_country"].nunique()

    return {
        "total_selected": total_selected,
        "yoy_growth": yoy_growth,
        "top_country": top_country,
        "top_sales": top_sales,
        "num_countries": num_countries,
    }


# ---------- PLOTTING HELPERS ----------

def make_global_line(df_long: pd.DataFrame):
    global_by_year = compute_global_summary(df_long)

    fig = px.line(
        global_by_year,
        x="Year",
        y="EV_Sales",
        markers=True,
    )

    fig.update_traces(line_color=BLUE, marker=dict(color=BLUE, size=6))

    latest_year = int(global_by_year["Year"].max())
    latest_value = float(
        global_by_year.loc[global_by_year["Year"] == latest_year, "EV_Sales"].iloc[0]
    )

    fig.add_annotation(
        x=latest_year,
        y=latest_value,
        text=f"Record year: {latest_year}",
        showarrow=True,
        arrowhead=2,
        ax=30,
        ay=-40,
        font=dict(color=DARK_GREY),
        bgcolor="white",
    )

    fig.update_layout(
        title=None,
        xaxis_title="Year",
        yaxis_title="EV sales (units)",
        showlegend=False,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=False, showline=False),
    )

    return fig


def make_top_countries_bar(df_long: pd.DataFrame, year_selected: int, top_n: int = 10):
    df_year = df_long[df_long["Year"] == year_selected].copy()
    df_year = df_year.sort_values("EV_Sales", ascending=False).head(top_n)

    # Highlight the top country and dim others a bit
    if not df_year.empty:
        top_country = df_year.iloc[0]["region_country"]
        df_year["color"] = df_year["region_country"].apply(
            lambda c: BLUE if c == top_country else LIGHT_BLUE
        )
    else:
        df_year["color"] = BLUE

    fig = px.bar(
        df_year,
        x="EV_Sales",
        y="region_country",
        orientation="h",
    )

    fig.update_traces(marker_color=df_year["color"])

    fig.update_layout(
        title=None,
        xaxis_title="EV sales (units)",
        yaxis_title="",
        margin=dict(l=120, r=40, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=False, showline=False),
    )

    return fig


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Global EV Sales Dashboard",
        page_icon="🚗",
        layout="wide",
    )

    st.title("Global EV Sales Dashboard")
    st.caption(
        "IEA EV Data Explorer 2025 – simple, clean view of historical EV sales by country."
    )

    if not DATA_PATH.exists():
        st.error(
            f"Excel file not found at `{DATA_PATH}`. "
            "Put `EV Data Explorer 2025.xlsx` in the same folder as this app."
        )
        return

    df_long = load_ev_sales(DATA_PATH, SHEET_NAME)

    min_year = int(df_long["Year"].min())
    max_year = int(df_long["Year"].max())

    # Sidebar controls
    st.sidebar.header("Filters")
    year_selected = st.sidebar.slider(
        "Select year",
        min_value=min_year,
        max_value=max_year,
        value=max_year,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Data source: IEA Global EV Data Explorer (2025)")

    # KPIs
    kpi = compute_kpis(df_long, year_selected)
    total_selected = kpi["total_selected"]
    yoy_growth = kpi["yoy_growth"]
    top_country = kpi["top_country"]
    top_sales = kpi["top_sales"]
    num_countries = kpi["num_countries"]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(
            label=f"Total EV sales in {year_selected}",
            value=f"{int(total_selected):,}",
        )

    with kpi_col2:
        if yoy_growth is not None:
            st.metric(
                label=f"YoY growth vs {year_selected - 1}",
                value=f"{yoy_growth:.1f} %",
            )
        else:
            st.metric(
                label=f"YoY growth vs {year_selected - 1}",
                value="n/a",
            )

    with kpi_col3:
        if top_country is not None:
            st.metric(
                label=f"Top EV market in {year_selected}",
                value=top_country,
                delta=f"{int(top_sales):,} units",
            )
        else:
            st.metric(
                label=f"Top EV market in {year_selected}",
                value="n/a",
            )

    with kpi_col4:
        st.metric(
            label=f"Countries / regions with sales data ({year_selected})",
            value=num_countries,
        )

    st.markdown("---")

    # Layout: main line chart on top, bar chart below
    line_col, = st.columns(1)
    with line_col:
        st.subheader("Global EV sales over time")
        st.caption(
            "EV sales have grown substantially over the last decade. "
            "Use this as the backbone for your story."
        )
        fig_line = make_global_line(df_long)
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("")

    bar_col, table_col = st.columns([2, 1])

    with bar_col:
        st.subheader(f"Top EV markets in {year_selected}")
        st.caption("A few countries drive most EV demand.")
        fig_bar = make_top_countries_bar(df_long, year_selected, top_n=10)
        st.plotly_chart(fig_bar, use_container_width=True)

    with table_col:
        st.subheader("Raw data (preview)")
        df_year_preview = (
            df_long[df_long["Year"] == year_selected]
            .sort_values("EV_Sales", ascending=False)
            .head(15)
        )
        st.dataframe(df_year_preview.reset_index(drop=True))

    st.markdown("---")
    st.caption(
        "Design choices follow Knaflic's principles: minimal chart junk, limited color palette, "
        "and clear, text-based explanations of what the viewer should notice."
    )


if __name__ == "__main__":
    main()
