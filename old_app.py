# === Imports ===
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   MultiLabelBinarizer, OneHotEncoder)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import networkx as nx
import seaborn as sns
from collections import Counter
from itertools import combinations

# === Load and Clean Data ===
df = pd.read_csv("merged_with_tags.csv")
df = df.dropna(subset=["budget", "revenue", "release_year", "title_y", "vote_count", 'genres_y'])
df = df.drop_duplicates(subset=['title_y'])
df["release_year"] = df["release_year"].astype(int)

# === Feature Engineering ===
df["profit"] = df["revenue"] - df["budget"]
df["profit_margin"] = (df["profit"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
df["roi"] = df["profit_margin"] * 100
df["primary_genre"] = df["genres_y"].str.split("-").str[0]

# === Genre Hierarchy Function ===
def prepare_genre_data(df):
    """Splits genre into primary and subgenres for hierarchy analysis."""
    genre_df = df.assign(genres=df['genres_y'].str.split('|')).explode('genres')
    genre_counts = genre_df.groupby('genres').size().reset_index(name='count')
    genre_hierarchy = []
    for _, row in genre_df.iterrows():
        genres = row['genres_y'].split('-')
        if len(genres) > 1:
            primary = genres[0]
            for sub in genres[1:]:
                genre_hierarchy.append({
                    'primary': primary,
                    'sub': sub,
                    'title': row['title_y'],
                    'budget': row['budget'],
                    'revenue': row['revenue'],
                    'profit': row['profit'],
                    'roi': row['roi'],
                    'vote_average': row['vote_average']
                })
    hierarchy_df = pd.DataFrame(genre_hierarchy)
    return genre_counts, hierarchy_df

genre_counts, hierarchy_df = prepare_genre_data(df)
numeric_features = ["budget", "revenue", "vote_average", "vote_count", "runtime", "profit", "roi"]

# === Modeling Preparation ===
df['genres_list'] = df['genres_y'].str.split('-')
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres_list']), columns=mlb.classes_, index=df.index)

df['success'] = (df['vote_average'] > 6.5) & (df['revenue'] > df['budget'] * 2)
df['budget_log'] = np.log1p(df['budget'])
df['revenue_log'] = np.log1p(df['revenue'])
df['is_english'] = (df['original_language'] == 'en').astype(int)

features = pd.concat([
    df[['budget_log', 'runtime', 'is_english', 'release_year']],
    genre_encoded
], axis=1)

success_target = df['success']
rating_target = df['vote_average']
revenue_target = df['revenue_log']

#splitting test and training
X_train, X_test, y_success_train, y_success_test, y_rating_train, y_rating_test, y_revenue_train, y_revenue_test = train_test_split(
    features, success_target, rating_target, revenue_target, test_size=0.2, random_state=42
)

# === Train Models ===
success_model = RandomForestClassifier(n_estimators=100, random_state=42)
success_model.fit(X_train, y_success_train)

rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model.fit(X_train, y_rating_train)

revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
revenue_model.fit(X_train, y_revenue_train)

all_genres = sorted(list(set([genre for sublist in df['genres_list'].dropna() for genre in sublist])))


# === Helper Function to Make Scatterplot Cleaner ===
#This function was implemented with the help of AI agent
def prettify_figure(fig, x_axis=None, y_axis=None, title=None):
    fig.update_layout(
        font=dict(family='Sans-serif', size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            title=x_axis.replace("_", " ").title() if x_axis else '',
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False
        ),
        yaxis=dict(
            title=y_axis.replace("_", " ").title() if y_axis else '',
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False
        ),
        hovermode="closest",
        title=dict(
            text=f"<b>{title}</b>" if title else '',
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        )
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey'),
            opacity=0.8,
            sizemode='area'
        )
    )

    return fig


# === Helper Function to Make Sensitivity Plot Cleaner ===
#This function was implemented with the help of AI agent
def prettify_sensitivity_figure(fig, title=None):
    fig.update_layout(
        font=dict(family='Sans-serif', size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=80, b=60),  
        title=dict(
            text=f"<b>{title}</b>" if title else '',
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor='blue'  # Match trace color
        ),
        yaxis2=dict(
            showgrid=False,  
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor='green'  # Match trace color
        ),
        yaxis3=dict(
            showgrid=False,
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor='red'  # Match trace color
        )
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate="%{y:.2f}<extra></extra>"  # Clean hover format
    )
    
    # Add axis title styling
    fig.update_yaxes(title_standoff=15)
    fig.update_xaxes(title_standoff=15)
    
    return fig







# === Dash App Setup ===
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Film Predict-R"

# === Layout Placeholder ===
#This layout was implemented with the help of AI agent to ensure a correct and cohesive coloring scheme 
app.layout = html.Div([ 
    dcc.Store(id="current-filter", data={"genres": None, "scatter_ids": None, "zoom_ids": None}),
    dcc.Store(id="prediction-store", data=None),

    # ==== TOP SECTION ====
    html.Div([
        # LEFT (20%)
        html.Div([
            #dropdown to chose points color
            html.Label("Color by:"),
            dcc.Dropdown(
                    id="color-by",
                    options=[
                        {"label": "Vote Average", "value": "vote_average"},
                        {"label": "ROI", "value": "roi"},
                        {"label": "Runtime", "value": "runtime"},
                        {"label": "Profit", "value": "profit"}
                    ],
                value="roi",
                style={"margin-bottom": "10px"}
            ),
            #icicle graph, check the icicle callout for implementation
            dcc.Graph(id="genre-icicle", style={"height": "40vh", "width": "100%"})
        ], style={"width": "20%", "padding": "10px", "display": "inline-block", "verticalAlign": "top"}),

        # RIGHT (80%)
        html.Div([
            #flexible html.div to better place the different dropodown and slider
            html.Div([            
                # X-axis dropdown 
                html.Div([
                    html.Label("X-axis:", style={"margin-bottom": "5px"}),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[
                            {"label": "Budget", "value": "budget"},
                            {"label": "Revenue", "value": "revenue"},
                            {"label": "Profit", "value": "profit"},
                            {"label": "ROI", "value": "roi"},
                            {"label": "Vote Average", "value": "vote_average"},
                            {"label": "Runtime", "value": "runtime"}
                        ],
                        value="budget",
                        style={"width": "200px", "marginRight": "20px"}
                    )
                ], style={"display": "inline-block", "marginRight": "20px"}),
                
                # Y-axis dropdown 
                html.Div([
                    html.Label("Y-axis:", style={"margin-bottom": "5px"}),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[
                            {"label": "Revenue", "value": "revenue"},
                            {"label": "Budget", "value": "budget"},
                            {"label": "Profit", "value": "profit"},
                            {"label": "ROI", "value": "roi"},
                            {"label": "Vote Average", "value": "vote_average"},
                            {"label": "Runtime", "value": "runtime"}
                        ],
                        value="revenue",
                        style={"width": "200px"}
                    )
                ], style={"display": "inline-block"}),
                
                #Slider for choosing the year to filter the data
                html.Div([
                    html.Label("Year Range:"),
                    dcc.RangeSlider(
                        id="shared-year-slider",
                        min=2000, max=2020, step=1, value=[2000, 2020],
                        marks={str(y): str(y) for y in range(2000, 2021, 5)},
                        tooltip={"always_visible": True},
                        allowCross=False,
                    ),
                ], style={"display": "inline-block", "flex": 1, "minWidth": "300px"})
                
            ], style={"margin-bottom": "15px", "display": "flex", "alignItems": "flex-start"}),
            #Scatterplot, check the icicle callout for implementation
            dcc.Graph(id="scatter-plot", style={"height": "47vh", "width": "100%"})
        ], style={"width": "80%", "padding": "10px", "display": "inline-block", "verticalAlign": "top"})
    ], style={"display": "flex", "height": "55vh"}),

    # ==== BOTTOM SECTION ====
    html.Div([
        # LEFT SIDE (40%)
        html.Div([
            html.H3("Movie Success Predictor"),
            html.Div([
                # Inputs (50% width of left side)
                html.Div([
                    html.Label("Budget (in millions)"),
                    dcc.Input(id='budget-input', type='number', value=50, min=0.1, step=0.01,
                              style={'width': '100%', 'marginBottom': '10px'}),

                    html.Label("Runtime (minutes)"),
                    dcc.Input(id='runtime-input', type='number', value=120, min=60, max=240, step=1,
                              style={'width': '100%', 'marginBottom': '10px'}),

                    html.Label("Original Language"),
                    dcc.Dropdown(
                        id='language-input',
                        options=[{'label': 'English', 'value': 1}, {'label': 'Non-English', 'value': 0}],
                        value=1,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),

                    html.Label("Genres"),
                    dcc.Dropdown(
                        id='genres-input',
                        options=[{'label': genre, 'value': genre} for genre in all_genres],
                        multi=True,
                        value=['Action'],
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),

                    html.Button('Predict Success', id='predict-button', n_clicks=0,
                                style={'background-color': '#4CAF50', 'color': 'white',
                                       'width': '100%', 'padding': '10px'})
                ], style={"width": "35%", "paddingRight": "10px"}),

                # Outputs (50% width)
                html.Div([
                    # First row container
                    html.Div([
                        # Success Probability
                        html.Div([
                            html.Div("Success Probability:", style={'fontWeight': 'bold'}),
                            html.Div(id='success-output', style={
                                'fontSize': '14px',
                                'padding': '8px',
                                'backgroundColor': '#ffffff',
                                'color': '#333333',
                                'borderRadius': '5px',
                                'marginTop': '5px'
                            })
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        # Rating
                        html.Div([
                            html.Div("Rating:", style={'fontWeight': 'bold'}),
                            html.Div(id='rating-output', style={
                                'fontSize': '14px',
                                'padding': '8px',
                                'backgroundColor': '#ffffff',
                                'color': '#333333',
                                'borderRadius': '5px',
                                'marginTop': '5px'
                            })
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ], style={'marginBottom': '15px'}),
                    
                    # Second row container
                    html.Div([
                        # Revenue
                        html.Div([
                            html.Div("Revenue:", style={'fontWeight': 'bold'}),
                            html.Div(id='revenue-output', style={
                                'fontSize': '14px',
                                'padding': '8px',
                                'backgroundColor': '#ffffff',
                                'color': '#333333',
                                'borderRadius': '5px',
                                'marginTop': '5px'
                            })
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        # ROI
                        html.Div([
                            html.Div("ROI:", style={'fontWeight': 'bold'}),
                            html.Div(id='roi-output', style={
                                'fontSize': '14px',
                                'padding': '8px',
                                'backgroundColor': '#ffffff',
                                'color': '#333333',
                                'borderRadius': '5px',
                                'marginTop': '5px'
                            })
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ], style={'marginBottom': '15px'}),
                    
                    # Recommendations section third row
                    html.Div([
                        html.Div("Recommendations for Improvement", 
                                style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        html.Div(id='recommendations', style={
                            'padding': '10px',
                            'backgroundColor': '#ffffff',
                            'borderRadius': '5px',
                            'minHeight': '80px',
                            'overflowY': 'auto',
                            'border': '1px solid #e0e0e0',  # Light gray border
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'  # Subtle shadow
                        })
                    ])
                ], style={
                    'width': '65%',
                    'paddingLeft': '10px',
                    'boxSizing': 'border-box'
                })
            ], style={"display": "flex"}),
        ], style={"width": "40%", "padding": "10px", "display": "inline-block", "verticalAlign": "top", "height": "100%"}),

        # RIGHT SIDE (60%) - Sensitivity Plot
        html.Div([
            dcc.Graph(id='sensitivity-plot', style={'height': '40vh', 'width': '100%'})
        ], style={"width": "60%", "padding": "10px", "display": "inline-block", "verticalAlign": "top", "height": "100%"})
    ], style={"display": "flex", "height": "45vh"})
], style={
    'backgroundColor': '#f5f5f5',  # Very light gray
    'color': '#333333',            # Dark gray text
    "fontFamily": 'Sans-serif', # prettier font
    'margin': '0',          # Remove outer margin
    'padding': '0',         # Remove outer padding
    'height': '100vh',      # Full viewport height
    'width': '100vw',       # Full viewport width

})



# === Callbacks ===
from dash.dependencies import Input, Output, State


# === Current filter Callback ===
#This description of the function was made with the help of AI agent
'''
Callback to update the current zoom-based filter for the scatter plot.

This function listens to zoom/pan events (`relayoutData`) from the scatter plot.
It updates the `zoom_ids` in the `current-filter` store with a list of movie titles
(`title_y`) that fall within the currently zoomed-in region of the scatter plot.

The function:
1. Initializes or preserves the current filter state.
2. Checks for valid axis range data in the relayout event.
3. Applies any existing genre filter to the dataset.
4. Converts axis range values appropriately for log-scaled axes.
5. Filters the dataset to include only points within the visible plot range.
6. Extracts and returns the titles of the points currently in view.

This filtered list (`zoom_ids`) is used to identify which data points are currently
visible within the zoomed area of the scatter plot, and are combined with other
filters (like genre) to dynamically update the visualization of other components.'''
@app.callback(
    Output("current-filter", "data", allow_duplicate=True),
    Input("scatter-plot", "relayoutData"),
    [
        State("current-filter", "data"),
        State("x-axis-dropdown", "value"),
        State("y-axis-dropdown", "value")
    ],
    prevent_initial_call=True
)
def update_filter_from_zoom(relayout_data, current_filter, x_axis, y_axis):
    if current_filter is None:
        current_filter = {"genres": None, "scatter_ids": None, "zoom_ids": None}

    if not relayout_data:
        return {**current_filter, "zoom_ids": None}

    # Check if we have axis range data (for either linear or log axes)
    x_keys = [f"xaxis.range[0]", f"xaxis.range[1]"]
    y_keys = [f"yaxis.range[0]", f"yaxis.range[1]"]
    
    if not (all(k in relayout_data for k in x_keys) and all(k in relayout_data for k in y_keys)):
        return {**current_filter, "zoom_ids": None}

    x0 = relayout_data[x_keys[0]]
    x1 = relayout_data[x_keys[1]]
    y0 = relayout_data[y_keys[0]]
    y1 = relayout_data[y_keys[1]]

    # Start with full dataset
    zoom_dff = df
    
    # Apply genre filter if one exists
    if current_filter.get("genres"):
        selected_genre = current_filter["genres"]
        if '-' in selected_genre:
            zoom_dff = zoom_dff[zoom_dff["genres_y"].str.startswith(selected_genre, na=False)]
        else:
            zoom_dff = zoom_dff[zoom_dff["genres_y"].str.startswith(selected_genre, na=False)]
    
    # Handle axis transformations based on current axis type
    x_log = x_axis in ["budget", "revenue", "profit", "roi"]
    y_log = y_axis in ["budget", "revenue", "profit", "roi"]
    
    # Apply x-axis filter
    if x_log:
        x_min = 10 ** x0 if x0 is not None else 0
        x_max = 10 ** x1 if x1 is not None else float('inf')
    else:
        x_min = x0 if x0 is not None else 0
        x_max = x1 if x1 is not None else float('inf')
    
    # Apply y-axis filter
    if y_log:
        y_min = 10 ** y0 if y0 is not None else 0
        y_max = 10 ** y1 if y1 is not None else float('inf')
    else:
        y_min = y0 if y0 is not None else 0
        y_max = y1 if y1 is not None else float('inf')
    
    # Apply range filters
    zoom_dff = zoom_dff[
        (zoom_dff[x_axis] >= x_min) & (zoom_dff[x_axis] <= x_max) &
        (zoom_dff[y_axis] >= y_min) & (zoom_dff[y_axis] <= y_max)
    ]

    zoom_ids = zoom_dff["title_y"].tolist()
    return {**current_filter, "zoom_ids": zoom_ids}



# === Updating the Current filter Callback ===
#This description of the function was made with the help of AI agent
"""
Callback to update the current filter state based on user interactions
with the scatter plot (point selection) or the genre icicle chart (genre selection).

This function listens for selections from:
- The scatter plot (`selectedData`) to identify which movie titles are selected.
- The genre icicle chart (`clickData`) to identify which genre or subgenre is clicked.

The function:
1. Initializes the filter state if it's not already set.
2. Detects which input triggered the callback.
3. If triggered by the scatter plot:
    - Extracts the `hovertext` from selected points.
    - Updates the `scatter_ids` in the filter with the selected movie titles.
4. If triggered by the icicle chart:
    - Parses the genre path from the clicked data.
    - Updates the `genres` in the filter with either the parent genre or subgenre.
5. Returns the updated filter dictionary with the new `scatter_ids` or `genres`.

This enables coordinated filtering across visual components: genre and scatter plot
selections update a shared filter store (`current-filter`) that can be used by
other callbacks to dynamically adjust what is shown in the app.
"""
@app.callback(
    Output("current-filter", "data"),
    [
        Input("scatter-plot", "selectedData"),
        Input("genre-icicle", "clickData")
    ],
    State("current-filter", "data")
)
def update_current_filter(scatter_selected, icicle_click, current_filter):
    ctx = callback_context

    if current_filter is None:
        current_filter = {"genres": None, "scatter_ids": None, "zoom_ids": None}

    if not ctx.triggered:
        return current_filter

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "scatter-plot":
        if scatter_selected is None or "points" not in scatter_selected:
            return {**current_filter, "scatter_ids": None}
        selected_titles = [point["hovertext"] for point in scatter_selected["points"]]
        return {**current_filter, "scatter_ids": selected_titles}


    elif triggered_id == "genre-icicle":
        if icicle_click is None:
            return {**current_filter, "genres": None}

        clicked_id = icicle_click["points"][0].get("id", None)
        path = clicked_id.split("/")

        if len(path) == 3:  # Clicked on a subgenre
            parent_genre = path[1]
            subgenre = path[2]
            clicked_id = f"{parent_genre}/{subgenre}"
        elif len(path) == 2:  # Clicked on a parent genre
            clicked_id = path[1]
        else:
            clicked_id = None
            
        return {**current_filter, "genres": clicked_id}
    
    return current_filter
    

# === Scatterplot Callback ===
#This description of the function was made with the help of AI agent
@app.callback(
    Output("scatter-plot", "figure"),
    [
        Input("shared-year-slider", "value"),
        Input("color-by", "value"),
        Input("x-axis-dropdown", "value"),
        Input("y-axis-dropdown", "value"),
        Input("current-filter", "data"),
        Input("prediction-store", "data")
    ],
    [State("scatter-plot", "relayoutData")]
)
def update_scatter(year_range, color_by, x_axis, y_axis,current_filter, prediction_data, relayout_data):
    """
    Callback to update the scatter plot figure based on multiple inputs:
    year range, axis selections, color grouping, current filters, and prediction data.

    This function:
    1. Filters the dataset based on the selected release year range.
    2. Applies genre filtering from the genre icicle chart if applicable.
    3. Applies axis-specific log scaling and outlier clipping for color.
    4. Normalizes marker size based on vote count.
    5. Adds a prediction point (if prediction data is provided) with custom styling.
    6. Preserves zoom state manually using `relayoutData`.
    7. Applies styling and formatting to the final plot.

    Returns:
        A Plotly scatter figure to be rendered in the "scatter-plot" component.
    """
    #filter by release year
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    #filter by genre if present
    if current_filter and current_filter.get("genres"):
        clicked_id = current_filter["genres"]
        genres_selected = clicked_id.split('/')

        if len(genres_selected) == 1:
            selected_primary = genres_selected[0]

            def match_primary_genre(row):
                if pd.isna(row["genres_y"]):
                    return False
                return row["genres_y"].split('-')[0] == selected_primary

            dff = dff[dff.apply(match_primary_genre, axis=1)]

        elif len(genres_selected) == 2:
            exact_genre = '-'.join(genres_selected)

            def match_exact_genre(row):
                return row["genres_y"] == exact_genre

            dff = dff[dff.apply(match_exact_genre, axis=1)]

    # If no data is available after filtering, return an empty plot with a message
    if dff.empty:
        return px.scatter(title="No data to display for this selection.")

    # Drop rows with missing required values
    dff = dff.dropna(subset=["budget", "revenue", "vote_count", color_by] if color_by != "cluster_label" else numeric_features)

    # Clip color variable to reduce outlier influence
    if color_by in ["roi", "vote_average"]:
        lower = dff[color_by].quantile(0.05)
        upper = dff[color_by].quantile(0.95)
        dff["color_clip"] = dff[color_by].clip(lower, upper)
        color_by_plot = "color_clip"
    elif color_by in ["runtime", "profit"]: 
        lower = dff[color_by].quantile(0.01)
        upper = dff[color_by].quantile(0.99)
        dff["color_clip"] = dff[color_by].clip(lower, upper)
        color_by_plot = "color_clip"
    else:
        color_by_plot = color_by

    # Normalize size of points based on vote_count for better visibility
    scaler = MinMaxScaler(feature_range=(2, 80))
    dff["scaled_size"] = scaler.fit_transform(dff[["vote_count"]])

    # Create the base scatter plot
    fig = px.scatter(
        dff,
        x=x_axis,
        y=y_axis,
        color=color_by_plot,
        size="vote_count",
        hover_name="title_y",
        hover_data=["release_year", "budget", "revenue", "profit", "roi", "vote_average"],
        labels={"color_clip": color_by},
        log_x=x_axis in ["budget", "revenue", "profit", "roi"],
        log_y=y_axis in ["budget", "revenue", "profit", "roi"]
    )


    # Add prediction marker if available
    if isinstance(prediction_data, dict):
        # Map selected axes to corresponding keys in prediction_data
        pred_axis_map = {
            'budget': 'budget',
            'revenue': 'revenue_pred',  
            'profit': 'profit_pred' if 'profit_pred' in prediction_data else None,
            'roi': 'roi_pred' if 'roi_pred' in prediction_data else None,
            'vote_average': 'rating_pred',
            'runtime': 'runtime_pred' if 'runtime_pred' in prediction_data else None,
        }

        pred_x_key = pred_axis_map.get(x_axis)
        pred_y_key = pred_axis_map.get(y_axis)

        # Only add prediction marker if both axes have prediction values
        if pred_x_key in prediction_data and pred_y_key in prediction_data:
            fig.add_trace(go.Scattergl(
                x=[prediction_data[pred_x_key]],
                y=[prediction_data[pred_y_key]],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    color='lime',
                    size=15,
                    line=dict(width=3, color='black')
                ),
                name='Prediction',
                hovertemplate=(
                    f"Predicted {x_axis.replace('_',' ').title()}: $%{{x:.2s}}<br>"
                    f"Predicted {y_axis.replace('_',' ').title()}: $%{{y:.2s}}<br>"
                    "Predicted Rating: %{customdata[0]:.1f}<br>"
                    "Success Probability: %{customdata[1]:.1%}<extra></extra>"
                ),
                customdata=[[prediction_data.get('rating_pred', None),
                            prediction_data.get('success_prob', None)]]
            ))

    # Update layout labels and hover mode
    fig.update_layout(
        xaxis_title=x_axis.replace("_", " ").title(),
        yaxis_title=y_axis.replace("_", " ").title(),
        hovermode="closest"
    )

    # Preserve zoom level manually based on relayout data
    if relayout_data:
        for axis in ["xaxis", "yaxis"]:
            r0 = relayout_data.get(f"{axis}.range[0]")
            r1 = relayout_data.get(f"{axis}.range[1]")
            if r0 is not None and r1 is not None:
                fig.update_layout(**{axis: dict(range=[r0, r1])})

    # Apply custom theming/styling
    fig = prettify_figure(
        fig,
        x_axis=x_axis,
        y_axis=y_axis
    )
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        font=dict(color='#333333', family='Sans-serif'),


    )
    return fig


# === Icicle Genre Callback ===
#This description of the function was made with the help of AI agent
@app.callback(
    Output("genre-icicle", "figure"),
    [
        Input("shared-year-slider", "value"),
        Input("color-by", "value"),
        Input("current-filter", "data")
    ],
    [State("genre-icicle", "relayoutData")]
)
def update_icicle(year_range, color_by, current_filter, relayout_data):
    """
    Callback to update the icicle chart showing genre hierarchy.

    Parameters:
    - year_range: Tuple containing (min_year, max_year) for filtering the dataset.
    - color_by: Metric to color the subgenres by (e.g., 'roi', 'profit', 'vote_average', or 'count').
    - current_filter: Dictionary storing selected filters (e.g., zoomed scatter plot or clicked icicle genre).

    Returns:
    - A Plotly icicle chart figure displaying primary and subgenres with relative metrics.
    """
    # Check if this update was triggered by a genre click
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "current-filter.data":
        # If the current filter change was due to a genre click, don't update the icicle
        if current_filter and current_filter.get("genres"):
            # Return dash.no_update to prevent the icicle from refreshing
            return dash.no_update
        
    #filter by release year
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    #Apply filters from scatter or zoom selections
    if current_filter:
        if current_filter.get("zoom_ids"):
            dff = dff[dff["title_y"].isin(current_filter["zoom_ids"])]
        elif current_filter.get("scatter_ids"):
            dff = dff[dff["title_y"].isin(current_filter["scatter_ids"])]

    #Build hierarchical genre structure
    genre_hierarchy = []

    # Collect unique primary genres
    primary_genres = set()
    for _, row in dff.iterrows():
        if pd.isna(row['genres_y']):
            continue
        genres = row['genres_y'].split('-')
        if genres:
            primary_genres.add(genres[0])

    # Populate hierarchy with primary/subgenre combinations
    for _, row in dff.iterrows():
        if pd.isna(row['genres_y']):
            continue
        genres = row['genres_y'].split('-')
        if len(genres) > 1:
            primary = genres[0]
            for sub in genres[1:]:
                genre_hierarchy.append({
                    'primary': primary,
                    'sub': sub,
                    'ids': f"{primary}-{sub}",
                    'labels': sub,
                    'parents': primary,
                    'value': 1,  # placeholder value; replaced later
                    'budget': row['budget'],
                    'revenue': row['revenue'],
                    'profit': row['profit'],
                    'roi': row['roi'],
                    'vote_average': row['vote_average']
                })

    #If no data to show, return empty icicle
    if not genre_hierarchy:
        return px.icicle(title="No genre data available.")

    #Create DataFrame from hierarchy
    df_hier = pd.DataFrame(genre_hierarchy)

    #Aggregate values based on selected color_by metric
    if color_by == 'count':
        agg_df = df_hier.groupby(['ids', 'labels', 'parents']).size().reset_index(name='value')
        color_col = 'value'
    else:
        agg_df = df_hier.groupby(['ids', 'labels', 'parents']).agg({
            'roi': 'mean',
            'profit': 'mean', 
            'vote_average': 'mean'
        }).reset_index()
        color_col = color_by

        # Clip outliers to avoid skewing colors
        if color_by in ['roi', 'vote_average']:
            lower = agg_df[color_by].quantile(0.01)
            upper = agg_df[color_by].quantile(0.99)
            agg_df[color_by] = agg_df[color_by].clip(lower, upper)

    #Normalize value by subgenre count to ensure equal size within each primary genre
    normalized_df = agg_df.copy()
    normalized_df['primary'] = normalized_df['ids'].apply(lambda x: x.split('-')[0])
    subgenres_only = normalized_df[normalized_df['parents'] != '']

    subgenre_counts = subgenres_only.groupby('primary').size().to_dict()
    subgenres_only['relative_value'] = subgenres_only['primary'].apply(
        lambda p: 1 / subgenre_counts[p] if subgenre_counts.get(p, 0) > 0 else 0
    )
    subgenres_only['root'] = 'All Movies'
    
    #Build icicle figure
    fig = px.icicle(
        subgenres_only,
        path=['root','parents', 'labels'],
        values='relative_value',
        color=color_col,
        branchvalues='total',
    )

    #Update layout and styling
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        showlegend=True,
        annotations=[], 
        coloraxis_showscale=False,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f5f5f5',
        font=dict(color='#333333', family='Sans-serif'),
    )

    return fig



# ====== Callback for Success Prediction Model ======
#This description of the function was made with the help of AI agent
@app.callback(
    [Output('success-output', 'children'),
     Output('rating-output', 'children'),
     Output('revenue-output', 'children'),
     Output('roi-output', 'children'),
     Output('recommendations', 'children'),
     Output('prediction-store', 'data')],  
    Output('sensitivity-plot', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [State('budget-input', 'value'),
     State('runtime-input', 'value'),
     State('language-input', 'value'),
     State('genres-input', 'value')]
)
def update_predictions(n_clicks, budget, runtime, language, genres):
    """
    Callback to generate movie outcome predictions based on user input.
    Outputs predicted success probability, rating, revenue, ROI, recommendations, 
    prediction data (for plotting), and a sensitivity chart.

    Parameters:
        n_clicks (int): Button click trigger.
        budget (float): Budget in millions.
        runtime (int): Movie runtime in minutes.
        language (bool): English language (True/False).
        genres (list): Selected genres.

    Returns:
        Tuple: UI outputs (success, rating, revenue, ROI), recommendations (HTML),
               prediction-store data (dict), sensitivity plot (Plotly figure).
    """
    #fixed year since we want to predict movie that will come out now
    year = 2025
    if n_clicks == 0:
        return "", "", "", "", "", "", go.Figure()
    
    # === Prepare input data ===
    input_data = pd.DataFrame({
        'budget_log': [np.log1p(budget * 1000000)],   # Budget converted to dollars and log-transformed
        'runtime': [runtime],
        'is_english': [language],
        'release_year': [year]
    })
    
    # Add genre one-hot encoded columns
    for genre in all_genres:
        input_data[genre] = 1 if genre in genres else 0
    
    # === Model Predictions ===
    success_prob = success_model.predict_proba(input_data)[0][1]
    rating_pred = rating_model.predict(input_data)[0]
    revenue_pred = np.expm1(revenue_model.predict(input_data)[0])
    
    # === ROI Calculation ===
    roi = revenue_pred / (budget * 1000000)
    
    # === Generate Text Recommendations ===
    recommendations = generate_recommendations(budget, runtime, language, year, genres, 
                                             success_prob, rating_pred, roi)
    
    # === Sensitivity Plot ===
    fig = create_sensitivity_plot(budget, runtime, language, year, genres)
    fig.update_layout(
        title = None,
        yaxis=dict(
            showticklabels=False,
            title=None
        ),
        yaxis2=dict(
            showticklabels=False,
            title=None
        ),
        yaxis3=dict(
            showticklabels=False,
            title=None
        )
    )

    # === Output formatting ===
    success_output = f"🎬 Success Probability: {success_prob:.1%}"
    rating_output = f"⭐ Predicted Rating: {rating_pred:.1f}/10"
    revenue_output = f"💰 Predicted Revenue: ${revenue_pred/1000000:.1f} million"
    roi_output = f"📈 Predicted ROI: {roi:.1f}x"

    # === Store prediction for scatter plot or tracking ===
    prediction_data = {
        'budget': budget * 1000000,
        'revenue_pred': revenue_pred,
        'profit_pred': revenue_pred- budget* 1000000,
        'rating_pred': rating_pred,
        'success_prob': success_prob,
        'runtime_pred': runtime,
        'roi_pred': roi
    }
    
    return success_output, rating_output, revenue_output, roi_output, recommendations, prediction_data, fig


def generate_recommendations(budget, runtime, language, year, genres, 
                           success_prob, rating_pred, roi):
    
    """
    Generate a list of tailored recommendations to improve movie performance.

    Recommendations include:
        - Adding beneficial genres
        - Reducing budget to improve ROI
        - Changing runtime for higher success probability

    Returns:
        HTML element with suggestions or default success message
    """
    recommendations = []
    
    # === Thresholds for 'low' performance based on training data ===
    low_success_threshold = success_model.predict_proba(X_train).T[1].mean()
    low_rating_threshold = y_rating_train.mean()
    low_roi_threshold = (np.expm1(y_revenue_train) / np.expm1(X_train['budget_log'])).mean()

    # === Genre enhancement for rating improvement ===
    if rating_pred < low_rating_threshold:
        best_improvement = 0
        best_genre = None
        for genre in all_genres:
            if genre not in genres:
                test_genres = genres + [genre]
                test_input = prepare_input(budget, runtime, language, year, test_genres)
                new_rating = rating_model.predict(test_input)[0]
                improvement = new_rating - rating_pred
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_genre = genre

        if best_genre and best_improvement > 0.3:
            recommendations.append(
                html.Li(f"🎭 Adding '{best_genre}' genre could increase predicted rating to {rating_pred + best_improvement:.1f}")
            )

    # === Budget reduction if ROI is poor ===
    if roi < low_roi_threshold:
        for factor in [0.9, 0.8, 0.7]:
            new_budget = budget * factor
            test_input = prepare_input(new_budget, runtime, language, year, genres)
            new_revenue = np.expm1(revenue_model.predict(test_input)[0])
            new_roi = new_revenue / (new_budget * 1_000_000)
            if new_roi > roi * 1.15:
                recommendations.append(
                    html.Li(f"💵 Reducing budget to ${new_budget:.1f}M may improve ROI to {new_roi:.1f}x")
                )
                break

    # === Runtime adjustment to improve success probability ===
    tested_runtimes = list(range(80, 160, 10))
    best_runtime = runtime
    best_success = success_prob

    for rt in tested_runtimes:
        if rt == runtime:
            continue  
        test_input = prepare_input(budget, rt, language, year, genres)
        test_success = success_model.predict_proba(test_input)[0][1]
        if test_success > best_success:
            best_success = test_success
            best_runtime = rt

    if best_runtime != runtime and best_success > success_prob + 0.05:
        recommendations.append(
            html.Li(f"⏱️ Adjusting runtime to {best_runtime} minutes may increase success probability to {best_success:.1%}")
        )

    # === Default fallback message ===
    if not recommendations:
        return html.P("✅ Your current parameters are well optimized for success!")

    return html.Ul(recommendations, style={'listStyleType': 'none', 'paddingLeft': '0'})


def prepare_input(budget, runtime, language, year, genres):
    """
    Prepares the input DataFrame for model predictions with proper feature engineering.

    Returns:
        DataFrame with transformed budget and one-hot encoded genres.
    """
    input_data = pd.DataFrame({
        'budget_log': [np.log1p(budget * 1000000)],
        'runtime': [runtime],
        'is_english': [language],
        'release_year': [year]
    })
    
    for genre in all_genres:
        input_data[genre] = 1 if genre in genres else 0
    
    return input_data

def create_sensitivity_plot(budget, runtime, language, year, genres):
    """
    Generates a sensitivity analysis plot showing how varying budget affects:
    - Predicted rating
    - Revenue
    - Success probability

    Returns:
        Plotly figure with three metrics plotted against budget.
    """
    budget_range = np.linspace(budget * 0.5, budget * 2, 10)
    ratings = []
    revenues = []
    success_probs = []
    
    for b in budget_range:
        input_data = prepare_input(b, runtime, language, year, genres)
        ratings.append(rating_model.predict(input_data)[0])
        revenues.append(np.expm1(revenue_model.predict(input_data)[0]) / 1000000)
        success_probs.append(success_model.predict_proba(input_data)[0][1] * 100)
    
    fig = go.Figure()
    
    # Add lines for each prediction
    fig.add_trace(go.Scatter(
        x=budget_range, y=ratings,
        name='Predicted Rating',
        yaxis='y1',
        line=dict(color='blue', width=3),
        hovertemplate="Rating: %{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=budget_range, y=revenues,
        name='Predicted Revenue (million $)',
        yaxis='y2',
        line=dict(color='green', width=3),
        hovertemplate="Revenue: %{y:.2f}M<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=budget_range, y=success_probs,
        name='Success Probability (%)',
        yaxis='y3',
        line=dict(color='red', width=3),
        hovertemplate="Success: %{y:.2f}%<extra></extra>"
    ))
    
    # Apply consistent styling
    fig = prettify_sensitivity_figure(
        fig,
        title='Sensitivity to Budget Changes'
    )
    
    # Axis formatting
    fig.update_layout(
        xaxis_title='Budget (million $)',
        yaxis=dict(title='Rating (1-10)', color='blue'),
        yaxis2=dict(title='Revenue (million $)', color='green', overlaying='y', side='right'),
        yaxis3=dict(title='Success Probability (%)', color='red', overlaying='y', side='left')
    )
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        font=dict(color='#333333', family='Sans-serif'),
        legend=dict(
            bgcolor='#f5f5f5',  # Set legend background color
            bordercolor="#000000",  # Optional: border color
            borderwidth=1,  # Optional: border width
            font=dict(
                color='#333333'  # Optional: change legend text color if needed
            )
        )
    )
    return fig


# Run app
if __name__ == "__main__":
    app.run(debug=True)