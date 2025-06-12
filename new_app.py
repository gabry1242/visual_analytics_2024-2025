import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.graph_objects as go
from itertools import combinations
import networkx as nx
import seaborn as sns
from collections import Counter
from dash import callback_context
from sklearn.preprocessing import MinMaxScaler

# Load and prepare data
df = pd.read_csv("merged_with_tags.csv")
df = df.dropna(subset=["budget", "revenue", "release_year", "title_y", "vote_count", 'genres_y'])
df = df.drop_duplicates(subset=['title_y'])
df["release_year"] = df["release_year"].astype(int)

# Extra metrics
df["profit"] = df["revenue"] - df["budget"]
df["profit_margin"] = (df["profit"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
df["roi"] = df["profit_margin"] * 100
df["primary_genre"] = df["genres_y"].str.split("-").str[0]

#tags preparation
all_tags = df['tag'].dropna().str.split(';').explode().str.strip().str.lower()
# Count the occurrences of each tag
tag_counts = all_tags.value_counts()
df['tag_list'] = df['tag'].fillna('').str.lower().str.split(';').apply(lambda tags: [t.strip() for t in tags if t.strip()])
# Filter tags that appear more than 5 times
popular_tags = tag_counts[tag_counts > 300]
# Get co-occurrences of popular tags in the same movie
# Prepare genre hierarchy data
def prepare_genre_data(df):
    # Split genres and explode into multiple rows
    genre_df = df.assign(genres=df['genres_y'].str.split('|')).explode('genres')
    
    # Create a count of movies per genre
    genre_counts = genre_df.groupby('genres').size().reset_index(name='count')
    
    # For simplicity, we'll create a two-level hierarchy (primary genre -> sub-genre)
    # In a real app, you might want a more sophisticated hierarchy
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

# Define features used for clustering
numeric_features = ["budget", "revenue", "vote_average", "vote_count", "runtime", "profit", "roi"]




# ====== Success Prediction Models ======
# Prepare data for success prediction
df['genres_list'] = df['genres_y'].str.split('-')
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres_list']), 
                           columns=mlb.classes_, 
                           index=df.index)

# Calculate success metric
df['success'] = (df['vote_average'] > 6.5) & (df['revenue'] > df['budget'] * 2)
df['budget_log'] = np.log1p(df['budget'])
df['revenue_log'] = np.log1p(df['revenue'])
df['is_english'] = (df['original_language'] == 'en').astype(int)

# Prepare features and targets
features = pd.concat([
    df[['budget_log', 'runtime', 'is_english', 'release_year']],
    genre_encoded
], axis=1)

success_target = df['success']
rating_target = df['vote_average']
revenue_target = df['revenue_log']

# Split data
X_train, X_test, y_success_train, y_success_test, y_rating_train, y_rating_test, y_revenue_train, y_revenue_test = train_test_split(
    features, success_target, rating_target, revenue_target, test_size=0.2, random_state=42
)

# Build models
success_model = RandomForestClassifier(n_estimators=100, random_state=42)
success_model.fit(X_train, y_success_train)

rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model.fit(X_train, y_rating_train)

revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
revenue_model.fit(X_train, y_revenue_train)

# Get all unique genres from the data
all_genres = sorted(list(set([genre for sublist in df['genres_list'].dropna() for genre in sublist])))




# Dash app setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# network_data = preprocess_tag_network(df)

app.title = "Movie Success Studio"

app.layout = html.Div([
    dcc.Store(id="current-filter", data={"genres": None, "scatter_ids": None, "zoom_ids": None}),
    dcc.Store(id="prediction-store", data=None),
    # Left Column
    html.Div([
        html.Label("Color by:"),
        dcc.Dropdown(
            id="genre-color-by",
            options=[
                {"label": "Count", "value": "count"},
                {"label": "ROI", "value": "roi"},
                {"label": "Vote Average", "value": "vote_average"}
            ],
            value="count",
            style={"margin-bottom": "10px"}
        ),
        html.Label("Year Range:"),
        dcc.RangeSlider(
            id="shared-year-slider",
            min=2000, max=2020, step=1, value=[2000, 2020],
            marks={str(y): str(y) for y in range(2000, 2021, 5)},
            tooltip={"always_visible": True},
            allowCross=False
        ),
        dcc.Graph(id="genre-sunburst", style={"height": "40vh", "width": "100%"}),

        html.H3("Movie Success Predictor", style={'marginTop': '20px'}),

        html.Div([
            html.Label("Budget (in millions)"),
            dcc.Input(id='budget-input', type='number', value=50, min=1, step=1,
                      style={'width': '100%'})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Runtime (minutes)"),
            dcc.Input(id='runtime-input', type='number', value=120, min=60, max=240, step=5,
                      style={'width': '100%'})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Original Language"),
            dcc.Dropdown(
                id='language-input',
                options=[{'label': 'English', 'value': 1},
                         {'label': 'Non-English', 'value': 0}],
                value=1,
                style={'width': '100%'}
            )
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Release Year"),
            dcc.Input(id='year-input', type='number', value=2023, min=1900, max=2030, step=1,
                      style={'width': '100%'})
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("Genres"),
            dcc.Dropdown(
                id='genres-input',
                options=[{'label': genre, 'value': genre} for genre in all_genres],
                multi=True,
                value=['Action'],
                style={'width': '100%'}
            )
        ], style={'marginBottom': '10px'}),

        html.Button('Predict Success', id='predict-button', n_clicks=0,
                    style={'background-color': '#4CAF50', 'color': 'white', 'width': '100%',
                           'padding': '10px'})
    ], style={"width": "20%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

    # Right Column (80%)
    html.Div([
        html.Div([
            html.Div([
                html.Label("Color by:"),
                dcc.Dropdown(
                    id="color-by",
                    options=[
                        {"label": "Vote Average", "value": "vote_average"},
                        {"label": "ROI", "value": "roi"},
                        {"label": "Runtime", "value": "runtime"},
                        {"label": "Primary Genre", "value": "primary_genre"},
                        {"label": "Profit", "value": "profit"}
                    ],
                    value="vote_average",
                    style={"width": "200px", "marginRight": "10px"}
                )
            ], style={"display": "inline-block", "verticalAlign": "top"}),

            html.Div([
                html.Label("X-axis:"),
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
                    style={"width": "200px"}
                )
            ], style={"display": "inline-block", "verticalAlign": "top", "marginLeft": "20px"}),

            html.Div([
                html.Label("Y-axis:"),
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
                    style={"width": "200px", "marginLeft": "20px"}
                )
            ], style={"display": "inline-block", "verticalAlign": "top", "marginLeft": "20px"})
        ], style={"margin-bottom": "10px"}),
        dcc.Store(id="cluster-count-store", data=5),
        dcc.Graph(id="scatter-plot", style={"height": "50vh", "width": "100%"}),

        # Results Section Split 1:3
        html.Div([
            # Recommendations (1 part)
            html.Div([
                html.Div(id='success-output', style={
                    'fontSize': '16px', 'padding': '10px', 'marginBottom': '8px',
                    'backgroundColor': '#f8f9fa', 'borderRadius': '5px'
                }),
                html.Div(id='rating-output', style={
                    'fontSize': '16px', 'padding': '10px', 'marginBottom': '8px',
                    'backgroundColor': '#f8f9fa', 'borderRadius': '5px'
                }),
                html.Div(id='revenue-output', style={
                    'fontSize': '16px', 'padding': '10px', 'marginBottom': '8px',
                    'backgroundColor': '#f8f9fa', 'borderRadius': '5px'
                }),
                html.Div(id='roi-output', style={
                    'fontSize': '16px', 'padding': '10px', 'marginBottom': '8px',
                    'backgroundColor': '#f8f9fa', 'borderRadius': '5px'
                }),
                html.H4("Recommendations for Improvement", style={'marginTop': '20px'}),
                html.Div(id='recommendations', style={
                    'padding': '10px',
                    'backgroundColor': '#e9ecef',
                    'borderRadius': '5px'
                })
            ], style={'flex': 1, 'padding': '10px'}),

            # Sensitivity Plot (3 parts)
            html.Div([
                dcc.Graph(id='sensitivity-plot', style={'height': '100%', 'width': '100%'})
            ], style={'flex': 3, 'padding': '10px'})
        ], style={'display': 'flex', 'marginTop': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
    ], style={"width": "80%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"})
], style={"display": "flex"})




# === Callbacks ===
from dash.dependencies import Input, Output, State



@app.callback(
    Output("current-filter", "data", allow_duplicate=True),
    Input("scatter-plot", "relayoutData"),
    State("current-filter", "data"),
    prevent_initial_call=True
)
def update_filter_from_zoom(relayout_data, current_filter):
    if current_filter is None:
        current_filter = {"genres": None, "scatter_ids": None, "zoom_ids": None}

    if not relayout_data:
        return {**current_filter, "zoom_ids": None}

    keys = ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]
    if not all(k in relayout_data for k in keys):
        return {**current_filter, "zoom_ids": None}

    x0 = relayout_data["xaxis.range[0]"]
    x1 = relayout_data["xaxis.range[1]"]
    y0 = relayout_data["yaxis.range[0]"]
    y1 = relayout_data["yaxis.range[1]"]

    budget_min = 10 ** x0
    budget_max = 10 ** x1
    revenue_min = 10 ** y0
    revenue_max = 10 ** y1

    zoom_dff = df[
        (df["budget"] >= budget_min) & (df["budget"] <= budget_max) &
        (df["revenue"] >= revenue_min) & (df["revenue"] <= revenue_max)
    ]

    zoom_ids = zoom_dff["title_y"].tolist()
    return {**current_filter, "zoom_ids": zoom_ids}



@app.callback(
    Output("current-filter", "data"),
    [
        Input("scatter-plot", "selectedData"),
        Input("genre-sunburst", "clickData")
    ],
    State("current-filter", "data")
)
def update_current_filter(scatter_selected, sunburst_click, current_filter):
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

    elif triggered_id == "genre-sunburst":
        if sunburst_click is None:
            return {**current_filter, "genres": None}
        clicked_id = sunburst_click["points"][0].get("id", None)
        return {**current_filter, "genres": clicked_id}

    return current_filter


###SCATTERPLOT###
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
    
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    if current_filter and current_filter.get("genres"):
        clicked_id = current_filter["genres"]
        genres_selected = clicked_id.split('-')
        if not genres_selected:
            # No valid genre - skip filtering
            pass
        else:
            primary_genre = genres_selected[0]
            if len(primary_genre.split('/')) == 1:
                sub_genres = primary_genre
            else:
                parts = primary_genre.split('/')
                sub_genres = parts[0] + '-' + parts[1]

            def genre_match(row):
                if pd.isna(row['genres_y']):
                    return False
                return sub_genres in row['genres_y']

            dff = dff[dff.apply(genre_match, axis=1)]


    if dff.empty:
        return px.scatter(title="No data to display for this selection.")
    # rest of your original code here ...
    dff = dff.dropna(subset=["budget", "revenue", "vote_count", color_by] if color_by != "cluster_label" else numeric_features)

    if color_by in ["roi", "vote_average"]:
        lower = dff[color_by].quantile(0.05)
        upper = dff[color_by].quantile(0.95)
        dff["color_clip"] = dff[color_by].clip(lower, upper)
        color_by_plot = "color_clip"
    elif color_by == "runtime":
        lower = dff[color_by].quantile(0.01)
        upper = dff[color_by].quantile(0.99)
        dff["color_clip"] = dff[color_by].clip(lower, upper)
        color_by_plot = "color_clip"
    else:
        color_by_plot = color_by

    scaler = MinMaxScaler(feature_range=(2, 80))
    dff["scaled_size"] = scaler.fit_transform(dff[["vote_count"]])

    fig = px.scatter(
        dff,
        x=x_axis,
        y=y_axis,
        color=color_by_plot,
        size="vote_count",
        hover_name="title_y",
        hover_data=["release_year", "budget", "revenue", "profit", "roi", "vote_average"],
        labels={"color_clip": color_by},
        title=f"{x_axis.title()} vs {y_axis.title()} ({year_range[0]}–{year_range[1]})",
        log_x=x_axis in ["budget", "revenue", "profit", "roi"],
        log_y=y_axis in ["budget", "revenue", "profit", "roi"]
    )

    # Add prediction marker if available
    if prediction_data and isinstance(prediction_data, dict) and 'budget' in prediction_data:
        fig.add_trace(go.Scatter(
            x=[prediction_data['budget']],
            y=[prediction_data['revenue_pred']],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name='Prediction',
            hoverinfo='text',
            hovertext=f"Predicted Revenue: ${prediction_data['revenue_pred'] / 1e6:.1f}M<br>"
                      f"Predicted Rating: {prediction_data['rating_pred']:.1f}<br>"
                      f"Success Probability: {prediction_data['success_prob']:.1%}"
        ))

    fig.update_layout(
        xaxis_title=x_axis.replace("_", " ").title(),
        yaxis_title=y_axis.replace("_", " ").title(),
        hovermode="closest"
    )

    # --- Preserve zoom manually ---
    if relayout_data:
        for axis in ["xaxis", "yaxis"]:
            r0 = relayout_data.get(f"{axis}.range[0]")
            r1 = relayout_data.get(f"{axis}.range[1]")
            if r0 is not None and r1 is not None:
                fig.update_layout(**{axis: dict(range=[r0, r1])})

    return fig

@app.callback(
    Output("movie-table", "children"),
    [Input("scatter-plot", "relayoutData"),
     Input("year-slider", "value"),
     Input("movie-sort-by", "value")]
)
def update_movie_table(relayout_data, year_range, sort_by):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    if relayout_data and all(k in relayout_data for k in ["xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"]):
        x0 = relayout_data["xaxis.range[0]"]
        x1 = relayout_data["xaxis.range[1]"]
        y0 = relayout_data["yaxis.range[0]"]
        y1 = relayout_data["yaxis.range[1]"]
        budget_min = 10 ** x0
        budget_max = 10 ** x1
        revenue_min = 10 ** y0
        revenue_max = 10 ** y1

        dff = dff[
            (dff["budget"] >= budget_min) & (dff["budget"] <= budget_max) &
            (dff["revenue"] >= revenue_min) & (dff["revenue"] <= revenue_max)
        ]

    top_movies = dff.sort_values(sort_by, ascending=False).head(15)
    movie_list = [
        html.Li([
            html.Strong(movie["title_y"]),
            html.Br(),
            f"Year: {movie['release_year']} | Revenue: ${movie['revenue']:,.0f} | ROI: {movie['roi']:.1f}% | Rating: {movie['vote_average']:.1f}"
        ]) for _, movie in top_movies.iterrows()
    ]
    return movie_list



# Genre Exploration tab
@app.callback(
    Output("genre-sunburst", "figure"),
    [
        Input("shared-year-slider", "value"),
        Input("genre-color-by", "value"),
        Input("current-filter", "data")
    ]
)
def update_sunburst(year_range, color_by, current_filter):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    if current_filter:
        if current_filter.get("zoom_ids"):
            dff = dff[dff["title_y"].isin(current_filter["zoom_ids"])]
        elif current_filter.get("scatter_ids"):
            dff = dff[dff["title_y"].isin(current_filter["scatter_ids"])]

    genre_hierarchy = []
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
                    'budget': row['budget'],
                    'revenue': row['revenue'],
                    'profit': row['profit'],
                    'roi': row['roi'],
                    'vote_average': row['vote_average']
                })

    if not genre_hierarchy:
        return px.sunburst(title="No genre data available.")

    hierarchy_df = pd.DataFrame(genre_hierarchy)

    if color_by == 'count':
        agg_df = hierarchy_df.groupby(['ids', 'labels', 'parents']).size().reset_index(name='count')
        color_col = 'count'
        values = agg_df['count']
    else:
        agg_df = hierarchy_df.groupby(['ids', 'labels', 'parents']).agg({
            'roi': 'mean',
            'profit': 'mean',
            'vote_average': 'mean'
        }).reset_index()

        if color_by in ['roi', 'vote_average']:
            lower = agg_df[color_by].quantile(0.01)
            upper = agg_df[color_by].quantile(0.99)
            agg_df['color_scaled'] = agg_df[color_by].clip(lower, upper)
            color_col = 'color_scaled'
        else:
            color_col = color_by

        values = agg_df[color_by].abs()

    fig = px.sunburst(
        agg_df,
        path=['parents', 'labels'],
        values='count' if color_by == 'count' else values,
        color=color_col,
        title=f"Movie Genre Hierarchy ({year_range[0]}–{year_range[1]})",
        branchvalues="total"
    )

    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        hovermode="closest"
    )

    return fig

    

# Handle clicks on the sunburst to show movies in the selected genre
@app.callback(
    Output("movie-list-items", "children"),
    [Input("genre-sunburst", "clickData"),
    Input("genre-year-slider", "value")]
)
def update_movie_list(click_data, year_range):
    if click_data is None:
        return []

    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]
    
    # Get the full ID clicked (e.g., 'Horror-Thriller' or just 'Horror')
    clicked_id = click_data['points'][0].get('id', '')
    
    # Split the clicked id to get hierarchy path
    genres_selected = clicked_id.split('-') 
    
    if not genres_selected:
        return []


    primary_genre = genres_selected[0]
    if len(primary_genre.split('/')) == 1:
        sub_genres = primary_genre
    else:
        sub_genres = primary_genre.split('/')
        sub_genres = sub_genres[0]+'-'+sub_genres[1]

    # Filter rows where genres match the full path (order matters!)
    def genre_match(row):
        if pd.isna(row['genres_y']):
            return False
        return sub_genres in row['genres_y']

    filtered_movies = dff[dff.apply(genre_match, axis=1)]
    filtered_movies = filtered_movies.sort_values('revenue', ascending=False).head(15)

    return [
        html.Li([
            html.Strong(movie['title_y']),
            html.Br(),
            f"Year: {movie['release_year']} | Revenue: ${movie['revenue']:,.0f} | ROI: {movie['roi']:.1f}%"
        ]) for _, movie in filtered_movies.iterrows()
    ]

# ====== New Callbacks for Success Prediction ======
@app.callback(
    [Output('success-output', 'children'),
     Output('rating-output', 'children'),
     Output('revenue-output', 'children'),
     Output('roi-output', 'children'),
     Output('recommendations', 'children'),
     Output('prediction-store', 'data')],  # Store prediction data
    Output('sensitivity-plot', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [State('budget-input', 'value'),
     State('runtime-input', 'value'),
     State('language-input', 'value'),
     State('year-input', 'value'),
     State('genres-input', 'value')]
)
def update_predictions(n_clicks, budget, runtime, language, year, genres):
    if n_clicks == 0:
        return "", "", "", "", "", "", go.Figure()
    
    # Prepare input features
    input_data = pd.DataFrame({
        'budget_log': [np.log1p(budget * 1000000)],  # Convert to dollars
        'runtime': [runtime],
        'is_english': [language],
        'release_year': [year]
    })
    
    # Add genre columns
    for genre in all_genres:
        input_data[genre] = 1 if genre in genres else 0
    
    # Make predictions
    success_prob = success_model.predict_proba(input_data)[0][1]
    rating_pred = rating_model.predict(input_data)[0]
    revenue_pred = np.expm1(revenue_model.predict(input_data)[0])
    
    # Calculate ROI
    roi = revenue_pred / (budget * 1000000)
    
    # Generate recommendations
    recommendations = generate_recommendations(budget, runtime, language, year, genres, 
                                             success_prob, rating_pred, roi)
    
    # Create sensitivity plot
    fig = create_sensitivity_plot(budget, runtime, language, year, genres)
    
    # Format outputs
    success_output = f"🎬 Success Probability: {success_prob:.1%}"
    rating_output = f"⭐ Predicted Rating: {rating_pred:.1f}/10"
    revenue_output = f"💰 Predicted Revenue: ${revenue_pred/1000000:.1f} million"
    roi_output = f"📈 Predicted ROI: {roi:.1f}x"

    # Store prediction data for scatter plot
    prediction_data = {
        'budget': budget * 1000000,  # Convert from millions to dollars
        'revenue_pred': revenue_pred,
        'rating_pred': rating_pred,
        'success_prob': success_prob
    }
    
    return success_output, rating_output, revenue_output, roi_output, recommendations, prediction_data, fig

def generate_recommendations(budget, runtime, language, year, genres, 
                           success_prob, rating_pred, roi):
    recommendations = []
    
    # === 1. Dynamic thresholding based on training data quantiles ===
    low_success_threshold = success_model.predict_proba(X_train).T[1].mean()
    low_rating_threshold = y_rating_train.mean()
    low_roi_threshold = (np.expm1(y_revenue_train) / np.expm1(X_train['budget_log'])).mean()

    # === 2. Genre-based improvement: test.py adding each genre individually ===
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

    # === 3. Budget optimization: try decreasing by steps if ROI is low ===
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

    # === 4. Runtime optimization: search for best runtime within range ===
    # Always explore if runtime adjustment improves success
    tested_runtimes = list(range(80, 160, 10))
    best_runtime = runtime
    best_success = success_prob

    for rt in tested_runtimes:
        if rt == runtime:
            continue  # skip current value
        test_input = prepare_input(budget, rt, language, year, genres)
        test_success = success_model.predict_proba(test_input)[0][1]
        if test_success > best_success:
            best_success = test_success
            best_runtime = rt

    # Recommend if improvement is meaningful
    if best_runtime != runtime and best_success > success_prob + 0.05:
        recommendations.append(
            html.Li(f"⏱️ Adjusting runtime to {best_runtime} minutes may increase success probability to {best_success:.1%}")
        )

    # === Default fallback ===
    if not recommendations:
        return html.P("✅ Your current parameters are well optimized for success!")

    return html.Ul(recommendations, style={'listStyleType': 'none', 'paddingLeft': '0'})

def prepare_input(budget, runtime, language, year, genres):
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
    # Create sensitivity analysis for budget
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
    
    fig.add_trace(go.Scatter(
        x=budget_range, y=ratings,
        name='Predicted Rating',
        yaxis='y1',
        line=dict(color='blue'))
    )
    
    fig.add_trace(go.Scatter(
        x=budget_range, y=revenues,
        name='Predicted Revenue (million $)',
        yaxis='y2',
        line=dict(color='green'))
    )
    
    fig.add_trace(go.Scatter(
        x=budget_range, y=success_probs,
        name='Success Probability (%)',
        yaxis='y3',
        line=dict(color='red'))
    )
    
    fig.update_layout(
        title='Sensitivity to Budget Changes',
        xaxis_title='Budget (million $)',
        yaxis=dict(title='Rating (1-10)', color='blue'),
        yaxis2=dict(title='Revenue (million $)', color='green', overlaying='y', side='right'),
        yaxis3=dict(title='Success Probability (%)', color='red', overlaying='y', side='left', position=0.15),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig




# Run app
if __name__ == "__main__":
    app.run(debug=True)