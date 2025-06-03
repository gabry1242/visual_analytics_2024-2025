import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and prepare data
df = pd.read_csv("cleaned_output.csv")
df = df.dropna(subset=["budget", "revenue", "release_year", "title_y", "vote_count", 'genres_y'])
df = df.drop_duplicates(subset=['title_y'])
df["release_year"] = df["release_year"].astype(int)

# Extra metrics
df["profit"] = df["revenue"] - df["budget"]
df["profit_margin"] = (df["profit"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
df["roi"] = df["profit_margin"] * 100
df["primary_genre"] = df["genres_y"].str.split("-").str[0]

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

# Dash app setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Movie Analytics Dashboard"

app.layout = html.Div([
    html.H1("Movie Financial Analytics", style={"textAlign": "center"}),

    dcc.Store(id="cluster-count-store", data=3),  # default number of clusters

    dcc.Tabs([
        dcc.Tab(label="Budget vs Revenue", children=[
            html.Div([
                html.Label("Release Year Range"),
                dcc.RangeSlider(
                    id="year-slider",
                    min=df["release_year"].min(),
                    max=df["release_year"].max(),
                    step=1,
                    value=[df["release_year"].min(), df["release_year"].max()],
                    marks={str(year): str(year) for year in range(df["release_year"].min(), df["release_year"].max() + 1, 5)},
                    tooltip={"placement": "bottom"}
                ),

                html.Label("Color By"),
                dcc.Dropdown(
                    id="color-by",
                    options=[
                        {"label": "ROI (%)", "value": "roi"},
                        {"label": "Profit Margin", "value": "profit_margin"},
                        {"label": "Vote Average", "value": "vote_average"},
                        {"label": "Primary Genre", "value": "primary_genre"},
                        {"label": "Runtime", "value": "runtime"},
                        {"label": "Cluster Label", "value": "cluster_label"}
                    ],
                    value="roi",
                    clearable=False
                ),

                dcc.Graph(id="scatter-plot", style={"height": "600px"}),

                html.H3("Movie Table"),
                dash_table.DataTable(
                    id="movie-table",
                    columns=[
                        {"name": "Title", "id": "title_y"},
                        {"name": "Year", "id": "release_year"},
                        {"name": "Budget ($)", "id": "budget"},
                        {"name": "Revenue ($)", "id": "revenue"},
                        {"name": "Profit ($)", "id": "profit"},
                        {"name": "ROI %", "id": "roi"},
                        {"name": "Rating", "id": "vote_average"}
                    ],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px"},
                    style_header={"fontWeight": "bold"}
                )
            ], style={"padding": "20px"})
        ]),

        dcc.Tab(label="Dimensionality Reduction (TSNE)", children=[
            html.Div([
                html.Label("Release Year Range"),
                dcc.RangeSlider(
                    id="tsne-year-slider",
                    min=df["release_year"].min(),
                    max=df["release_year"].max(),
                    step=1,
                    value=[df["release_year"].min(), df["release_year"].max()],
                    marks={str(year): str(year) for year in range(df["release_year"].min(), df["release_year"].max() + 1, 5)},
                    tooltip={"placement": "bottom"}
                ),

                html.Label("Number of Clusters"),
                dcc.Dropdown(
                    id="cluster-count",
                    options=[{"label": str(i), "value": i} for i in range(2, 11)],
                    value=3,
                    clearable=False
                ),

                dcc.Graph(id="tsne-plot", style={"height": "600px"})
            ], style={"padding": "20px"})
        ]),
        
        dcc.Tab(label="Genre Exploration", children=[
            html.Div([
                html.Label("Release Year Range"),
                dcc.RangeSlider(
                    id="genre-year-slider",
                    min=df["release_year"].min(),
                    max=df["release_year"].max(),
                    step=1,
                    value=[df["release_year"].min(), df["release_year"].max()],
                    marks={str(year): str(year) for year in range(df["release_year"].min(), df["release_year"].max() + 1, 5)},
                    tooltip={"placement": "bottom"}
                ),
                
                html.Label("Color By"),
                dcc.Dropdown(
                    id="genre-color-by",
                    options=[
                        {"label": "Average ROI (%)", "value": "roi"},
                        {"label": "Average Profit", "value": "profit"},
                        {"label": "Average Rating", "value": "vote_average"},
                        {"label": "Number of Movies", "value": "count"}
                    ],
                    value="count",
                    clearable=False
                ),
                
                dcc.Graph(id="genre-sunburst", style={"height": "700px"}),
                
                html.Div(id="genre-movie-list", style={"marginTop": "20px"})
            ], style={"padding": "20px"})
        ])
    ])
])

# === Callbacks ===

# Update stored cluster count when user changes it in TSNE tab
@app.callback(
    Output("cluster-count-store", "data"),
    Input("cluster-count", "value")
)
def update_cluster_store(n_clusters):
    return n_clusters

# Budget vs Revenue tab
@app.callback(
    [Output("scatter-plot", "figure"),
     Output("movie-table", "data")],
    [Input("year-slider", "value"),
     Input("color-by", "value"),
     Input("cluster-count-store", "data")]
)
def update_scatter(year_range, color_by, n_clusters):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]
    dff = dff.dropna(subset=["budget", "revenue", "vote_count", color_by] if color_by != "cluster_label" else numeric_features)

    # Apply clustering only if needed
    if color_by == "cluster_label":
        X = dff[numeric_features]
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dff["cluster_label"] = kmeans.fit_predict(X_scaled).astype(str)
    else:
        if "cluster_label" in dff.columns:
            dff = dff.drop(columns=["cluster_label"])

    fig = px.scatter(
        dff,
        x="budget",
        y="revenue",
        color=color_by,
        size="vote_count",
        hover_name="title_y",
        hover_data=["release_year", "budget", "revenue", "profit", "roi", "vote_average"],
        labels={
            "budget": "Budget ($)",
            "revenue": "Revenue ($)",
            "roi": "ROI %",
            "vote_average": "Rating",
            "primary_genre": "Genre",
            "cluster_label": "Cluster"
        },
        title=f"Budget vs Revenue ({year_range[0]}–{year_range[1]})",
        log_x=True,
        log_y=True
    )

    fig.update_layout(
        xaxis_title="Budget (log)",
        yaxis_title="Revenue (log)",
        hovermode="closest"
    )

    table_data = dff.sort_values("revenue", ascending=False).head(500)[[
        "title_y", "release_year", "budget", "revenue", "profit", "roi", "vote_average"
    ]].to_dict("records")

    return fig, table_data

# TSNE tab
@app.callback(
    Output("tsne-plot", "figure"),
    [Input("tsne-year-slider", "value"),
     Input("cluster-count-store", "data")]
)
def update_tsne(year_range, n_clusters):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]
    dff = dff.dropna(subset=numeric_features + ["primary_genre"])

    X = dff[numeric_features]
    X_scaled = StandardScaler().fit_transform(X)

    # TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    components = tsne.fit_transform(X_scaled)
    dff["Dim1"], dff["Dim2"] = components[:, 0], components[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dff["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    fig = px.scatter(
        dff,
        x="Dim1",
        y="Dim2",
        color="cluster",
        hover_name="title_y",
        title=f"TSNE Projection of Movies (k={n_clusters})"
    )
    fig.update_layout(hovermode="closest")
    return fig

# Genre Exploration tab
@app.callback(
    [Output("genre-sunburst", "figure"),
     Output("genre-movie-list", "children")],
    [Input("genre-year-slider", "value"),
     Input("genre-color-by", "value")]
)
def update_genre_sunburst(year_range, color_by):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]
    
    # Create genre hierarchy data for the selected years
    genre_hierarchy = []
    for _, row in dff.iterrows():
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
                    'title': row['title_y'],
                    'budget': row['budget'],
                    'revenue': row['revenue'],
                    'profit': row['profit'],
                    'roi': row['roi'],
                    'vote_average': row['vote_average']
                })
    # for _, row in dff.iterrows():
    #     genres = row['genres_y'].split('-')
    #     if len(genres) > 0:
    #         primary = genres[0]
    #         genre_hierarchy.append({
    #             'ids': primary,
    #             'labels': primary,
    #             'parents': '',
    #             'values': 1,
    #             'roi': row['roi'],
    #             'profit': row['profit'],
    #             'vote_average': row['vote_average']
    #         })
            
    #         if len(genres) > 1:
    #             for sub in genres[1:]:
    #                 genre_hierarchy.append({
    #                     'ids': f"{primary}-{sub}",
    #                     'labels': sub,
    #                     'parents': primary,
    #                     'values': 1,
    #                     'roi': row['roi'],
    #                     'profit': row['profit'],
    #                     'vote_average': row['vote_average']
    #                 })
    
    hierarchy_df = pd.DataFrame(genre_hierarchy)
    
    # Aggregate data for the sunburst
    if color_by == 'count':
        agg_df = hierarchy_df.groupby(['ids', 'labels', 'parents']).size().reset_index(name='count')
        color_col = 'count'
    else:
        agg_df = hierarchy_df.groupby(['ids', 'labels', 'parents']).agg({
            'roi': 'mean',
            'profit': 'mean',
            'vote_average': 'mean'
        }).reset_index()
        color_col = color_by
    
    # Create sunburst chart
    fig = px.sunburst(
        agg_df,
        path=['parents', 'labels'],
        values='count' if color_by == 'count' else agg_df[color_by].abs(),
        color=color_col,
        hover_data=['labels'],
        title=f"Movie Genre Hierarchy ({year_range[0]}-{year_range[1]})",
        branchvalues="total"
    )
    
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        hovermode="closest"
    )
    
    # Create a simple list of movies for the selected genre when clicked
    movie_list = html.Div([
        html.H4("Click on a genre to see movies"),
        html.Ul(id="movie-list-items")
    ])
    
    return fig, movie_list

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
    filtered_movies = filtered_movies.sort_values('revenue', ascending=False).head(20)

    return [
        html.Li([
            html.Strong(movie['title_y']),
            html.Br(),
            f"Year: {movie['release_year']} | Revenue: ${movie['revenue']:,.0f} | ROI: {movie['roi']:.1f}%"
        ]) for _, movie in filtered_movies.iterrows()
    ]

# Run app
if __name__ == '__main__':
    app.run(debug=True)
