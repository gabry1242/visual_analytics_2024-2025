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


##GRAPH CREATION AND DATA PREPARATION###

def build_full_graph(network_data):
    G = nx.Graph()
    for node in network_data['nodes']:
        G.add_node(node['id'], **node)
    for link in network_data['links']:
        G.add_edge(link['source'], link['target'], weight=link['value'])
    return G

# Get subgraph nodes connected to all selected tags (intersection of neighbors)
def get_expanded_subgraph(G, selected_tags):
    if not selected_tags:
        # Return empty graph or full graph if you prefer
        return nx.Graph()
    
    # Use union instead of intersection to expand from all selected tags
    expanded_nodes = set(selected_tags)
    for tag in selected_tags:
        if tag in G:
            expanded_nodes.update(G.neighbors(tag))
    
    return G.subgraph(expanded_nodes).copy()


def prune_edges_to_hierarchy(G, pos):
    import networkx as nx
    from collections import defaultdict

    layers = defaultdict(list)
    for node, (x, y) in pos.items():
        layers[x].append(node)

    sorted_layers = sorted(layers.keys())

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for i, layer_x in enumerate(sorted_layers):
        nodes_in_layer = layers[layer_x]

        if i == 0:
            # First layer, no incoming edges
            continue

        prev_layer_x = sorted_layers[i - 1]
        nodes_in_prev_layer = layers[prev_layer_x]

        for node in nodes_in_layer:
            neighbors_in_prev = [nbr for nbr in G.neighbors(node) if nbr in nodes_in_prev_layer]

            if not neighbors_in_prev:
                continue

            # Choose edge with highest weight
            best_nbr = max(
                neighbors_in_prev,
                key=lambda nbr: G.edges[node, nbr].get('weight', 1)
            )

            H.add_edge(node, best_nbr, **G.edges[node, best_nbr])

    return H
# Create figure from subgraph and highlight selected tags
def create_network_figure(G, selected_tags, color_metric):
    if len(G) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Select tag(s) to start",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    def multi_level_hierarchical_layout(G, tag_path, level_distance=300, vertical_gap=50):
        from collections import deque

        pos = {}
        level = 0
        visited = set()
        canvas_height = 700
        # for i, current_tag in enumerate(tag_path):
        #     if current_tag not in G:
        #         continue
        #     # Place the current tag
        #     pos[current_tag] = (level * level_distance, 0)
        #     visited.add(current_tag)
            
        #     # Expand to neighbors at next level
        #     neighbors = [n for n in G.neighbors(current_tag) if n not in visited]
        #     for j, neighbor in enumerate(sorted(neighbors)):
        #         y_offset = j * vertical_gap - (len(neighbors) * vertical_gap / 2)
        #         print(j, neighbor)
        #         #y_offset = ((j + 0.5) / num_neighbors) * canvas_height - (canvas_height / 2)
        #         pos[neighbor] = ((len(tag_path) + 1) * level_distance, y_offset)
        #         visited.add(neighbor)

        #     level += 1
        for i, current_tag in enumerate(tag_path):
            if current_tag not in G:
                continue
            pos[current_tag] = (i * level_distance, 0)
            visited.add(current_tag)

        # Collect unique neighbors that are not in selected tags
        all_neighbors = set()
        for tag in tag_path:
            if tag not in G:
                continue
            neighbors = set(G.neighbors(tag))
            all_neighbors.update(neighbors)

        # Remove already visited nodes (e.g., selected tags)
        last_layer_nodes = sorted(all_neighbors - visited)
        num_neighbors = len(last_layer_nodes)

        # Distribute them vertically in canvas_height
        canvas_height = 700
        vertical_step = canvas_height / (num_neighbors + 1)

        for j, neighbor in enumerate(last_layer_nodes):
            y_offset = (j + 1) * vertical_step - (canvas_height / 2)
            pos[neighbor] = ((len(tag_path)) * level_distance, y_offset)
            visited.add(neighbor)

        return pos
    

    if selected_tags and all(tag in G for tag in selected_tags):
        pos = multi_level_hierarchical_layout(G, tag_path=selected_tags)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    pruned_G = prune_edges_to_hierarchy(G, pos)

    edge_x, edge_y = [], []
    for edge in pruned_G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color, node_size, line_colors = [], [], [], [], [], []
    for node in pruned_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = pruned_G.nodes[node]
        node_text.append(
            f"<b>{node}</b><br>Frequency: {node_data['frequency']}<br>"
            f"Avg ROI: {node_data['avg_roi']:.2f}<br>"
            f"Avg Profit: {node_data['avg_profit']:.2f}<br>"
            f"Avg Rating: {node_data['avg_rating']:.2f}"
        )
        color_value = node_data.get(color_metric, 0)
        node_color.append(color_value)
        node_size.append(20)
        line_colors.append('red' if node in selected_tags else 'DarkSlateGrey')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in pruned_G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,
            size=node_size,
            colorbar=dict(title=color_metric.replace('_', ' ').title()),
            line_width=3,
            line_color=line_colors
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Tag Relationship Network (Tree View)',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        transition={'duration': 500}
    )

    return fig


def preprocess_tag_network(df):
    # Flatten all tags & count frequency
    df['tag_list'] = df['tag'].fillna('').str.lower().str.split(';').apply(lambda tags: [t.strip() for t in tags if t.strip()])

    all_tags = df['tag_list'].explode()
    tag_counts = all_tags.value_counts()

    # Keep only popular tags
    popular_tags = tag_counts[tag_counts >= 300].index.tolist()

    # Filter df rows to only include tags in popular_tags
    df['filtered_tags'] = df['tag_list'].apply(lambda tags: [t for t in tags if t in popular_tags])

    # Build co-occurrence counts efficiently
    from collections import Counter
    cooc_counter = Counter()

    for tags in df['filtered_tags']:
        unique_tags = list(set(tags))
        for i in range(len(unique_tags)):
            for j in range(i + 1, len(unique_tags)):
                pair = tuple(sorted([unique_tags[i], unique_tags[j]]))
                cooc_counter[pair] += 1

    # Build nodes with aggregated metrics
    nodes = []
    for tag in popular_tags:
        tag_movies = df[df['filtered_tags'].apply(lambda tags: tag in tags)]
        nodes.append({
            'id': tag,
            'label': tag,
            'frequency': tag_counts[tag],
            'avg_roi': tag_movies['roi'].mean() if not tag_movies.empty else 0,
            'avg_profit': tag_movies['profit'].mean() if not tag_movies.empty else 0,
            'avg_rating': tag_movies['vote_average'].mean() if not tag_movies.empty else 0,
        })

    # Build links
    links = [{'source': src, 'target': tgt, 'value': count} for (src, tgt), count in cooc_counter.items()]

    return {'nodes': nodes, 'links': links}

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
app.title = "Movie Analytics Dashboard"
network_data = preprocess_tag_network(df)

app.layout = html.Div([
    html.H1("Movie Financial Analytics", style={"textAlign": "center"}),
    dcc.Store(id="cluster-count-store", data=3),  # default number of clusters
    dcc.Store(id='network-data-store', data=network_data),
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

        dcc.Tab(label="Dimensionality Reduction (PCA)", children=[
            html.Div([
                html.Label("Release Year Range"),
                dcc.RangeSlider(
                    id="pca-year-slider",
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

                dcc.Graph(id="pca-plot", style={"height": "600px"})
            ], style={"padding": "20px"})
        ]),
        
        dcc.Tab(label="Genre Exploration", children=[
            html.Div([
                html.Div([  # Controls section
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
                ], style={"width": "100%", "marginBottom": "20px"}),
                
                # Visualizations section
        html.Div([
            # Sunburst graph (fills space)
            html.Div(
                dcc.Graph(
                    id="genre-sunburst",
                    config={"responsive": True},  # Makes graph fill container
                    style={
                        "height": "100%",  # Fills parent div
                        "width": "100%",   # Fills parent div
                    }
                ),
                style={
                    "flex": "3",          # Takes 75% of width
                    "minHeight": "600px",  # Dynamic but constrained
                    "maxHeight": "700px", # Preoversizing
                    "padding": "0px",     # Remove padding (can add back if needed)
                }
            ),
            
            # Movie list (fixed width)
            html.Div(
                html.Div(id="genre-movie-list"),
                style={
                    "flex": "1",          # Takes 25% of width
                    "overflowY": "auto",  # Scroll if content overflows
                    "minHeight": "600px",  # Match sunburst height
                    "maxHeight": "700px",
                    "padding": "10px",
                }
            )
        ], style={
            "display": "flex",
            "gap": "10px",
            "width": "100%",
            "height": "auto",  # Prevents container from expanding
        })


            ], style={"padding": "20px"})
        ]),
        #Success Predictor tab
        dcc.Tab(label="Success Predictor", children=[
            html.Div([
                html.H2("Movie Success Prediction Tool", style={'textAlign': 'center'}),

                # Main container - 2 columns: left for input/results, right for graph
                html.Div([
                    # Left Column
                    html.Div([
                        # Inputs (Top Left)
                        html.Div([
                            html.Div([
                                # Row 1: Budget + Runtime
                                html.Div([
                                    html.Div([
                                        html.Label("Budget (in millions)"),
                                        dcc.Input(id='budget-input', type='number', value=50, min=1, step=1, style={'width': '100%'})
                                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),

                                    html.Div([
                                        html.Label("Runtime (minutes)"),
                                        dcc.Input(id='runtime-input', type='number', value=120, min=60, max=240, step=5, style={'width': '100%'})
                                    ], style={'width': '48%', 'display': 'inline-block'})
                                ], style={'marginBottom': '10px'}),

                                # Row 2: Language + Year
                                html.Div([
                                    html.Div([
                                        html.Label("Original Language"),
                                        dcc.Dropdown(
                                            id='language-input',
                                            options=[{'label': 'English', 'value': 1},
                                                    {'label': 'Non-English', 'value': 0}],
                                            value=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),

                                    html.Div([
                                        html.Label("Release Year"),
                                        dcc.Input(id='year-input', type='number', value=2023, min=1900, max=2030, step=1, style={'width': '100%'})
                                    ], style={'width': '48%', 'display': 'inline-block'})
                                ], style={'marginBottom': '10px'}),

                                # Row 3: Genres full width
                                html.Div([
                                    html.Label("Genres"),
                                    dcc.Dropdown(
                                        id='genres-input',
                                        options=[{'label': genre, 'value': genre} for genre in all_genres],
                                        multi=True,
                                        value=['Action'],
                                        style={'width': '100%'}
                                    )
                                ], style={'marginBottom': '15px'}),

                                # Predict button
                                html.Button('Predict Success', id='predict-button', n_clicks=0,
                                            style={'background-color': '#4CAF50', 'color': 'white', 'width': '100%', 'padding': '10px'})
                            ])
                        ], style={'marginBottom': '20px'}),

                        
                        # Results + Recommendations (Bottom Left)
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
                            html.H3("Recommendations for Improvement", style={'marginTop': '20px'}),
                            html.Div(id='recommendations', style={
                                'padding': '10px',
                                'backgroundColor': '#e9ecef',
                                'borderRadius': '5px'
                            })
                        ])
                    ], style={'width': '40%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    # Right Column: Sensitivity Plot
                    html.Div([
                        dcc.Graph(id='sensitivity-plot', style={'height': '100%', 'width': '100%'})
                    ], style={'width': '58%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            ], style={'padding': '20px', 'height': '100vh', 'boxSizing': 'border-box', 'overflow': 'hidden'})
        ]),
        dcc.Tab(label="Tag Network Analysis", children=[
            html.Div([
                html.H2("Movie Tag Network Visualization"),

                html.Label("Select Tags:"),
                dcc.Dropdown(
                    id='selected-tags-dropdown',
                    options=[],  # start empty, filled by callback
                    multi=True,
                    style={"width": "100%"}
                ),

                html.Br(),

                html.Label("Node Color Metric:"),
                dcc.Dropdown(
                    id='node-color-metric',
                    options=[
                        {"label": "Frequency", "value": "frequency"},
                        {"label": "Average ROI", "value": "avg_roi"},
                        {"label": "Average Profit", "value": "avg_profit"},
                        {"label": "Average Rating", "value": "avg_rating"},
                    ],
                    value="frequency",
                    clearable=False,
                ),

                dcc.Graph(id='tag-network-graph', style={'height': '700px'}),

                html.Div(id='genre-distribution', style={"marginTop": "20px"})
            ])
        ])
    ])#close all the tabs
])






# === Callbacks ===

# Update stored cluster count when user changes it in PCA tab
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

# PCA tab
@app.callback(
    Output("pca-plot", "figure"),
    [Input("pca-year-slider", "value"),
     Input("cluster-count-store", "data")]
)
def update_pca(year_range, n_clusters):
    dff = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]
    dff = dff.dropna(subset=numeric_features + ["primary_genre"])

    X = dff[numeric_features]
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    dff["PC1"], dff["PC2"] = components[:, 0], components[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dff["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    fig = px.scatter(
        dff,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name="title_y",
        title=f"PCA Projection of Movies (k={n_clusters})"
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
     Output('sensitivity-plot', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('budget-input', 'value'),
     State('runtime-input', 'value'),
     State('language-input', 'value'),
     State('year-input', 'value'),
     State('genres-input', 'value')]
)
def update_predictions(n_clicks, budget, runtime, language, year, genres):
    if n_clicks == 0:
        return "", "", "", "", "", go.Figure()
    
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
    
    return success_output, rating_output, revenue_output, roi_output, recommendations, fig

def generate_recommendations(budget, runtime, language, year, genres, 
                           success_prob, rating_pred, roi):
    recommendations = []
    
    # Check if we can improve rating
    if rating_pred < 7.0:
        # Try adding highly rated genres
        high_rating_genres = ['Drama', 'Biography', 'History', 'Crime']
        new_genres = list(set(genres + [g for g in high_rating_genres if g not in genres][:1]))
        if len(new_genres) > len(genres):
            test_input = prepare_input(budget, runtime, language, year, new_genres)
            new_rating = rating_model.predict(test_input)[0]
            if new_rating > rating_pred + 0.3:
                recommendations.append(
                    html.Li(f"🎭 Consider adding {new_genres[-1]} genre to increase predicted rating to {new_rating:.1f}")
                )
    
    # Check if we can improve ROI
    if roi < 3.0:
        # Try reducing budget
        new_budget = budget * 0.8
        test_input = prepare_input(new_budget, runtime, language, year, genres)
        new_revenue = np.expm1(revenue_model.predict(test_input)[0])
        new_roi = new_revenue / (new_budget * 1000000)
        if new_roi > roi * 1.2:
            recommendations.append(
                html.Li(f"💵 Reducing budget to ${new_budget:.1f} million could improve ROI to {new_roi:.1f}x")
            )
    
    # Check if we can improve success probability
    if success_prob < 0.7:
        # Try adjusting runtime
        optimal_runtime = 110  # Based on data analysis
        if abs(runtime - optimal_runtime) > 15:
            test_input = prepare_input(budget, optimal_runtime, language, year, genres)
            new_success_prob = success_model.predict_proba(test_input)[0][1]
            if new_success_prob > success_prob + 0.1:
                recommendations.append(
                    html.Li(f"⏱️ Adjusting runtime to {optimal_runtime} minutes could increase success probability to {new_success_prob:.1%}")
                )
    
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


###CALLBACK FOR GRAPH OF TAGS###
@app.callback(
    Output('selected-tags-dropdown', 'options'),
    Input('network-data-store', 'data')
)
def update_dropdown_options(network_data):
    nodes = network_data.get('nodes', [])
    options = [{'label': node['label'], 'value': node['id']} for node in nodes]
    return options

@app.callback(
    Output('tag-network-graph', 'figure'),
    Input('selected-tags-dropdown', 'value'),
    Input('node-color-metric', 'value'),  
    State('network-data-store', 'data')
)
def update_graph(selected_tags, color_metric, network_data):
    if not selected_tags:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Select tag(s) to start",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Filter dataframe to movies containing all selected tags
    selected_tags_set = set(selected_tags)
    
    # df['filtered_tags'] is a list of tags per movie (preprocessed)
    filtered_df = df[df['filtered_tags'].apply(lambda tags: selected_tags_set.issubset(set(tags)))]

    if filtered_df.empty:
        # No movies with all selected tags, return empty graph
        fig = go.Figure()
        fig.update_layout(title="No movies found with all selected tags.")
        return fig

    # Get all tags in filtered movies (flatten)
    all_tags_in_filtered = filtered_df['filtered_tags'].explode().value_counts()

    # Build nodes only for these tags
    nodes = []
    for tag, freq in all_tags_in_filtered.items():
        tag_movies = filtered_df[filtered_df['filtered_tags'].apply(lambda tags: tag in tags)]
        nodes.append({
            'id': tag,
            'label': tag,
            'frequency': freq,
            'avg_roi': tag_movies['roi'].mean(),
            'avg_profit': tag_movies['profit'].mean(),
            'avg_rating': tag_movies['vote_average'].mean()
        })

    # Build co-occurrence in filtered_df (same as before but only for filtered movies)
    from collections import Counter
    cooc_counter = Counter()
    for tags in filtered_df['filtered_tags']:
        unique_tags = list(set(tags))
        for i in range(len(unique_tags)):
            for j in range(i + 1, len(unique_tags)):
                pair = tuple(sorted([unique_tags[i], unique_tags[j]]))
                cooc_counter[pair] += 1

    links = [{'source': src, 'target': tgt, 'value': count} for (src, tgt), count in cooc_counter.items()]

    # Build graph
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'], **node)
    for link in links:
        G.add_edge(link['source'], link['target'], weight=link['value'])


    fig = create_network_figure(G, selected_tags, color_metric)
    return fig

# Run app
if __name__ == "__main__":
    app.run(debug=True)