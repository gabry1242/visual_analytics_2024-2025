# 🎬 Film Predict-R

Film Predict-R is a data-driven web application that empowers users to explore historical movie data, analyze trends, and simulate success predictions for hypothetical or planned film projects. The application is built using Python's Dash framework and integrates machine learning models to provide actionable recommendations.

## 🎥 Overview

This project uses real-world movie data to:

- Visualize the relationship between key movie attributes (budget, genre, ROI, rating).
- Predict the success probability, expected revenue, and average rating for a movie.
- Recommend how to improve a movie’s potential success based on adjustable inputs.
- Enable genre-based navigation to see how the predicted value compare to other films.

---

## 💻 How It Works

### 1. **Data Preprocessing**
- The dataset original dataset is cleaned to remove missing or duplicate entries.
- The data are then augmented by adding revenue and budget coming from a IMDB dataset.
- New features like `profit`, `ROI`, and `success` are computed.
- Genre data is split and transformed to support both visual and machine learning needs.

### 2. **Feature Engineering**
- Genres are one-hot encoded using `MultiLabelBinarizer`.
- Log-transforms are applied to highly skewed features like budget and revenue.
- Composite features like language type (`is_english`) and runtime are incorporated.

### 3. **Model Training**
Three models are trained using `scikit-learn`:
- 🎯 `RandomForestClassifier`: predicts if a movie will be successful.
- ⭐ `RandomForestRegressor`: predicts the average user rating.
- 💰 `RandomForestRegressor`: predicts average revenue (log-transformed).

All models are trained on a training split and evaluated on held-out test data.

### 4. **Visualization and Interactivity**
The app uses Dash and Plotly for:
- Scatter plots with filtering and zoom.
- Icicle plots for genre hierarchy.
- Dynamic UI components (dropdowns, sliders, prediction inputs).
- Interactive prediction display and sensitivity analysis.

---

## 📷 App Interface Highlights

- **Scatter Plot**: Explore feature relationships (budget vs. revenue, etc.).
- **Genre Explorer**: Dive into nested genres with ROI or rating color schemes.
- **Success Predictor**: Input your movie's specs and get instant predictions.
- **Sensitivity Graph**: See how budget affects success, rating, and revenue.
- **Recommendations**: Dynamic suggestions to improve your project's outcome.

---

## 📁 Project Structure
- `merged_with_tags.csv` → Main input dataset 
- `app.py` → Main Dash application code
- `README.md` → Project overview and documentation
- `requirements.txt` → Python dependencies 

---

## ⚙️ Installation

- Clone this repository
    >git clone https://github.com/gabry1242/visual_analytics_2024-2025

- (Optional) Create a virtual environment
    >On macOS/Linux: python -m venv venv 
    >On Windows: venv\Scripts\activate 
    
- Install dependencies:
    >pip install -r requirements.txt

- Run the app
    >python app.py
    >Then open http://127.0.0.1:8050 in your browser
