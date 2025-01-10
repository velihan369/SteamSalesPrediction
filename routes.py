from flask import Blueprint, render_template, request
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.model import train_model, predict_sales_pre_release
from src.preprocess import preprocess_data
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_genre_vs_ownership
)

import pandas as pd

# Blueprint tanımlama
routes = Blueprint('routes', __name__)

# Dataset yükleme ve preprocess
file_path = "data/steam.csv"
X, y, data = preprocess_data(file_path)

# Model eğitimi
model, X_test, y_test, y_pred = train_model(X, y)

# Model performans metriklerini hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Terminalde çıktı
print("Model Accuracy Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Frontend'e gönderilecek değerler
publishers = sorted(data['publisher_mapped'].unique())
developers = sorted(data['developer_mapped'].unique())
genres = sorted(set(";".join(data['genres'].dropna()).split(";")))
categories = sorted(set(";".join(data['categories'].dropna()).split(";")))
tags = sorted(set(";".join(data['steamspy_tags'].dropna()).split(";")))

@routes.route('/')
def welcome():
    """Ana sayfa"""
    return render_template('welcome.html')

@routes.route('/input', methods=['GET'])
def input_screen():
    """Kullanıcıdan veri girişi alma ekranı"""
    return render_template(
        'input.html',
        publishers=publishers,
        developers=developers,
        genres=genres,
        categories=categories,
        tags=tags
    )

@routes.route('/output', methods=['POST'])
def output_screen():
    """Tahmin sonucunu ve grafikleri gösterme ekranı"""
    # Kullanıcı verilerini toplama
    try:
        price = float(request.form.get('price', 0.0))
        average_playtime = int(request.form.get('average_playtime', 0))
        achievements = int(request.form.get('achievements', 0))
        platforms = request.form.getlist('platforms')
        genres_input = request.form.getlist('genres')
        categories_input = request.form.getlist('categories')
        tags_input = request.form.getlist('tags')
        developers_input = request.form.getlist('developers')
        publishers_input = request.form.getlist('publishers')
        positive_ratings_range = [
            int(request.form.get('positive_min', 0)),
            int(request.form.get('positive_max', 0))
        ]
        negative_ratings_range = [
            int(request.form.get('negative_min', 0)),
            int(request.form.get('negative_max', 0))
        ]
        english = int(request.form.get('english', 1))
        required_age = int(request.form.get('required_age', 0))
        release_month = int(request.form.get('release_month', 1))
    except ValueError:
        return render_template('error.html', message="Invalid input data.")

    # Model tahmini
    predicted_sales = predict_sales_pre_release(
        price=price,
        average_playtime=average_playtime,
        achievements=achievements,
        tags=tags_input,
        categories=categories_input,
        genres=genres_input,
        developers=developers_input,
        publishers=publishers_input,
        platforms=";".join(platforms),
        positive_ratings_range=positive_ratings_range,
        negative_ratings_range=negative_ratings_range,
        english=english,
        required_age=required_age,
        release_month=release_month,
        X=X,
        model=model
    )

    # Grafiklerin çizilmesi
    actual_vs_predicted_path = plot_actual_vs_predicted(y_test, y_pred)
    residuals_path = plot_residuals(y_test, y_pred)
    genre_vs_ownership_path = plot_genre_vs_ownership(data)

    # Tahmini sonuçları ve grafik yollarını output.html sayfasına gönder
    return render_template(
        'output.html',
        predicted_sales=predicted_sales,
        actual_vs_predicted_path="images/actual_vs_predicted.png",
        residuals_path="images/residuals_distribution.png",
        genre_vs_ownership_path="images/genre_vs_ownership.png",
        actual_vs_predicted_comparison_path="images/actual_vs_predicted_comparison.png",
        feature_importances_path="images/feature_importances.png"
    )
