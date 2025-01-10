import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Agg') 

def plot_actual_vs_predicted(y_test, y_pred, output_dir="static/images"):
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Owners")
    plt.xlabel("Actual Owners")
    plt.ylabel("Predicted Owners")
    plt.grid(True)

    # Dosya yolunu belirleyin ve grafiği kaydedin
    file_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_feature_importances(model, feature_names, output_dir="static/images"):
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Özellik önem derecelerini al
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # İlk 20 özelliği göster
    top_features = feature_names[indices][:20]
    top_importances = importances[indices][:20]

    # Grafik oluşturma
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances, align='center')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()

    # Grafiği kaydet
    file_path = os.path.join(output_dir, "feature_importances.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_residuals(y_test, y_pred, output_dir="static/images"):
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    residuals = y_test - y_pred

    # Hata dağılım grafiği
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")

    # Grafiği kaydet
    file_path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_actual_vs_predicted_comparison(y_test, y_pred, output_dir="static/images"):
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Owners")
    plt.legend()

    # Grafiği kaydet
    file_path = os.path.join(output_dir, "actual_vs_predicted_comparison.png")
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_genre_vs_ownership(data, output_dir="static/images"):
    # Klasör oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    genre_ownership = data.groupby("genres")["owners_mean"].mean().sort_values(ascending=False).head(10)

    # Grafik oluşturma
    plt.figure(figsize=(10, 6))
    genre_ownership.plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title("Average Ownership by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Average Ownership")
    plt.xticks(rotation=45)

    # Grafiği kaydet
    file_path = os.path.join(output_dir, "genre_vs_ownership.png")
    plt.savefig(file_path)
    plt.close()

    return file_path
