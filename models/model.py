from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred


def predict_sales_pre_release(price, average_playtime, achievements, tags, categories, genres, developers, publishers,
                              platforms, positive_ratings_range, negative_ratings_range, english, required_age,
                              release_month, X, model):
    def dynamic_price_weight(price, average_playtime, achievements):
        base_weight = price * 300
        playtime_weight = min(average_playtime / 1000, 2)
        achievement_weight = min(achievements / 50, 1.5)
        return base_weight * (1 + playtime_weight + achievement_weight)

    price_weighted = dynamic_price_weight(price, average_playtime, achievements)
    platform_data = {f'platform_{platform}': 1 if platform in platforms.split(";") else 0 for platform in
                     ['Windows', 'Mac', 'Linux']}

    input_data = pd.DataFrame([{
        'price': price,
        'price_weighted': price_weighted,
        'positive_ratings': np.mean(positive_ratings_range),
        'negative_ratings': np.mean(negative_ratings_range),
        'average_playtime': average_playtime,
        'achievements': achievements,
        'english': english,
        'required_age': required_age,
        'release_month': release_month,
        **{f'tag_{tag}': 1 if tag in tags else 0 for tag in X.columns if tag.startswith('tag_')},
        **{f'cat_{cat}': 1 if cat in categories else 0 for cat in X.columns if cat.startswith('cat_')},
        **{f'genre_{genre}': 1 if genre in genres else 0 for genre in X.columns if genre.startswith('genre_')},
        **{f'dev_{dev}': 1 if dev in developers else 0 for dev in X.columns if dev.startswith('dev_')},
        **{f'pub_{pub}': 1 if pub in publishers else 0 for pub in X.columns if pub.startswith('pub_')},
        **platform_data
    }], columns=X.columns).fillna(0)

    return int(model.predict(input_data)[0])
