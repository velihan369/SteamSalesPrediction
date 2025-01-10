import pandas as pd
import numpy as np

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data['owners_mean'] = data['owners'].str.split('-').apply(
        lambda x: (int(x[0]) + int(x[1])) // 2 if len(x) == 2 else 0
    )

    top_25_developers = data['developer'].value_counts().nlargest(25).index
    top_25_publishers = data['publisher'].value_counts().nlargest(25).index

    data['developer_mapped'] = data['developer'].apply(lambda dev: dev if dev in top_25_developers else 'Other')
    data['publisher_mapped'] = data['publisher'].apply(lambda pub: pub if pub in top_25_publishers else 'Other')

    data = pd.concat([
        data,
        data['developer_mapped'].str.get_dummies().add_prefix('dev_'),
        data['publisher_mapped'].str.get_dummies().add_prefix('pub_')
    ], axis=1)

    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    data['release_month'] = data['release_date'].dt.month.fillna(0).astype(int)
    data.fillna(0, inplace=True)

    data['price_weighted'] = data.apply(
        lambda row: dynamic_price_weight(row['price'], row['average_playtime'], row['achievements']),
        axis=1
    )

    data['positive_ratings'] *= 0.005
    data['negative_ratings'] *= 0.005

    features = [
        'price',
        'price_weighted',
        'positive_ratings',
        'negative_ratings',
        'average_playtime',
        'achievements',
        'english',
        'required_age',
        'release_month'
    ] + [
        col for col in data.columns if col.startswith('tag_') or col.startswith('cat_') or col.startswith('genre_') or col.startswith('dev_') or col.startswith('pub_')
    ]
    target = 'owners_mean'

    return data[features], data[target], data

def dynamic_price_weight(price, average_playtime, achievements):
    base_weight = price * 300
    playtime_weight = min(average_playtime / 1000, 2)
    achievement_weight = min(achievements / 50, 1.5)
    return base_weight * (1 + playtime_weight + achievement_weight)
