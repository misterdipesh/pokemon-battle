import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("pokemondata.csv", encoding="latin1")
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
df = df.dropna(subset=features + ['name'])

def create_battle_dataset(df):
    battles = []
    labels = []
    for _ in range(5000):
        sample = df.sample(2)
        p1 = sample.iloc[0]
        p2 = sample.iloc[1]
        p1_stats = p1[features].values
        p2_stats = p2[features].values
        battle_input = np.concatenate([p1_stats, p2_stats])
        label = 1 if p1['total_points'] > p2['total_points'] else 0
        battles.append(battle_input)
        labels.append(label)
    return np.array(battles), np.array(labels)

X, y = create_battle_dataset(df)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model.pkl')
