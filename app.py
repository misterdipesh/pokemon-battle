from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import random
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
df = pd.read_csv("pokemondata.csv", encoding="latin1")
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    names = df["name"].dropna().unique()
    if request.method == "POST":
        mode = request.form.get("mode")
        p1_name = request.form.get("p1")
        p2_name = request.form.get("p2") if mode == "multi" else random.choice(names)
        
        p1 = df[df["name"] == p1_name][features].values[0]
        p2 = df[df["name"] == p2_name][features].values[0]
        
        input_data = np.concatenate([p1, p2]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        winner = p1_name if prediction == 1 else p2_name
        result = f"{p1_name} vs {p2_name} → Winner: {winner}"
        
    return render_template("index.html", names=sorted(names), result=result)

if __name__ == "__main__":
    print("server starting")
    app.run(debug=True, host='127.0.0.1', port=5000)