"""HomeValue engine: Bengaluru-style housing market simulation, price regressors (incl. MLP),
quantile intervals, comparable-sales search, explanations and investment projections."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# locality: (lat, lon, base ₹/sqft, yearly appreciation, rent ₹/sqft/month, tier)
LOCALITIES = {
    "Indiranagar": (12.978, 77.640, 16500, 0.072, 52, "Premium"),
    "Koramangala": (12.935, 77.624, 15200, 0.068, 50, "Premium"),
    "Jayanagar": (12.925, 77.583, 14000, 0.060, 42, "Premium"),
    "Whitefield": (12.970, 77.750, 8600, 0.085, 33, "IT corridor"),
    "Sarjapur Road": (12.900, 77.690, 7900, 0.090, 30, "IT corridor"),
    "Electronic City": (12.845, 77.660, 6100, 0.075, 25, "IT corridor"),
    "Hebbal": (13.035, 77.597, 11200, 0.082, 36, "Emerging"),
    "Yelahanka": (13.100, 77.596, 7200, 0.088, 24, "Emerging"),
    "HSR Layout": (12.912, 77.638, 12800, 0.070, 44, "Premium"),
    "JP Nagar": (12.905, 77.585, 10400, 0.062, 34, "Established"),
    "Marathahalli": (12.956, 77.701, 8900, 0.071, 33, "IT corridor"),
    "Hennur": (13.045, 77.640, 7600, 0.087, 26, "Emerging"),
    "Rajajinagar": (12.991, 77.552, 11800, 0.055, 37, "Established"),
    "Banashankari": (12.925, 77.546, 9200, 0.058, 30, "Established"),
}
NUM = ["sqft", "bhk", "bath", "age", "floor", "total_floors", "metro_km", "parking", "amenity_score"]
CAT = ["locality", "type", "furnishing", "facing"]
FEATURES = NUM + CAT


def generate(n=6000, seed=8):
    rng = np.random.default_rng(seed)
    loc = rng.choice(list(LOCALITIES), n)
    typ = rng.choice(["Apartment", "Villa", "Independent House"], n, p=[0.72, 0.1, 0.18])
    bhk = np.clip(rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.38, 0.36, 0.12, 0.04]) + (typ != "Apartment"), 1, 6)
    sqft = np.clip(bhk * rng.normal(520, 70, n) + rng.normal(180, 120, n) + 500 * (typ == "Villa"), 380, 6500).round(-1)
    bath = np.clip(bhk + rng.choice([-1, 0, 0, 1], n), 1, 6)
    age = np.clip(rng.gamma(1.6, 5, n), 0, 40).round()
    tot = np.where(typ == "Apartment", rng.integers(4, 33, n), rng.integers(1, 4, n))
    floor = np.array([rng.integers(0, t + 1) for t in tot])
    metro = np.clip(rng.gamma(2, 1.4, n), 0.2, 15).round(1)
    park = np.clip(rng.choice([0, 1, 1, 2], n) + (bhk >= 4), 0, 3)
    amen = np.clip(rng.normal(6, 2, n) + 1.5 * (typ == "Apartment"), 0, 10).round(1)
    furn = rng.choice(["Unfurnished", "Semi-furnished", "Fully furnished"], n, p=[0.45, 0.4, 0.15])
    facing = rng.choice(["East", "North", "West", "South", "North-East"], n)
    base = np.array([LOCALITIES[l][2] for l in loc])
    psf = (base * (1 + 0.18 * (typ == "Villa") + 0.06 * (typ == "Independent House")) * (1 - 0.009 * age) * (1 - 0.018 * np.minimum(metro, 8)) * (1 + 0.015 * (amen - 6))
           * (1 + 0.006 * np.minimum(floor, 15)) * (1 + 0.05 * (furn == "Semi-furnished") + 0.11 * (furn == "Fully furnished")) * (1 + 0.03 * (facing == "East") + 0.04 * (facing == "North-East"))
           * (1 - 0.00002 * (sqft - 1300)) * rng.lognormal(0, 0.09, n))
    price = (psf * sqft + park * 250000).round(-4)
    lat = np.array([LOCALITIES[l][0] for l in loc]) + rng.normal(0, 0.008, n)
    lon = np.array([LOCALITIES[l][1] for l in loc]) + rng.normal(0, 0.008, n)
    return pd.DataFrame({"locality": loc, "type": typ, "bhk": bhk, "sqft": sqft, "bath": bath, "age": age, "floor": floor, "total_floors": tot, "metro_km": metro, "parking": park,
                         "amenity_score": amen, "furnishing": furn, "facing": facing, "price": price, "price_psf": (price / sqft).round(), "lat": lat.round(5), "lon": lon.round(5)})


def _pre():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


MODELS = {
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "Ridge (log-linear)": lambda: Ridge(alpha=1.0),
    "Neural Net (MLP)": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500, early_stopping=True, random_state=0),
}


def train(df):
    X, y = df[FEATURES], np.log(df["price"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=4)
    out = {"models": {}, "yte": np.exp(yte.values)}
    for n, f in MODELS.items():
        p = Pipeline([("pre", _pre()), ("m", f())]).fit(Xtr, ytr)
        pr = np.exp(p.predict(Xte))
        out["models"][n] = {"pipe": p, "pred": pr, "MAE ₹L": mean_absolute_error(np.exp(yte), pr) / 1e5, "MAPE %": mean_absolute_percentage_error(np.exp(yte), pr) * 100, "R²": r2_score(np.exp(yte), pr)}
    out["q"] = {a: Pipeline([("pre", _pre()), ("m", HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, learning_rate=0.06, random_state=0))]).fit(Xtr, ytr) for a in (0.1, 0.9)}
    pre = _pre().fit(X)
    nn = NearestNeighbors(n_neighbors=6).fit(pre.transform(X))
    out["knn"] = (pre, nn)
    return out


def predict(R, name, row):
    X = pd.DataFrame([row])[FEATURES]
    p = float(np.exp(R["models"][name]["pipe"].predict(X)[0]))
    lo, hi = float(np.exp(R["q"][0.1].predict(X)[0])), float(np.exp(R["q"][0.9].predict(X)[0]))
    return p, min(lo, p * 0.97), max(hi, p * 1.03)


def comparables(R, df, row, k=6):
    pre, nn = R["knn"]
    _, idx = nn.kneighbors(pre.transform(pd.DataFrame([row])[FEATURES]), n_neighbors=k)
    return df.iloc[idx[0]]


def contributions(R, row):
    pipe = R["models"]["Ridge (log-linear)"]["pipe"]
    pre, m = pipe.named_steps["pre"], pipe.named_steps["m"]
    x = pre.transform(pd.DataFrame([row])[FEATURES])[0]
    names = pre.get_feature_names_out()
    c = pd.Series(x * m.coef_, index=names)
    agg = {}
    for k, v in c.items():
        base = k.split("__", 1)[1]
        key = next((cc for cc in CAT if base.startswith(cc + "_")), base)
        if key in CAT and not base.endswith("_" + str(row[key])):
            continue
        label = key.replace("_", " ") + (f": {row[key]}" if key in CAT else "")
        agg[label] = agg.get(label, 0) + v
    s = (np.exp(pd.Series(agg)) - 1) * 100  # % effect on price
    return s[s.abs() > 0.5].sort_values(key=abs, ascending=False)


def fmt(v):
    return f"₹{v / 1e7:.2f} Cr" if v >= 1e7 else f"₹{v / 1e5:.1f} L"


def emi(P, rate, years):
    r, n = rate / 1200, years * 12
    return P * r * (1 + r) ** n / ((1 + r) ** n - 1)
