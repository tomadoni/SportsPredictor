import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, log_loss, brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt

# NBA
# --- Load Data ---
matchup_df = pd.read_csv("NBA_Matchup_Training_Data_with_Home.csv")
team_stats = pd.read_csv("Full_NBA_Team_Stats.csv")

# --- Flip and Scale Home Indicator ---
matchup_df['home_indicator'] *= -3.0  # Flipping the sign and amplifying

# --- Define Features ---
raw_features = ['FG%', '3P%', 'REB', '3P%_def', 'REB_def']
diff_features = ['diff_' + f for f in raw_features]
model_features = diff_features + ['home_indicator']

# --- Separate features ---
X_stats = matchup_df[diff_features]
X_home = matchup_df[['home_indicator']]
y = matchup_df["Winner"]

# --- Scale stat features only ---
scaler = StandardScaler()
X_stats_scaled = scaler.fit_transform(X_stats)

# --- Combine scaled stats with unscaled home_indicator ---
X_combined = np.hstack([X_stats_scaled, X_home.values])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Calibrated Logistic Regression Model ---
lr = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs', random_state=42)
model = CalibratedClassifierCV(estimator=lr, method='sigmoid', cv=5)
model.fit(X_train, y_train)

# --- Extract Feature Importance ---
coef = model.calibrated_classifiers_[0].estimator.coef_[0]
feature_importance_df = pd.DataFrame({
    "Feature": model_features,
    "Coefficient": coef
}).sort_values(by="Coefficient", ascending=False)

# --- Model Evaluation ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
logloss = log_loss(y_test, model.predict_proba(X_test))
brier = brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# --- Streamlit App ---
st.title("🏀 NBA Matchup Predictor")
st.markdown("Predict NBA matchups using team stat differentials and home court edge.")

teams = sorted(team_stats["Team"].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", teams)
with col2:
    away_team = st.selectbox("✈️ Away Team", teams)

if home_team != away_team:
    try:
        home_stats = team_stats[team_stats["Team"] == home_team][raw_features].values[0]
        away_stats = team_stats[team_stats["Team"] == away_team][raw_features].values[0]
        diff_vector = home_stats - away_stats
        clipped_vector = np.clip(diff_vector, -1.1, 1.1)

        # Scale only the 5 features (not home indicator)
        diff_scaled = scaler.transform([clipped_vector])[0]

        # ✅ Manually append exaggerated home advantage
        final_vector = np.append(diff_scaled, 25)

        probs = model.predict_proba([final_vector])[0]
        prob_home = probs[1]
        prob_away = probs[0]
        predicted_winner = home_team if prob_home > prob_away else away_team

        st.subheader("📈 Prediction")
        st.success(f"**Predicted Winner: {predicted_winner}**")
        st.metric(f"{home_team} Win Probability", f"{prob_home * 100:.1f}%")
        st.metric(f"{away_team} Win Probability", f"{prob_away * 100:.1f}%")

    except IndexError:
        st.error("Team stats not found.")
else:
    st.warning("Please select two different teams.")


# NHL

# --- Load Data ---
df = pd.read_csv("NHL_Matchup_Training_Data (1).csv")

# --- Features ---
raw_features = ['GF/G', 'S%', 'PP%', 'PIM', 'SOG', 'GA/G', 'SV%', 'PK%']
diff_features = ['diff_' + f for f in raw_features]
model_features = diff_features + ['home_indicator']

# --- Separate features ---
X_stats = df[diff_features]
X_home = df[['home_indicator']]
y = df["Winner"]

# --- Scale stat features only ---
scaler = StandardScaler()
X_stats_scaled = scaler.fit_transform(X_stats)

# --- Combine scaled stats with raw home_indicator ---
X_combined = np.hstack([X_stats_scaled, X_home.values])

# --- Train/Calibrate Split ---
X_train, X_calib, y_train, y_calib = train_test_split(
    X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# --- Logistic Regression Model ---
lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
calibrated_model = CalibratedClassifierCV(estimator=lr, method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred_train = calibrated_model.predict(X_train)
y_pred_test = calibrated_model.predict(X_calib)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_calib, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_calib, y_pred_test)
logloss = log_loss(y_calib, calibrated_model.predict_proba(X_calib))
brier = brier_score_loss(y_calib, calibrated_model.predict_proba(X_calib)[:, 1])
auc = roc_auc_score(y_calib, calibrated_model.predict_proba(X_calib)[:, 1])

# --- Teams ---
teams = sorted(set(df["home_team"]).union(df["away_team"]))

# --- Streamlit UI ---
st.title("🏒 NHL Matchup Predictor")
st.markdown("Predict NHL matchups using stat differentials and home-ice advantage.")

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("🏠 Home Team (NHL)", teams)
with col2:
    away = st.selectbox("✈️ Away Team (NHL)", teams)

if home != away:
    matchup = df[(df["home_team"] == home) & (df["away_team"] == away)]
    if matchup.empty:
        st.error("Not enough data for this matchup.")
        st.stop()

    # Clip stat diff and scale
    stat_diff = np.clip(matchup[diff_features].iloc[0].values, -0.05, 0.05)
    input_scaled = scaler.transform([stat_diff])

    # ✅ Multiply home_indicator by 25 during prediction
    home_indicator = matchup['home_indicator'].iloc[0] * -20
    input_combined = np.hstack([input_scaled, [[home_indicator]]])

    # Predict
    prob = calibrated_model.predict_proba(input_combined)[0]
    prob_home = prob[1]
    prob_away = prob[0]
    winner = home if prob_home > prob_away else away

    # Display results
    st.subheader("📈 Prediction Result")
    st.success(f"**Predicted Winner: {winner}**")
    st.metric(f"{home} Win Probability", f"{prob_home * 100:.1f}%")
    st.metric(f"{away} Win Probability", f"{prob_away * 100:.1f}%")
else:
    st.warning("Please select two different teams.")

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # sigmoid

# =========================
# Config
# =========================
FEATURES = ["OPS", "OBP", "SLG", "ERA", "WHIP", "SO", "BB"]
CLIP_MIN, CLIP_MAX = -3.0, 3.0
# Positive value nudges predictions toward home team.
# ~0.20 ≈ about +5 percentage points near 50/50.
HOME_LOGIT_BONUS = 0.20

# =========================
# Load Data
# =========================
stats = (
    pd.read_csv("MLB_Run_Differential_Clean_FIXED.csv")
      .apply(pd.to_numeric, errors="coerce")
      .dropna()
)
games = pd.read_csv("mlb-2025-asplayed.csv", encoding="ISO-8859-1")

# Teams list: union of Home & Away
teams = sorted(pd.unique(pd.concat([games["Home"], games["Away"]], ignore_index=True).dropna()))

# =========================
# Build Per-Team Stats (mean across rows)
# Assumes stats rows align with games rows for home_/away_ columns.
# =========================
stat_rows = []
n = min(len(stats), len(games))
for i in range(n):
    home_team = games.iloc[i]["Home"]
    away_team = games.iloc[i]["Away"]
    try:
        home_vals = [stats.iloc[i][f"home_{f}"] for f in FEATURES]
        away_vals = [stats.iloc[i][f"away_{f}"] for f in FEATURES]
    except KeyError:
        continue
    stat_rows.append([home_team] + home_vals)
    stat_rows.append([away_team] + away_vals)

df_team_stats = pd.DataFrame(stat_rows, columns=["Team"] + FEATURES)
team_stats = df_team_stats.groupby("Team", as_index=True).mean()

# =========================
# Build Training Matrix (home - away) and Fit
# =========================
games = games.copy()
games["home_win"] = (games["Home Score"] > games["Away Score"]).astype(int)

rows, labels = [], []
for _, g in games.iterrows():
    home = g.get("Home")
    away = g.get("Away")
    if home not in team_stats.index or away not in team_stats.index:
        continue
    h = team_stats.loc[home]
    a = team_stats.loc[away]
    diff = [h[f] - a[f] for f in FEATURES]  # home - away
    rows.append(diff)
    labels.append(g["home_win"])

diff_features = [f"diff_{f}" for f in FEATURES]
X_df = pd.DataFrame(rows, columns=diff_features)
y = pd.Series(labels, name="home_win")

# Clip BEFORE scaler to keep train/serve consistent
X_clipped = X_df.clip(lower=CLIP_MIN, upper=CLIP_MAX)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clipped.values)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_scaled, y)

# =========================
# Streamlit UI (no sidebar)
# =========================
st.title("⚾ MLB Matchup Predictor")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", teams, index=0 if teams else None)
with col2:
    away_team = st.selectbox("✈️ Away Team", teams, index=1 if len(teams) > 1 else None)

if home_team and away_team:
    if home_team == away_team:
        st.warning("Please select two different teams.")
    elif home_team not in team_stats.index or away_team not in team_stats.index:
        st.error("❌ Could not find team stats for one or both teams.")
    else:
        # Build (home - away) features for the selected matchup
        h = team_stats.loc[home_team]
        a = team_stats.loc[away_team]
        diff = np.array([h[f] - a[f] for f in FEATURES], dtype=float)
        clipped = np.clip(diff, CLIP_MIN, CLIP_MAX)
        scaled = scaler.transform([clipped])[0]

        # Model log-odds for home win + fixed home advantage bump
        home_logit = model.decision_function([scaled])[0] + HOME_LOGIT_BONUS
        prob_home = float(expit(home_logit))
        prob_away = 1.0 - prob_home
        winner = home_team if prob_home >= prob_away else away_team

        st.subheader("📈 Prediction")
        st.success(f"**Predicted Winner: {winner}**")
        c1, c2 = st.columns(2)
        c1.metric(f"{home_team} Win Probability", f"{prob_home * 100:.1f}%")
        c2.metric(f"{away_team} Win Probability", f"{prob_away * 100:.1f}%")



# =========================
# 🏈 NFL Matchup Predictor — Simple, conservative, clipped stat edges
# =========================
import os, glob, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# ---------- file finder ----------
def _find_one(possibles):
    for p in possibles:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

OFF_PATH = _find_one(["NFL_Offense.csv", "data/NFL_Offense.csv"])
DEF_PATH = _find_one(["NFL_Defense.csv", "data/NFL_Defense.csv"])
LOG_PATH = _find_one(["NFL_Game_log.csv", "data/NFL_Game_log.csv"])

MODEL_PATH = Path("nfl_classifier_simple.joblib")

if not OFF_PATH or not DEF_PATH or not LOG_PATH:
    st.error("Missing NFL CSV files (`NFL_Offense.csv`, `NFL_Defense.csv`, `NFL_Game_log.csv`).")
    st.stop()

# ---------- load ----------
off = pd.read_csv(OFF_PATH)
defn = pd.read_csv(DEF_PATH)
glog = pd.read_csv(LOG_PATH)

# ---------- canonicalize headers ----------
def _canonize_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip()).lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[/\-]", " ", s)
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _auto_rename(df: pd.DataFrame, is_defense: bool) -> pd.DataFrame:
    norm2orig = {_canonize_name(c): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm2orig:
                return norm2orig[c]
        return None

    team_col  = pick("team","teams")
    tot_perg  = pick("tot_yds_perg","total_yds_perg","yds_g","yds_perg","total_yards_per_game")
    pass_perg = pick("pass_yds_perg","passing_yds_perg","pass_yds_g","passing_yards_per_game")
    rush_perg = pick("rush_yds_perg","rushing_yds_perg","rush_yds_g","rushing_yards_per_game")
    pts_perg  = pick("pts_perg","points_per_game","pts_g")

    rename_map = {}
    if team_col: rename_map[team_col] = "Team"
    if is_defense:
        if tot_perg:  rename_map[tot_perg]  = "Tot_YDS_perG_Allowed"
        if pass_perg: rename_map[pass_perg] = "Pass_YDS_perG_Allowed"
        if rush_perg: rename_map[rush_perg] = "Rush_YDS_perG_Allowed"
        if pts_perg:  rename_map[pts_perg]  = "PTS_perG_Allowed"
    else:
        if tot_perg:  rename_map[tot_perg]  = "Tot_YDS_perG"
        if pass_perg: rename_map[pass_perg] = "Pass_YDS_perG"
        if rush_perg: rename_map[rush_perg] = "Rush_YDS_perG"
        if pts_perg:  rename_map[pts_perg]  = "PTS_perG"

    return df.rename(columns=rename_map)

off = _auto_rename(off, is_defense=False)
defn = _auto_rename(defn, is_defense=True)

# ---------- validate ----------
off_req = {"Team","Tot_YDS_perG","Pass_YDS_perG","Rush_YDS_perG","PTS_perG"}
def_req = {"Team","Tot_YDS_perG_Allowed","Pass_YDS_perG_Allowed","Rush_YDS_perG_Allowed","PTS_perG_Allowed"}
log_req = {"Home Team","Away Team","Home Score","Away Score"}

missing = []
if not off_req.issubset(off.columns): missing.append(f"Offense CSV missing: {off_req - set(off.columns)}")
if not def_req.issubset(defn.columns): missing.append(f"Defense CSV missing: {def_req - set(defn.columns)}")
if not log_req.issubset(glog.columns): missing.append(f"Game log CSV missing: {log_req - set(glog.columns)}")
if missing:
    st.error("Column mismatch:\n- " + "\n- ".join(missing))
    st.stop()

# ---------- name mapping (game log → full names in off/def files) ----------
NAME_MAP = {
    "49ers": "San Francisco 49ers",
    "Bears": "Chicago Bears",
    "Bengals": "Cincinnati Bengals",
    "Bills": "Buffalo Bills",
    "Broncos": "Denver Broncos",
    "Browns": "Cleveland Browns",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Cardinals": "Arizona Cardinals",
    "Chargers": "Los Angeles Chargers",
    "Chiefs": "Kansas City Chiefs",
    "Colts": "Indianapolis Colts",
    "Commanders": "Washington Commanders",
    "Cowboys": "Dallas Cowboys",
    "Dolphins": "Miami Dolphins",
    "Eagles": "Philadelphia Eagles",
    "Falcons": "Atlanta Falcons",
    "Giants": "New York Giants",
    "Jaguars": "Jacksonville Jaguars",
    "Jets": "New York Jets",
    "Lions": "Detroit Lions",
    "Packers": "Green Bay Packers",
    "Panthers": "Carolina Panthers",
    "Patriots": "New England Patriots",
    "Raiders": "Las Vegas Raiders",
    "Rams": "Los Angeles Rams",
    "Ravens": "Baltimore Ravens",
    "Saints": "New Orleans Saints",
    "Seahawks": "Seattle Seahawks",
    "Steelers": "Pittsburgh Steelers",
    "Texans": "Houston Texans",
    "Titans": "Tennessee Titans",
    "Vikings": "Minnesota Vikings",
}

glog["Home Team"] = glog["Home Team"].map(lambda s: NAME_MAP.get(str(s).strip(), str(s).strip()))
glog["Away Team"] = glog["Away Team"].map(lambda s: NAME_MAP.get(str(s).strip(), str(s).strip()))

# ---------- contexts ----------
off_means = off.drop(columns=["Team"]).mean()
off_stds  = off.drop(columns=["Team"]).std().replace(0, 1.0)
def_means = defn.drop(columns=["Team"]).mean()
def_stds  = defn.drop(columns=["Team"]).std().replace(0, 1.0)

off_d = {r.Team: r for _, r in off.iterrows()}
def_d = {r.Team: r for _, r in defn.iterrows()}
teams = sorted(set(off["Team"]).intersection(defn["Team"]))

def z_off(row, col):  # offense per-game (higher=better)
    return (row[col] - off_means[col]) / off_stds[col]

def inv_def(row, col_allowed):  # defense allowed (lower=better) -> invert so higher=better
    return -((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed])

FEATURES = [
    "edge_pts","edge_pass","edge_rush","edge_tot",
    "edge_pts_def","edge_pass_def","edge_rush_def","edge_tot_def",
    "edge_pass_minus_rush","edge_pts_combo","edge_tot_combo"
]

def build_edges(h_off, a_off, h_def, a_def):
    x = {
        "edge_pts":  z_off(h_off,"PTS_perG") - inv_def(a_def,"PTS_perG_Allowed"),
        "edge_pass": z_off(h_off,"Pass_YDS_perG") - inv_def(a_def,"Pass_YDS_perG_Allowed"),
        "edge_rush": z_off(h_off,"Rush_YDS_perG") - inv_def(a_def,"Rush_YDS_perG_Allowed"),
        "edge_tot":  z_off(h_off,"Tot_YDS_perG") - inv_def(a_def,"Tot_YDS_perG_Allowed"),
        "edge_pts_def":  inv_def(h_def,"PTS_perG_Allowed") - z_off(a_off,"PTS_perG"),
        "edge_pass_def": inv_def(h_def,"Pass_YDS_perG_Allowed") - z_off(a_off,"Pass_YDS_perG"),
        "edge_rush_def": inv_def(h_def,"Rush_YDS_perG_Allowed") - z_off(a_off,"Rush_YDS_perG"),
        "edge_tot_def":  inv_def(h_def,"Tot_YDS_perG_Allowed") - z_off(a_off,"Tot_YDS_perG"),
    }
    x["edge_pass_minus_rush"] = x["edge_pass"] - x["edge_rush"]
    x["edge_pts_combo"]       = x["edge_pts"]  + x["edge_pts_def"]
    x["edge_tot_combo"]       = x["edge_tot"]  + x["edge_tot_def"]
    return x

# ---------- feature clipping (90th percentile of |edge|) ----------
def learn_caps(df, cols, q=0.90):
    caps = {}
    for c in cols:
        if c not in df.columns or df[c].dropna().empty:
            caps[c] = 0.5
            continue
        cap = float(np.nanpercentile(np.abs(df[c].values), q*100))
        if not np.isfinite(cap) or cap < 0.5:
            cap = 0.5
        caps[c] = cap
    return caps

def clip_feats(d, caps):
    return {k: float(np.clip(v, -caps.get(k, 999),  caps.get(k, 999))) for k, v in d.items()}

# ---------- training ----------
@st.cache_resource(show_spinner=False)
def train_or_load_model():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["caps"]

    rows = []
    for _, g in glog.iterrows():
        ht, at = str(g["Home Team"]).strip(), str(g["Away Team"]).strip()
        if ht not in off_d or at not in off_d or ht not in def_d or at not in def_d:
            continue
        h_off, a_off, h_def, a_def = off_d[ht], off_d[at], def_d[ht], def_d[at]
        edges = build_edges(h_off, a_off, h_def, a_def)   # raw edges
        win = 1 if float(g["Home Score"]) > float(g["Away Score"]) else 0
        key = "||".join(sorted([ht, at]))
        rows.append({**edges, "home_win": win, "pair_key": key})

    if not rows:
        st.error("No training rows built. Check team names in game log vs Offense/Defense CSVs.")
        st.stop()

    feat = pd.DataFrame(rows)
    # ensure all feature columns exist
    for c in FEATURES:
        if c not in feat.columns:
            feat[c] = 0.0

    caps = learn_caps(feat, FEATURES, q=0.90)

    # clip features for training
    for c in FEATURES:
        feat[c] = np.clip(feat[c].values, -caps[c], caps[c])

    X = feat[FEATURES].values
    y = feat["home_win"].values
    groups = feat["pair_key"].values

    # group-aware OOF (for sanity; not displayed)
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, y, groups):
        m = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", solver="lbfgs")
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    # (Metrics computed but not shown)
    _ = {
        "auc": float(roc_auc_score(y, oof)),
        "log_loss": float(log_loss(y, oof, eps=1e-15)),
        "brier": float(brier_score_loss(y, oof)),
        "accuracy@0.5": float(accuracy_score(y, (oof>=0.5).astype(int))),
        "n": int(len(y)),
    }

    model = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(X, y)

    joblib.dump({"model": model, "caps": caps}, MODEL_PATH)
    return model, caps

# ---------- UI ----------
st.title("NFL Matchup Predictor")

clf, caps = train_or_load_model()

home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", teams)

if home and away and home != away:
    if home not in off_d or away not in off_d:
        st.error("Missing team stats for one or both teams.")
    else:
        h_off, a_off = off_d[home], off_d[away]
        h_def, a_def = def_d[home], def_d[away]
        raw = build_edges(h_off, a_off, h_def, a_def)
        x = clip_feats(raw, caps)
        vec = pd.DataFrame([x])[FEATURES].values
        prob_home = float(clf.predict_proba(vec)[0, 1])
        prob_away = 1.0 - prob_home
        winner = home if prob_home >= prob_away else away

        st.subheader("Prediction")
        st.success(f"Predicted Winner: {winner}")
        st.metric(f"{home} Win Probability", f"{prob_home * 100:.1f}%")
        st.metric(f"{away} Win Probability", f"{prob_away * 100:.1f}%")
elif home == away:
    st.warning("Please select two different teams.")
