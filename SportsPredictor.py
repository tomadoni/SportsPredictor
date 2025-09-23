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
# 🏈 NFL Matchup Predictor — CSV-only, header-normalized, group-aware CV, calibrated, capped confidence
# =========================
import os, glob, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# -------- file finder (supports data/ subfolder) --------
def _find_one(possibles):
    for p in possibles:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

OFF_PATH = _find_one(["NFL_Offense.csv", "data/NFL_Offense.csv"])
DEF_PATH = _find_one(["NFL_Defense.csv", "data/NFL_Defense.csv"])
LOG_PATH = _find_one(["NFL_Game_log.csv", "data/NFL_Game_log.csv"])

MODEL_PATH = Path("nfl_best_classifier.joblib")
METRICS_CSV = Path("nfl_best_classifier_metrics.csv")

st.caption(f"🗂 Working dir: `{os.getcwd()}`")
st.caption(f"🔎 Offense CSV: `{OFF_PATH}`")
st.caption(f"🔎 Defense CSV: `{DEF_PATH}`")
st.caption(f"🔎 Game Log CSV: `{LOG_PATH}`")

# -------- load CSVs --------
if not OFF_PATH or not DEF_PATH or not LOG_PATH:
    st.error("Missing files. Ensure these exist (in app folder or `data/`): "
             "`NFL_Offense.csv`, `NFL_Defense.csv`, `NFL_Game_log.csv`.")
    st.stop()

off = pd.read_csv(OFF_PATH)
defn = pd.read_csv(DEF_PATH)
glog = pd.read_csv(LOG_PATH)

# ---------- Normalize & auto-rename headers ----------
def _canonize_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    s = s.lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[/\-]", " ", s)
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _auto_rename(df: pd.DataFrame, is_defense: bool) -> pd.DataFrame:
    norm2orig = { _canonize_name(c): c for c in df.columns }

    def pick(*cands):
        for c in cands:
            if c in norm2orig: return norm2orig[c]
        return None

    # Common aliases -> canonical
    team_col  = pick("team","teams")
    tot_perg  = pick("tot_yds_perg","total_yds_perg","yds_g","yds_perg","total_yards_per_game","total_yds_g","total_yds_game")
    pass_perg = pick("pass_yds_perg","passing_yds_perg","pass_yds_g","passing_yards_per_game","pass_yards_g","pass_yards_per_game")
    rush_perg = pick("rush_yds_perg","rushing_yds_perg","rush_yds_g","rushing_yards_per_game","rush_yards_g","rush_yards_per_game")
    pts_perg  = pick("pts_perg","points_per_game","pts_g","points_g")

    rename_map = {}
    if team_col:  rename_map[team_col]  = "Team"

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

    df = df.rename(columns=rename_map)

    with st.expander(("Defense" if is_defense else "Offense") + " column mapping"):
        st.write(rename_map)
    return df

# Apply auto-rename, then trim whitespace
off = _auto_rename(off, is_defense=False)
defn = _auto_rename(defn, is_defense=True)
off.columns  = [c.strip() for c in off.columns]
defn.columns = [c.strip() for c in defn.columns]
glog.columns = [c.strip() for c in glog.columns]

# -------- validate expected columns --------
off_req = {"Team","Tot_YDS_perG","Pass_YDS_perG","Rush_YDS_perG","PTS_perG"}
def_req = {"Team","Tot_YDS_perG_Allowed","Pass_YDS_perG_Allowed","Rush_YDS_perG_Allowed","PTS_perG_Allowed"}
log_req = {"Home Team","Away Team","Home Score","Away Score"}

missing = []
if not off_req.issubset(off.columns): missing.append(f"Offense CSV missing: {off_req - set(off.columns)}")
if not def_req.issubset(defn.columns): missing.append(f"Defense CSV missing: {def_req - set(defn.columns)}")
if not log_req.issubset(glog.columns): missing.append(f"Game log CSV missing: {log_req - set(glog.columns)}")
if missing:
    st.error("Column mismatch after auto-rename:\n- " + "\n- ".join(missing))
    with st.expander("Preview files (post-rename)"):
        st.write("Offense head:", off.head())
        st.write("Defense head:", defn.head())
        st.write("Game log head:", glog.head())
    st.stop()

# -------- z-score contexts & lookups --------
off_means = off.drop(columns=["Team"]).mean()
off_stds  = off.drop(columns=["Team"]).std().replace(0, 1.0)
def_means = defn.drop(columns=["Team"]).mean()
def_stds  = defn.drop(columns=["Team"]).std().replace(0, 1.0)

off_d = {r.Team: r for _, r in off.iterrows()}
def_d = {r.Team: r for _, r in defn.iterrows()}
teams = sorted(set(off["Team"]).intersection(defn["Team"]))

def z_off(row, col):  # offense per-G (higher = better)
    return (row[col] - off_means[col]) / off_stds[col]

def inv_def(row, col_allowed):  # defense allowed (lower = better) -> invert so higher is better
    return -((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed])

# -------- features --------
BASE_FEATURES = [
    "edge_pts","edge_pass","edge_rush","edge_tot",
    "edge_pts_def","edge_pass_def","edge_rush_def","edge_tot_def",
]
EXTRA_FEATURES = ["edge_pass_minus_rush","edge_pts_combo","edge_tot_combo"]
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES

def build_feature_dict(h_off, a_off, h_def, a_def):
    x = {
        "edge_pts":  z_off(h_off, "PTS_perG")        - inv_def(a_def, "PTS_perG_Allowed"),
        "edge_pass": z_off(h_off, "Pass_YDS_perG")   - inv_def(a_def, "Pass_YDS_perG_Allowed"),
        "edge_rush": z_off(h_off, "Rush_YDS_perG")   - inv_def(a_def, "Rush_YDS_perG_Allowed"),
        "edge_tot":  z_off(h_off, "Tot_YDS_perG")    - inv_def(a_def, "Tot_YDS_perG_Allowed"),

        "edge_pts_def":  inv_def(h_def, "PTS_perG_Allowed")        - z_off(a_off, "PTS_perG"),
        "edge_pass_def": inv_def(h_def, "Pass_YDS_perG_Allowed")   - z_off(a_off, "Pass_YDS_perG"),
        "edge_rush_def": inv_def(h_def, "Rush_YDS_perG_Allowed")   - z_off(a_off, "Rush_YDS_perG"),
        "edge_tot_def":  inv_def(h_def, "Tot_YDS_perG_Allowed")    - z_off(a_off, "Tot_YDS_perG"),
    }
    x["edge_pass_minus_rush"] = x["edge_pass"] - x["edge_rush"]
    x["edge_pts_combo"]       = x["edge_pts"]  + x["edge_pts_def"]
    x["edge_tot_combo"]       = x["edge_tot"]  + x["edge_tot_def"]
    return x

# nickname → full team name (for aligning game log names)
NAME_MAP = {
    "49ers":"San Francisco 49ers","Niners":"San Francisco 49ers","Bears":"Chicago Bears","Lions":"Detroit Lions",
    "Packers":"Green Bay Packers","Vikings":"Minnesota Vikings","Cowboys":"Dallas Cowboys","Giants":"New York Giants",
    "Eagles":"Philadelphia Eagles","Commanders":"Washington Commanders","Rams":"Los Angeles Rams","Seahawks":"Seattle Seahawks",
    "Cardinals":"Arizona Cardinals","Saints":"New Orleans Saints","Buccaneers":"Tampa Bay Buccaneers","Bucs":"Tampa Bay Buccaneers",
    "Falcons":"Atlanta Falcons","Panthers":"Carolina Panthers",
    "Bills":"Buffalo Bills","Dolphins":"Miami Dolphins","Patriots":"New England Patriots","Jets":"New York Jets",
    "Ravens":"Baltimore Ravens","Bengals":"Cincinnati Bengals","Browns":"Cleveland Browns","Steelers":"Pittsburgh Steelers",
    "Texans":"Houston Texans","Colts":"Indianapolis Colts","Jaguars":"Jacksonville Jaguars","Titans":"Tennessee Titans",
    "Broncos":"Denver Broncos","Chiefs":"Kansas City Chiefs","Raiders":"Las Vegas Raiders","Chargers":"Los Angeles Chargers",
}

# -------- group-aware OOF with calibrated RF (no leakage by matchup) --------
def _oof_calibrated_proba_grouped(X, y, groups, rf_params, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]

        rf = RandomForestClassifier(**rf_params, random_state=seed)
        rf.fit(X_tr, y_tr)

        cal = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
        cal.fit(X_tr, y_tr)

        oof[va_idx] = cal.predict_proba(X_va)[:, 1]
    return oof

@st.cache_resource(show_spinner=False)
def train_or_load_model():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        metrics = pd.read_csv(METRICS_CSV).iloc[0].to_dict() if METRICS_CSV.exists() else None
        return bundle["model"], bundle.get("feature_columns", ALL_FEATURES), set(bundle.get("seen_pairs", [])), metrics

    with st.spinner("Training NFL classifier (group-aware CV)…"):
        gl = glog.copy()
        gl["Home Team"] = gl["Home Team"].map(lambda s: NAME_MAP.get(str(s).strip(), str(s).strip()))
        gl["Away Team"] = gl["Away Team"].map(lambda s: NAME_MAP.get(str(s).strip(), str(s).strip()))

        rows = []
        for _, g in gl.iterrows():
            ht = str(g["Home Team"]).strip()
            at = str(g["Away Team"]).strip()
            if ht not in off_d or ht not in def_d or at not in off_d or at not in def_d:
                continue
            h_off, a_off, h_def, a_def = off_d[ht], off_d[at], def_d[ht], def_d[at]
            x = build_feature_dict(h_off, a_off, h_def, a_def)
            home_win = 1 if float(g["Home Score"]) > float(g["Away Score"]) else 0
            pair_key = "||".join(sorted([ht, at]))  # unordered matchup ID
            rows.append({**x, "home_win": home_win, "pair_key": pair_key})

        feat = pd.DataFrame(rows)
        if len(feat) < 25:
            st.error("Not enough rows from the game log to train the NFL model. Check team names/scores.")
            st.stop()

        X = feat.drop(columns=["home_win","pair_key"]).copy()
        X = X[ALL_FEATURES]
        y = feat["home_win"].values
        groups = feat["pair_key"].values
        Xv = X.values

        rf_params = dict(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

        # Group-aware OOF probs for honest metrics
        oof = _oof_calibrated_proba_grouped(Xv, y, groups, rf_params, n_splits=5, seed=42)
        y_hat = (oof >= 0.5).astype(int)
        metrics = {
            "auc": float(roc_auc_score(y, oof)),
            "log_loss": float(log_loss(y, oof, eps=1e-15)),
            "brier": float(brier_score_loss(y, oof)),
            "accuracy@0.5": float(accuracy_score(y, y_hat)),
            "n": int(len(y)),
        }

        # Fit final model on full data + calibrate
        rf_full = RandomForestClassifier(**rf_params, random_state=42)
        rf_full.fit(Xv, y)
        cal_full = CalibratedClassifierCV(rf_full, method="sigmoid", cv="prefit")
        cal_full.fit(Xv, y)

        seen_pairs = set(feat["pair_key"].unique())
        joblib.dump({"model": cal_full, "feature_columns": ALL_FEATURES, "seen_pairs": seen_pairs}, MODEL_PATH)
        pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)
        return cal_full, ALL_FEATURES, seen_pairs, metrics

def predict_proba_with_limit(clf, vec, limit=0.30):
    """
    Enforce symmetric max deviation from 0.5.
    Example: limit=0.30 → p in [0.20, 0.80].
    """
    p = float(clf.predict_proba(vec)[0, 1])
    lo, hi = 0.5 - limit, 0.5 + limit
    if p < lo: return lo
    if p > hi: return hi
    return p

# -------- UI --------
st.title("🏈 NFL Matchup Predictor (CSV-only, group-aware CV, capped confidence)")
st.caption("RF (regularized) + sigmoid calibration. GroupKFold by matchup prevents leakage; outputs capped around 50/50.")

with st.expander("⚙️ Probability cap (max confidence)"):
    limit = st.slider(
        "Max deviation from 50/50",
        0.05, 0.45, 0.30, 0.01,
        help="Sets the furthest a probability may move away from 0.5. 0.30 → 20–80; 0.25 → 25–75."
    )
    stricter_seen = st.checkbox(
        "Use stricter cap for seen matchups",
        value=True,
        help="If a pair existed in training, clamp closer to 50/50."
    )
    limit_seen = st.slider(
        "Max deviation for seen matchups",
        0.05, 0.45, 0.25, 0.01,
        help="Only used when 'stricter cap' is enabled."
    )

clf, feature_columns, seen_pairs, cv_metrics = train_or_load_model()

c1, c2 = st.columns(2)
with c1:
    home_team = st.selectbox("🏠 Home Team (NFL)", teams, index=0 if teams else None)
with c2:
    away_team = st.selectbox("✈️ Away Team (NFL)", teams, index=1 if len(teams) > 1 else None)

if home_team and away_team:
    if home_team == away_team:
        st.warning("Please select two different teams.")
    elif home_team not in off_d or away_team not in off_d:
        st.error("Could not find team stats for one or both teams.")
    else:
        h_off, a_off, h_def, a_def = off_d[home_team], off_d[away_team], def_d[home_team], def_d[away_team]
        try:
            x = build_feature_dict(h_off, a_off, h_def, a_def)
        except KeyError as e:
            st.error(f"Missing expected column `{e.args[0]}` after auto-rename.")
            with st.expander("Available offense/defense columns now"):
                st.write("Offense:", list(off.columns))
                st.write("Defense:", list(defn.columns))
            st.stop()

        vec = pd.DataFrame([{k: x[k] for k in feature_columns}]).values

        # Apply limit (stricter for seen pairs, if selected)
        pair_key_ui = "||".join(sorted([home_team, away_team]))
        this_limit = limit_seen if (stricter_seen and pair_key_ui in seen_pairs) else limit

        prob_home = predict_proba_with_limit(clf, vec, limit=this_limit)
        prob_away = 1.0 - prob_home
        winner = home_team if prob_home >= prob_away else away_team

        st.subheader("📈 Prediction")
        st.success(f"**Predicted Winner: {winner}**")
        c3, c4 = st.columns(2)
        c3.metric(f"{home_team} Win Probability", f"{prob_home*100:.1f}%")
        c4.metric(f"{away_team} Win Probability", f"{prob_away*100:.1f}%")

        if pair_key_ui in seen_pairs:
            st.info("ℹ️ This matchup pair exists in the training data. Group-aware CV prevents overconfident repeats.")

        with st.expander("Inputs used by the model"):
            st.json({k: f"{x[k]:+.3f}" for k in feature_columns})

        if cv_metrics:
            with st.expander("Model CV metrics (group-aware OOF)"):
                st.write(cv_metrics)





