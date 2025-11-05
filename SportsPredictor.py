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

# =========================
# 🏀 NBA Matchup Predictor — No game log (prior-based, tunable)
# =========================
import glob, re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NBA Matchup Predictor", layout="centered")

# ---------- file finder ----------
def _find_one(possibles):
    for p in possibles:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

OFF_PATH = _find_one(["NBA_Offense.csv", "data/NBA_Offense.csv"])
DEF_PATH = _find_one(["NBA_Defense.csv", "data/NBA_Defense.csv"])

if not OFF_PATH or not DEF_PATH:
    st.error("Missing NBA CSV files (`NBA_Offense.csv`, `NBA_Defense.csv`).")
    st.stop()

# ---------- canonicalize ----------
def _canonize_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip()).lower()
    s = s.replace("%", "pct")
    s = s.replace("3p", "threep").replace("3-pt", "threep").replace("3 pt", "threep")
    s = re.sub(r"[/\\\-]", " ", s)
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _auto_rename_nba(df: pd.DataFrame, is_defense: bool) -> pd.DataFrame:
    norm2orig = {_canonize_name(c): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm2orig:
                return norm2orig[c]
        return None

    team_col   = pick("team","teams","franchise","club","squad")
    pts_perg   = pick("pts_perg","points_per_game","pts_g","ppg")
    fg_pct     = pick("fg_pct","field_goal_pct","field_goals_pct","fgp")
    threep_pct = pick("threep_pct","three_point_pct","three_point_percentage","3p_pct","3pt_pct")
    reb_perg   = pick("reb_perg","rebounds_per_game","rebs_perg","rpg","trb_perg","trb_g")

    rn = {}
    if team_col: rn[team_col] = "Team"
    if is_defense:
        if pts_perg:   rn[pts_perg]   = "PTS_perG_Allowed"
        if fg_pct:     rn[fg_pct]     = "FG_pct_Allowed"
        if threep_pct: rn[threep_pct] = "ThreeP_pct_Allowed"
        if reb_perg:   rn[reb_perg]   = "REB_perG_Allowed"
    else:
        if pts_perg:   rn[pts_perg]   = "PTS_perG"
        if fg_pct:     rn[fg_pct]     = "FG_pct"
        if threep_pct: rn[threep_pct] = "ThreeP_pct"
        if reb_perg:   rn[reb_perg]   = "REB_perG"
    return df.rename(columns=rn)

@st.cache_data(show_spinner=False)
def load_frames():
    off = pd.read_csv(OFF_PATH)
    de  = pd.read_csv(DEF_PATH)
    off = _auto_rename_nba(off, is_defense=False)
    de  = _auto_rename_nba(de,  is_defense=True)
    return off, de

off, defn = load_frames()

# ---------- validate ----------
off_req = {"Team","PTS_perG","FG_pct","ThreeP_pct","REB_perG"}
def_req = {"Team","PTS_perG_Allowed","FG_pct_Allowed","ThreeP_pct_Allowed","REB_perG_Allowed"}

missing = []
if not off_req.issubset(off.columns): missing.append(f"Offense CSV missing: {off_req - set(off.columns)}")
if not def_req.issubset(defn.columns): missing.append(f"Defense CSV missing: {def_req - set(defn.columns)}")
if missing:
    st.error("Column mismatch:\n- " + "\n- ".join(missing))
    st.stop()

# ---------- contexts ----------
off_means = off.drop(columns=["Team"]).mean(numeric_only=True)
off_stds  = off.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)
def_means = defn.drop(columns=["Team"]).mean(numeric_only=True)
def_stds  = defn.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)

off_d = {r.Team: r for _, r in off.iterrows()}
def_d = {r.Team: r for _, r in defn.iterrows()}
teams = sorted(set(off["Team"]).intersection(defn["Team"]))

def z_off(row, col):  # offense (higher=better)
    return float((row[col] - off_means[col]) / off_stds[col])

def inv_def(row, col_allowed):  # defense allowed (lower=better) -> invert so higher=better
    return float(-((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed]))

FEATURES = [
    "edge_pts","edge_fg","edge_3p","edge_reb",
    "edge_pts_def","edge_fg_def","edge_3p_def","edge_reb_def",
    "edge_shooting_gap","edge_pts_combo","edge_reb_combo"
]

def build_edges(h_off, a_off, h_def, a_def):
    x = {
        "edge_pts":  z_off(h_off,"PTS_perG")   - inv_def(a_def,"PTS_perG_Allowed"),
        "edge_fg":   z_off(h_off,"FG_pct")     - inv_def(a_def,"FG_pct_Allowed"),
        "edge_3p":   z_off(h_off,"ThreeP_pct") - inv_def(a_def,"ThreeP_pct_Allowed"),
        "edge_reb":  z_off(h_off,"REB_perG")   - inv_def(a_def,"REB_perG_Allowed"),

        "edge_pts_def":  inv_def(h_def,"PTS_perG_Allowed")    - z_off(a_off,"PTS_perG"),
        "edge_fg_def":   inv_def(h_def,"FG_pct_Allowed")      - z_off(a_off,"FG_pct"),
        "edge_3p_def":   inv_def(h_def,"ThreeP_pct_Allowed")  - z_off(a_off,"ThreeP_pct"),
        "edge_reb_def":  inv_def(h_def,"REB_perG_Allowed")    - z_off(a_off,"REB_perG"),
    }
    x["edge_shooting_gap"] = x["edge_fg"] + x["edge_3p"]
    x["edge_pts_combo"]    = x["edge_pts"] + x["edge_pts_def"]
    x["edge_reb_combo"]    = x["edge_reb"] + x["edge_reb_def"]
    return x

# ---------- clipping ----------
def learn_caps_from_grid(teams, default_cap=0.5, q=0.90):
    rows = []
    for h in teams:
        for a in teams:
            if h == a: continue
            h_off, a_off = off_d[h], off_d[a]
            h_def, a_def = def_d[h], def_d[a]
            rows.append(build_edges(h_off, a_off, h_def, a_def))
    df = pd.DataFrame(rows)
    caps = {}
    for c in FEATURES:
        if c not in df or df[c].dropna().empty:
            caps[c] = default_cap
            continue
        cap = float(np.nanpercentile(np.abs(df[c].values), int(q*100)))
        if not np.isfinite(cap) or cap < default_cap:
            cap = default_cap
        caps[c] = cap
    return caps

@st.cache_data(show_spinner=False)
def get_caps():
    return learn_caps_from_grid(teams, default_cap=0.5, q=0.90)

CAPS = get_caps()

def clip_feats(d):
    return {k: float(np.clip(v, -CAPS.get(k, 999), CAPS.get(k, 999))) for k, v in d.items()}

# ---------- prior-based scoring (no training) ----------
DEFAULT_WEIGHTS = {
    # offense vs opp D
    "edge_pts": 0.45,
    "edge_fg":  0.25,
    "edge_3p":  0.20,
    "edge_reb": 0.10,
    # defense vs opp O
    "edge_pts_def": 0.30,
    "edge_fg_def":  0.15,
    "edge_3p_def":  0.15,
    "edge_reb_def": 0.10,
    # composites
    "edge_shooting_gap": 0.15,
    "edge_pts_combo":    0.20,
    "edge_reb_combo":    0.05,
}

def logistic(x, k=1.35):  # temperature k controls sharpness
    return 1.0 / (1.0 + np.exp(-k * x))

def score_from_edges(ed, weights, bias=0.0):
    # Weighted sum over clipped edges
    s = bias
    for f, w in weights.items():
        s += w * ed.get(f, 0.0)
    return float(s)

# ---------- UI ----------
st.title("NBA Matchup Predictor")
st.caption("Prior-based model using offense vs defense ‘stat edges’. Tune weights & home-court in the sidebar.")

with st.sidebar:
    st.header("Model Settings")
    hca = st.slider("Home-court advantage (edge units)", 0.00, 0.80, 0.20, 0.01)
    temp = st.slider("Sigmoid temperature (k)", 0.50, 3.00, 1.35, 0.05)

    st.subheader("Weights (click to expand)")
    with st.expander("Adjust feature weights", expanded=False):
        w = {}
        for key in DEFAULT_WEIGHTS:
            w[key] = st.slider(key, -1.00, 1.50, float(DEFAULT_WEIGHTS[key]), 0.05)
    st.markdown("---")
    st.caption("Tip: Increase shooting weights (FG/3P) for pace/spacing eras; increase REB for physical matchups.")

home = st.selectbox("Home Team", teams, index=0)
away = st.selectbox("Away Team", [t for t in teams if t != home], index=0)

if home and away and home != away:
    h_off, a_off = off_d[home], off_d[away]
    h_def, a_def = def_d[home], def_d[away]
    raw = build_edges(h_off, a_off, h_def, a_def)
    x = clip_feats(raw)

    # base edge (home perspective)
    base_score = score_from_edges(x, w if 'w' in locals() else DEFAULT_WEIGHTS, bias=0.0)

    # symmetric check: approximate away edge by flipping home/away
    raw_rev = build_edges(a_off, h_off, a_def, h_def)
    x_rev = clip_feats(raw_rev)
    away_base = score_from_edges(x_rev, w if 'w' in locals() else DEFAULT_WEIGHTS, bias=0.0)

    # ensure antisymmetry by averaging (optional, helps stability)
    sym_score = 0.5 * (base_score - away_base)

    # add home-court advantage
    sym_score += hca

    prob_home = float(logistic(sym_score, k=temp))
    prob_away = 1.0 - prob_home
    winner = home if prob_home >= prob_away else away

    st.subheader("Prediction")
    st.success(f"Predicted Winner: {winner}")
    c1, c2 = st.columns(2)
    c1.metric(f"{home} Win Probability", f"{prob_home * 100:.1f}%")
    c2.metric(f"{away} Win Probability", f"{prob_away * 100:.1f}%")

    with st.expander("Feature edges (clipped)"):
        st.dataframe(pd.DataFrame([x]).T.rename(columns={0:"value"}))

else:
    st.info("Select two different teams.")

st.markdown(
    """
    **Notes**
    - No game log needed: probabilities come from standardized offense vs defense edges and a tunable sigmoid.
    - Use the sidebar to calibrate weights or home-court advantage to your era/data source.
    """
)



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
