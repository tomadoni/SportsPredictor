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
# 🏀 NBA Matchup Predictor — clipped, conservative, no tuners
# =========================
import glob, re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


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
    pts_perg   = pick("pts_perg","points_per_game","pts_g","ppg","pts")
    fg_pct     = pick("fg_pct","field_goal_pct","field_goals_pct","fgp","fg_percentage")
    threep_pct = pick("threep_pct","three_point_pct","three_point_percentage","3p_pct","3pt_pct")
    reb_perg   = pick("reb_perg","rebounds_per_game","rebs_perg","rpg","trb_perg","trb_g","reb")

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

# ---------- clipping learned from the full grid ----------
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

# ---------- conservative scoring ----------
FIXED_WEIGHTS = {
    # offense vs opp D
    "edge_pts": 0.40,
    "edge_fg":  0.22,
    "edge_3p":  0.18,
    "edge_reb": 0.08,
    # defense vs opp O
    "edge_pts_def": 0.24,
    "edge_fg_def":  0.12,
    "edge_3p_def":  0.12,
    "edge_reb_def": 0.08,
    # composites
    "edge_shooting_gap": 0.10,
    "edge_pts_combo":    0.14,
    "edge_reb_combo":    0.04,
}

HOME_EDGE = 0.18       # in edge units
TEMP_K    = 0.90       # softer sigmoid
SHRINK_L  = 0.65       # pull toward 0.5
CLAMP_MIN = 0.10
CLAMP_MAX = 0.90

def logistic(x, k=TEMP_K):
    return 1.0 / (1.0 + np.exp(-k * x))

def score_from_edges(ed, weights):
    s = 0.0
    for f, w in weights.items():
        s += w * ed.get(f, 0.0)
    return float(s)

# ---------- UI ----------
st.title("NBA Matchup Predictor")
home = st.selectbox("Home Team", teams, index=0)
away = st.selectbox("Away Team", [t for t in teams if t != home], index=0)

if home and away and home != away:
    h_off, a_off = off_d[home], off_d[away]
    h_def, a_def = def_d[home], def_d[away]

    # home perspective
    raw_h = build_edges(h_off, a_off, h_def, a_def)
    x_h   = clip_feats(raw_h)
    s_h   = score_from_edges(x_h, FIXED_WEIGHTS)

    # away perspective (symmetry)
    raw_a = build_edges(a_off, h_off, a_def, h_def)
    x_a   = clip_feats(raw_a)
    s_a   = score_from_edges(x_a, FIXED_WEIGHTS)

    # antisymmetric score + fixed home-court
    sym_score = 0.5 * (s_h - s_a) + HOME_EDGE

    # probability with conservative shaping
    p_home = float(logistic(sym_score, k=TEMP_K))
    # shrink toward 0.5 (calibration-by-design)
    p_home = 0.5 + SHRINK_L * (p_home - 0.5)
    # final clamp to avoid overconfidence
    p_home = float(np.clip(p_home, CLAMP_MIN, CLAMP_MAX))
    p_away = 1.0 - p_home

    winner = home if p_home >= p_away else away

    st.subheader("Prediction")
    st.success(f"Predicted Winner: {winner}")
    c1, c2 = st.columns(2)
    c1.metric(f"{home} Win Probability", f"{p_home * 100:.1f}%")
    c2.metric(f"{away} Win Probability", f"{p_away * 100:.1f}%")

    with st.expander("Feature edges (clipped)"):
        show = pd.DataFrame([x_h]).T.rename(columns={0:"home_view_edge"})
        st.dataframe(show)

else:
    st.info("Select two different teams.")



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
# 🏈 NFL Matchup Predictor — Direct edges, no training, no home bias
# =========================
import glob, re
import numpy as np
import pandas as pd
import streamlit as st

# ---------- file finder ----------
def _find_one(possibles):
    for p in possibles:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

OFF_PATH = _find_one(["NFL_Offense.csv", "data/NFL_Offense.csv"])
DEF_PATH = _find_one(["NFL_Defense.csv", "data/NFL_Defense.csv"])

if not OFF_PATH or not DEF_PATH:
    st.error("Missing NFL CSV files (`NFL_Offense.csv`, `NFL_Defense.csv`).")
    st.stop()

# ---------- load ----------
off = pd.read_csv(OFF_PATH)
defn = pd.read_csv(DEF_PATH)

# ---------- canonicalize headers ----------
def _canonize_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip()).lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[/\\-]", " ", s)
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

    out = df.rename(columns=rename_map)

    # normalize Rush_YDS_perG_Allowed variants
    for c in list(out.columns):
        if _canonize_name(c) == "rush_yds_perg_allowed" and c != "Rush_YDS_perG_Allowed":
            out = out.rename(columns={c: "Rush_YDS_perG_Allowed"})

    return out

off = _auto_rename(off, is_defense=False)
defn = _auto_rename(defn, is_defense=True)

# ---------- validate ----------
off_req = {"Team","Tot_YDS_perG","Pass_YDS_perG","Rush_YDS_perG","PTS_perG"}
def_req = {"Team","Tot_YDS_perG_Allowed","Pass_YDS_perG_Allowed","Rush_YDS_perG_Allowed","PTS_perG_Allowed"}

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

def z_off(row, col):  # offense per-game (higher=better)
    return (row[col] - off_means[col]) / off_stds[col]

def inv_def(row, col_allowed):  # defense allowed (lower=better) -> invert so higher=better
    return -((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed])

# ---------- anti-symmetric features ----------
FEATURES = ["edge_pts", "edge_pass", "edge_rush", "edge_tot", "edge_pass_minus_rush"]

def build_edges(h_off, a_off, h_def, a_def):
    # "attack score" = offense z - inverted defense z
    home_pts_att  = z_off(h_off,"PTS_perG")       - inv_def(a_def,"PTS_perG_Allowed")
    away_pts_att  = z_off(a_off,"PTS_perG")       - inv_def(h_def,"PTS_perG_Allowed")
    home_pass_att = z_off(h_off,"Pass_YDS_perG")  - inv_def(a_def,"Pass_YDS_perG_Allowed")
    away_pass_att = z_off(a_off,"Pass_YDS_perG")  - inv_def(h_def,"Pass_YDS_perG_Allowed")
    home_rush_att = z_off(h_off,"Rush_YDS_perG")  - inv_def(a_def,"Rush_YDS_perG_Allowed")
    away_rush_att = z_off(a_off,"Rush_YDS_perG")  - inv_def(h_def,"Rush_YDS_perG_Allowed")
    home_tot_att  = z_off(h_off,"Tot_YDS_perG")   - inv_def(a_def,"Tot_YDS_perG_Allowed")
    away_tot_att  = z_off(a_off,"Tot_YDS_perG")   - inv_def(h_def,"Tot_YDS_perG_Allowed")

    # anti-symmetric edges: swap home/away → all flip sign
    edge_pts  = home_pts_att  - away_pts_att
    edge_pass = home_pass_att - away_pass_att
    edge_rush = home_rush_att - away_rush_att
    edge_tot  = home_tot_att  - away_tot_att

    x = {
        "edge_pts":  edge_pts,
        "edge_pass": edge_pass,
        "edge_rush": edge_rush,
        "edge_tot":  edge_tot,
    }
    x["edge_pass_minus_rush"] = x["edge_pass"] - x["edge_rush"]
    return x

def learn_caps_from_off_def(q=0.90):
    # rough global caps from all pairwise edges (off/def tables only, no game log)
    rows = []
    for h_team in teams:
        for a_team in teams:
            if h_team == a_team:
                continue
            h_off, a_off = off_d[h_team], off_d[a_team]
            h_def, a_def = def_d[h_team], def_d[a_team]
            rows.append(build_edges(h_off, a_off, h_def, a_def))
    df = pd.DataFrame(rows)
    caps = {}
    for c in FEATURES:
        col = df[c].values
        if df[c].dropna().empty:
            caps[c] = 0.5
            continue
        cap = float(np.nanpercentile(np.abs(col), q*100))
        if not np.isfinite(cap) or cap < 0.5:
            cap = 0.5
        caps[c] = cap
    return caps

CAPS = learn_caps_from_off_def()

def clip_feats(d, caps=CAPS):
    return {k: float(np.clip(v, -caps.get(k, 999), caps.get(k, 999))) for k, v in d.items()}

def direct_prob(h_off, a_off, h_def, a_def, k=0.9):
    edges = build_edges(h_off, a_off, h_def, a_def)
    x = clip_feats(edges, CAPS)
    # simple weighted net: positive → home edge, negative → away edge
    net = (
        0.5 * x["edge_pts"] +
        0.3 * x["edge_tot"] +
        0.2 * x["edge_pass_minus_rush"]
    )
    # logistic to [0,1]
    return float(1.0 / (1.0 + np.exp(-k * net))), edges

# ---------- UI ----------
st.title("NFL Matchup Predictor (No Training • Pure Stat Edges)")

home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", teams)

if home and away and home != away:
    if home not in off_d or away not in off_d or home not in def_d or away not in def_d:
        st.error("Missing team stats for one or both teams.")
    else:
        h_off, a_off = off_d[home], off_d[away]
        h_def, a_def = def_d[home], def_d[away]
        prob_home, raw_edges = direct_prob(h_off, a_off, h_def, a_def, k=0.9)
        prob_away = 1.0 - prob_home
        winner = home if prob_home >= prob_away else away

        st.subheader("Prediction")
        st.success(f"Predicted Winner: {winner}")
        c1, c2 = st.columns(2)
        c1.metric(f"{home} Win Probability", f"{prob_home * 100:.1f}%")
        c2.metric(f"{away} Win Probability", f"{prob_away * 100:.1f}%")

        with st.expander("Debug: inputs & features"):
            st.markdown("**Raw Offense (Home / Away)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        h_off[["Tot_YDS_perG","Pass_YDS_perG","Rush_YDS_perG","PTS_perG"]],
                        a_off[["Tot_YDS_perG","Pass_YDS_perG","Rush_YDS_perG","PTS_perG"]],
                    ],
                    index=[f"{home} OFF", f"{away} OFF"],
                )
            )

            st.markdown("**Raw Defense Allowed (Home / Away)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        h_def[["Tot_YDS_perG_Allowed","Pass_YDS_perG_Allowed","Rush_YDS_perG_Allowed","PTS_perG_Allowed"]],
                        a_def[["Tot_YDS_perG_Allowed","Pass_YDS_perG_Allowed","Rush_YDS_perG_Allowed","PTS_perG_Allowed"]],
                    ],
                    index=[f"{home} DEF", f"{away} DEF"],
                )
            )

            def zrow_off(r):
                return pd.Series({
                    "z_pts": z_off(r,"PTS_perG"),
                    "z_pass": z_off(r,"Pass_YDS_perG"),
                    "z_rush": z_off(r,"Rush_YDS_perG"),
                    "z_tot": z_off(r,"Tot_YDS_perG"),
                })
            def zrow_def(r):
                return pd.Series({
                    "z_pts_allow(inv)": inv_def(r,"PTS_perG_Allowed"),
                    "z_pass_allow(inv)": inv_def(r,"Pass_YDS_perG_Allowed"),
                    "z_rush_allow(inv)": inv_def(r,"Rush_YDS_perG_Allowed"),
                    "z_tot_allow(inv)": inv_def(r,"Tot_YDS_perG_Allowed"),
                })

            st.markdown("**Z-scores (higher is better)**")
            zdf = pd.concat([
                zrow_off(h_off).rename(f"{home} OFF"),
                zrow_def(h_def).rename(f"{home} DEF(inv)"),
                zrow_off(a_off).rename(f"{away} OFF"),
                zrow_def(a_def).rename(f"{away} DEF(inv)"),
            ], axis=1).T
            st.dataframe(zdf)

            st.markdown("**Edges (anti-symmetric, pre-clip)**")
            st.dataframe(pd.DataFrame([raw_edges]))
elif home == away:
    st.warning("Please select two different teams.")



# streamlit_ncaab_predictor.py
# Streamlit-ready NCAAB matchup predictor (fits model from ncaab.csv, then predicts A vs B)
# Run: streamlit run streamlit_ncaab_predictor.py

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# =========================
# Data + Model Config
# =========================
CSV_PATH = "/mnt/data/ncaab.csv"  # <-- your uploaded file path

FEATURE_COLS = [
    "conference strength",  # # of March Madness teams from the conference
    "ortg",
    "drtg",
    "pythagorean wins",
    "3ppg",
    "ppg",
    "fg%",
    "3p%",
    "ft%",
    "reb",
    "ast",
]

TEAM_COL = "team"
RECORD_COL = "record"

DEFAULT_EMPHASIS = {
    "conference strength": 1.60,
    "drtg": 1.55,
    "ortg": 1.55,
    "pythagorean wins": 1.45,
    "3ppg": 1.35,
}

DEFAULT_HOME_ADV_LOGIT = 0.12  # tweak if you want more/less home edge


# =========================
# Helper functions
# =========================
def parse_record_to_win_pct(record: str) -> float:
    # expects "22-5"
    w, l = record.split("-")
    w, l = int(w), int(l)
    games = max(w + l, 1)
    return w / games


def clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1 - eps))


def logit(p: float) -> float:
    p = clip01(p)
    return float(np.log(p / (1 - p)))


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def apply_emphasis(X: pd.DataFrame, emphasis: dict) -> pd.DataFrame:
    X2 = X.copy()
    for col, factor in emphasis.items():
        if col in X2.columns:
            X2[col] = X2[col].astype(float) * float(factor)
    return X2


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if TEAM_COL not in df.columns:
        raise ValueError(f"Expected '{TEAM_COL}' column in CSV.")
    if RECORD_COL not in df.columns:
        raise ValueError(f"Expected '{RECORD_COL}' column like '22-5' in CSV.")

    # target: win%
    df["win_pct"] = df[RECORD_COL].astype(str).apply(parse_record_to_win_pct)

    # ensure numeric
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing expected feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURE_COLS + ["win_pct"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def fit_model(df: pd.DataFrame, emphasis: dict, ridge_alpha: float) -> Pipeline:
    X = df[FEATURE_COLS].copy()
    y = df["win_pct"].astype(float).values

    X = apply_emphasis(X, emphasis)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(ridge_alpha), random_state=42)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict_team_win_pct(model: Pipeline, team_stats: dict, emphasis: dict) -> float:
    X = pd.DataFrame([team_stats], columns=FEATURE_COLS)
    X = apply_emphasis(X, emphasis)
    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 0.05, 0.95))


def predict_matchup(model: Pipeline, teamA_stats: dict, teamB_stats: dict, emphasis: dict, home_team: str) -> dict:
    pA = predict_team_win_pct(model, teamA_stats, emphasis)
    pB = predict_team_win_pct(model, teamB_stats, emphasis)

    sA = logit(pA)
    sB = logit(pB)

    adv = 0.0
    if home_team == "Team A":
        adv = DEFAULT_HOME_ADV_LOGIT
    elif home_team == "Team B":
        adv = -DEFAULT_HOME_ADV_LOGIT

    probA = sigmoid((sA - sB) + adv)
    return {
        "teamA_win_prob": probA,
        "teamB_win_prob": 1.0 - probA,
        "teamA_model_win_pct": pA,
        "teamB_model_win_pct": pB,
        "teamA_strength_logit": sA,
        "teamB_strength_logit": sB,
        "home_adv_logit_used": adv,
    }


def get_team_row(df: pd.DataFrame, team_name: str) -> pd.Series:
    row = df.loc[df[TEAM_COL].astype(str).str.lower().eq(str(team_name).lower())]
    if row.empty:
        raise ValueError(f"Team not found: {team_name}")
    return row.iloc[0]


def stats_editor(df: pd.DataFrame, team_name: str, prefix: str) -> dict:
    row = get_team_row(df, team_name)

    st.markdown(f"**{prefix} stats inputs** (edit anything you want)")

    cols = st.columns(3)

    # Put your “emphasis” stats in the first spots so they’re most visible
    key_order = [
        "conference strength",
        "ortg",
        "drtg",
        "pythagorean wins",
        "3ppg",
        "ppg",
        "fg%",
        "3p%",
        "ft%",
        "reb",
        "ast",
    ]

    out = {}
    for i, k in enumerate(key_order):
        with cols[i % 3]:
            default_val = float(row[k])
            # Conference strength is an integer conceptually
            if k == "conference strength":
                out[k] = float(
                    st.number_input(
                        f"{prefix} {k}",
                        value=int(round(default_val)),
                        min_value=0,
                        max_value=20,
                        step=1,
                        key=f"{prefix}_{k}",
                        help="Number of teams from this conference in March Madness (use your own estimate).",
                    )
                )
            else:
                out[k] = float(
                    st.number_input(
                        f"{prefix} {k}",
                        value=default_val,
                        step=0.1,
                        key=f"{prefix}_{k}",
                    )
                )
    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="NCAAB Predictor", layout="wide")

st.title("🏀 College Basketball Matchup Predictor (NCAAB)")
st.caption(
    "Trains a simple model from your `ncaab.csv` team stats, then predicts Team A vs Team B. "
    "You can edit inputs — especially conference strength (# tourney teams), ORtg, DRtg, Pythagorean wins, and 3PPG."
)

# Sidebar: model knobs
st.sidebar.header("Model settings")

ridge_alpha = st.sidebar.slider("Ridge alpha (stability)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

st.sidebar.subheader("Feature emphasis (multiplier)")
emphasis = {}
for k, v in DEFAULT_EMPHASIS.items():
    emphasis[k] = st.sidebar.slider(k, min_value=0.8, max_value=2.5, value=float(v), step=0.05)

st.sidebar.caption("Higher multipliers = those features influence the model more.")

# Load data
try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# Fit model
model = fit_model(df, emphasis, ridge_alpha)

# Team selectors
teams = sorted(df[TEAM_COL].astype(str).unique().tolist())

top = st.columns([2, 2, 1, 1])
with top[0]:
    teamA_name = st.selectbox("Team A", teams, index=0)
with top[1]:
    default_idx = 1 if len(teams) > 1 else 0
    teamB_name = st.selectbox("Team B", teams, index=default_idx)
with top[2]:
    home_team = st.selectbox("Home court", ["Neutral", "Team A", "Team B"], index=0)
with top[3]:
    show_details = st.checkbox("Show model details", value=False)

# Input editors (pull defaults from CSV but editable)
left, right = st.columns(2)
with left:
    teamA_stats = stats_editor(df, teamA_name, "Team A")
with right:
    teamB_stats = stats_editor(df, teamB_name, "Team B")

# Predict
if st.button("Predict matchup", type="primary", use_container_width=True):
    if teamA_name == teamB_name:
        st.warning("Pick two different teams.")
    else:
        out = predict_matchup(model, teamA_stats, teamB_stats, emphasis, home_team)

        probA = out["teamA_win_prob"]
        probB = out["teamB_win_prob"]

        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(f"{teamA_name} win probability", f"{probA*100:.1f}%")
        with c2:
            st.metric(f"{teamB_name} win probability", f"{probB*100:.1f}%")
        with c3:
            # simple “edge” indicator
            edge = (probA - 0.5) * 100
            st.metric("Edge (Team A vs 50/50)", f"{edge:+.1f}%")

        if show_details:
            st.divider()
            st.markdown("**Model details**")
            st.write(
                {
                    "Team A model win%": round(out["teamA_model_win_pct"], 4),
                    "Team B model win%": round(out["teamB_model_win_pct"], 4),
                    "Team A strength (logit)": round(out["teamA_strength_logit"], 4),
                    "Team B strength (logit)": round(out["teamB_strength_logit"], 4),
                    "Home adv logit used": round(out["home_adv_logit_used"], 4),
                    "Emphasis multipliers": emphasis,
                    "Ridge alpha": ridge_alpha,
                }
            )

        st.divider()
        st.markdown("**Current inputs used**")
        preview = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                f"{teamA_name} (A)": [teamA_stats[c] for c in FEATURE_COLS],
                f"{teamB_name} (B)": [teamB_stats[c] for c in FEATURE_COLS],
            }
        )
        st.dataframe(preview, use_container_width=True)

# Small footer info
st.caption(
    "Note: conference strength is treated as your input = number of March Madness teams from that conference. "
    "If you want this to auto-update by season, tell me what source you’re using and I’ll wire it in."
)
