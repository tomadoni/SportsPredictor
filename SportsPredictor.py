# SportsPredictor.py
# CLEAN single-file Streamlit app (NBA + NHL + MLB + NFL + NCAAB)
# - One st.set_page_config at the top (required)
# - Each sport lives in a render_*() function so nothing “runs” until selected
# - Unique widget keys per sport to avoid DuplicateWidgetID errors
# - Your existing logic is preserved as closely as possible, just organized safely

import streamlit as st

st.set_page_config(page_title="Sports Predictor", layout="wide")  # MUST be first Streamlit call

import pandas as pd
import numpy as np
import glob
import re
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, log_loss, brier_score_loss, roc_auc_score

from scipy.special import expit  # sigmoid


# =========================================================
# Shared helpers
# =========================================================
def _find_one(possibles):
    for p in possibles:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def _canonize_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip()).lower()
    s = s.replace("%", "pct")
    s = s.replace("3p", "threep").replace("3-pt", "threep").replace("3 pt", "threep")
    s = re.sub(r"[/\\\-]", " ", s)
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


# =========================================================
# NBA (your conservative clipped edges approach)
# =========================================================
def _auto_rename_nba(df: pd.DataFrame, is_defense: bool) -> pd.DataFrame:
    norm2orig = {_canonize_name(c): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm2orig:
                return norm2orig[c]
        return None

    team_col = pick("team", "teams", "franchise", "club", "squad")
    pts_perg = pick("pts_perg", "points_per_game", "pts_g", "ppg", "pts")
    fg_pct = pick("fg_pct", "field_goal_pct", "field_goals_pct", "fgp", "fg_percentage")
    threep_pct = pick("threep_pct", "three_point_pct", "three_point_percentage", "3p_pct", "3pt_pct")
    reb_perg = pick("reb_perg", "rebounds_per_game", "rebs_perg", "rpg", "trb_perg", "trb_g", "reb")

    rn = {}
    if team_col:
        rn[team_col] = "Team"
    if is_defense:
        if pts_perg:
            rn[pts_perg] = "PTS_perG_Allowed"
        if fg_pct:
            rn[fg_pct] = "FG_pct_Allowed"
        if threep_pct:
            rn[threep_pct] = "ThreeP_pct_Allowed"
        if reb_perg:
            rn[reb_perg] = "REB_perG_Allowed"
    else:
        if pts_perg:
            rn[pts_perg] = "PTS_perG"
        if fg_pct:
            rn[fg_pct] = "FG_pct"
        if threep_pct:
            rn[threep_pct] = "ThreeP_pct"
        if reb_perg:
            rn[reb_perg] = "REB_perG"

    return df.rename(columns=rn)


@st.cache_data(show_spinner=False)
def _nba_load_frames(off_path: str, def_path: str):
    off = pd.read_csv(off_path)
    de = pd.read_csv(def_path)
    off = _auto_rename_nba(off, is_defense=False)
    de = _auto_rename_nba(de, is_defense=True)
    return off, de


@st.cache_data(show_spinner=False)
def _nba_learn_caps_from_grid(off: pd.DataFrame, defn: pd.DataFrame, features: list, q=0.90, default_cap=0.5):
    off_means = off.drop(columns=["Team"]).mean(numeric_only=True)
    off_stds = off.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)
    def_means = defn.drop(columns=["Team"]).mean(numeric_only=True)
    def_stds = defn.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)

    off_d = {r.Team: r for _, r in off.iterrows()}
    def_d = {r.Team: r for _, r in defn.iterrows()}
    teams = sorted(set(off["Team"]).intersection(defn["Team"]))

    def z_off(row, col):
        return float((row[col] - off_means[col]) / off_stds[col])

    def inv_def(row, col_allowed):
        return float(-((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed]))

    def build_edges(h_off, a_off, h_def, a_def):
        x = {
            "edge_pts": z_off(h_off, "PTS_perG") - inv_def(a_def, "PTS_perG_Allowed"),
            "edge_fg": z_off(h_off, "FG_pct") - inv_def(a_def, "FG_pct_Allowed"),
            "edge_3p": z_off(h_off, "ThreeP_pct") - inv_def(a_def, "ThreeP_pct_Allowed"),
            "edge_reb": z_off(h_off, "REB_perG") - inv_def(a_def, "REB_perG_Allowed"),
            "edge_pts_def": inv_def(h_def, "PTS_perG_Allowed") - z_off(a_off, "PTS_perG"),
            "edge_fg_def": inv_def(h_def, "FG_pct_Allowed") - z_off(a_off, "FG_pct"),
            "edge_3p_def": inv_def(h_def, "ThreeP_pct_Allowed") - z_off(a_off, "ThreeP_pct"),
            "edge_reb_def": inv_def(h_def, "REB_perG_Allowed") - z_off(a_off, "REB_perG"),
        }
        x["edge_shooting_gap"] = x["edge_fg"] + x["edge_3p"]
        x["edge_pts_combo"] = x["edge_pts"] + x["edge_pts_def"]
        x["edge_reb_combo"] = x["edge_reb"] + x["edge_reb_def"]
        return x

    rows = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            rows.append(build_edges(off_d[h], off_d[a], def_d[h], def_d[a]))

    df = pd.DataFrame(rows)
    caps = {}
    for c in features:
        if c not in df or df[c].dropna().empty:
            caps[c] = default_cap
            continue
        cap = float(np.nanpercentile(np.abs(df[c].values), int(q * 100)))
        if not np.isfinite(cap) or cap < default_cap:
            cap = default_cap
        caps[c] = cap

    return caps, teams, off_d, def_d, off_means, off_stds, def_means, def_stds


def render_nba():
    st.header("🏀 NBA Matchup Predictor")

    off_path = _find_one(["NBA_Offense.csv", "data/NBA_Offense.csv"])
    def_path = _find_one(["NBA_Defense.csv", "data/NBA_Defense.csv"])
    if not off_path or not def_path:
        st.error("Missing NBA CSV files (`NBA_Offense.csv`, `NBA_Defense.csv`).")
        return

    off, defn = _nba_load_frames(off_path, def_path)

    off_req = {"Team", "PTS_perG", "FG_pct", "ThreeP_pct", "REB_perG"}
    def_req = {"Team", "PTS_perG_Allowed", "FG_pct_Allowed", "ThreeP_pct_Allowed", "REB_perG_Allowed"}
    missing = []
    if not off_req.issubset(off.columns):
        missing.append(f"Offense CSV missing: {off_req - set(off.columns)}")
    if not def_req.issubset(defn.columns):
        missing.append(f"Defense CSV missing: {def_req - set(defn.columns)}")
    if missing:
        st.error("Column mismatch:\n- " + "\n- ".join(missing))
        return

    FEATURES = [
        "edge_pts",
        "edge_fg",
        "edge_3p",
        "edge_reb",
        "edge_pts_def",
        "edge_fg_def",
        "edge_3p_def",
        "edge_reb_def",
        "edge_shooting_gap",
        "edge_pts_combo",
        "edge_reb_combo",
    ]

    CAPS, teams, off_d, def_d, off_means, off_stds, def_means, def_stds = _nba_learn_caps_from_grid(
        off, defn, FEATURES, q=0.90, default_cap=0.5
    )

    def z_off(row, col):
        return float((row[col] - off_means[col]) / off_stds[col])

    def inv_def(row, col_allowed):
        return float(-((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed]))

    def build_edges(h_off, a_off, h_def, a_def):
        x = {
            "edge_pts": z_off(h_off, "PTS_perG") - inv_def(a_def, "PTS_perG_Allowed"),
            "edge_fg": z_off(h_off, "FG_pct") - inv_def(a_def, "FG_pct_Allowed"),
            "edge_3p": z_off(h_off, "ThreeP_pct") - inv_def(a_def, "ThreeP_pct_Allowed"),
            "edge_reb": z_off(h_off, "REB_perG") - inv_def(a_def, "REB_perG_Allowed"),
            "edge_pts_def": inv_def(h_def, "PTS_perG_Allowed") - z_off(a_off, "PTS_perG"),
            "edge_fg_def": inv_def(h_def, "FG_pct_Allowed") - z_off(a_off, "FG_pct"),
            "edge_3p_def": inv_def(h_def, "ThreeP_pct_Allowed") - z_off(a_off, "ThreeP_pct"),
            "edge_reb_def": inv_def(h_def, "REB_perG_Allowed") - z_off(a_off, "REB_perG"),
        }
        x["edge_shooting_gap"] = x["edge_fg"] + x["edge_3p"]
        x["edge_pts_combo"] = x["edge_pts"] + x["edge_pts_def"]
        x["edge_reb_combo"] = x["edge_reb"] + x["edge_reb_def"]
        return x

    def clip_feats(d):
        return {k: float(np.clip(v, -CAPS.get(k, 999), CAPS.get(k, 999))) for k, v in d.items()}

    FIXED_WEIGHTS = {
        "edge_pts": 0.40,
        "edge_fg": 0.22,
        "edge_3p": 0.18,
        "edge_reb": 0.08,
        "edge_pts_def": 0.24,
        "edge_fg_def": 0.12,
        "edge_3p_def": 0.12,
        "edge_reb_def": 0.08,
        "edge_shooting_gap": 0.10,
        "edge_pts_combo": 0.14,
        "edge_reb_combo": 0.04,
    }

    HOME_EDGE = 0.18
    TEMP_K = 0.90
    SHRINK_L = 0.65
    CLAMP_MIN = 0.10
    CLAMP_MAX = 0.90

    def logistic(x, k=TEMP_K):
        return 1.0 / (1.0 + np.exp(-k * x))

    def score_from_edges(ed, weights):
        s = 0.0
        for f, w in weights.items():
            s += w * ed.get(f, 0.0)
        return float(s)

    home = st.selectbox("Home Team", teams, index=0, key="nba_home")
    away = st.selectbox("Away Team", [t for t in teams if t != home], index=0, key="nba_away")

    if home and away and home != away:
        h_off, a_off = off_d[home], off_d[away]
        h_def, a_def = def_d[home], def_d[away]

        raw_h = build_edges(h_off, a_off, h_def, a_def)
        x_h = clip_feats(raw_h)
        s_h = score_from_edges(x_h, FIXED_WEIGHTS)

        raw_a = build_edges(a_off, h_off, a_def, h_def)
        x_a = clip_feats(raw_a)
        s_a = score_from_edges(x_a, FIXED_WEIGHTS)

        sym_score = 0.5 * (s_h - s_a) + HOME_EDGE

        p_home = float(logistic(sym_score, k=TEMP_K))
        p_home = 0.5 + SHRINK_L * (p_home - 0.5)
        p_home = float(np.clip(p_home, CLAMP_MIN, CLAMP_MAX))
        p_away = 1.0 - p_home

        winner = home if p_home >= p_away else away

        st.subheader("Prediction")
        st.success(f"Predicted Winner: {winner}")
        c1, c2 = st.columns(2)
        c1.metric(f"{home} Win Probability", f"{p_home * 100:.1f}%")
        c2.metric(f"{away} Win Probability", f"{p_away * 100:.1f}%")

        with st.expander("Feature edges (clipped)"):
            show = pd.DataFrame([x_h]).T.rename(columns={0: "home_view_edge"})
            st.dataframe(show, use_container_width=True)
    else:
        st.info("Select two different teams.")


# =========================================================
# NHL (your calibrated logistic regression approach)
# =========================================================
@st.cache_data(show_spinner=False)
def _nhl_load_training(path: str):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def _nhl_fit_model(df: pd.DataFrame):
    raw_features = ["GF/G", "S%", "PP%", "PIM", "SOG", "GA/G", "SV%", "PK%"]
    diff_features = ["diff_" + f for f in raw_features]
    model_features = diff_features + ["home_indicator"]

    X_stats = df[diff_features]
    X_home = df[["home_indicator"]]
    y = df["Winner"]

    scaler = StandardScaler()
    X_stats_scaled = scaler.fit_transform(X_stats)
    X_combined = np.hstack([X_stats_scaled, X_home.values])

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_combined, y, test_size=0.2, stratify=y, random_state=42
    )

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    calibrated = CalibratedClassifierCV(estimator=lr, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    # metrics (optional)
    y_pred_train = calibrated.predict(X_train)
    y_pred_test = calibrated.predict(X_calib)
    metrics = {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_calib, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_calib, y_pred_test),
        "logloss": log_loss(y_calib, calibrated.predict_proba(X_calib)),
        "brier": brier_score_loss(y_calib, calibrated.predict_proba(X_calib)[:, 1]),
        "auc": roc_auc_score(y_calib, calibrated.predict_proba(X_calib)[:, 1]),
    }

    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    return calibrated, scaler, diff_features, teams, metrics


def render_nhl():
    st.header("🏒 NHL Matchup Predictor")

    path = "NHL_Matchup_Training_Data (1).csv"
    try:
        df = _nhl_load_training(path)
    except Exception as e:
        st.error(f"Could not load NHL data: {e}")
        return

    calibrated_model, scaler, diff_features, teams, metrics = _nhl_fit_model(df)

    st.caption("Predict NHL matchups using stat differentials and home-ice advantage.")

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Home Team (NHL)", teams, key="nhl_home")
    with col2:
        away = st.selectbox("✈️ Away Team (NHL)", teams, key="nhl_away")

    if home != away:
        matchup = df[(df["home_team"] == home) & (df["away_team"] == away)]
        if matchup.empty:
            st.error("Not enough data for this matchup.")
            return

        stat_diff = np.clip(matchup[diff_features].iloc[0].values, -0.05, 0.05)
        input_scaled = scaler.transform([stat_diff])

        home_indicator = matchup["home_indicator"].iloc[0] * -20  # your original behavior
        input_combined = np.hstack([input_scaled, [[home_indicator]]])

        prob = calibrated_model.predict_proba(input_combined)[0]
        prob_home = float(prob[1])
        prob_away = float(prob[0])
        winner = home if prob_home > prob_away else away

        st.subheader("📈 Prediction Result")
        st.success(f"Predicted Winner: {winner}")
        c1, c2 = st.columns(2)
        c1.metric(f"{home} Win Probability", f"{prob_home * 100:.1f}%")
        c2.metric(f"{away} Win Probability", f"{prob_away * 100:.1f}%")

        with st.expander("Model metrics (debug)"):
            st.write({k: round(v, 4) for k, v in metrics.items()})
    else:
        st.warning("Please select two different teams.")


# =========================================================
# MLB (your run differential based logistic regression)
# =========================================================
@st.cache_data(show_spinner=False)
def _mlb_load_data(stats_path: str, games_path: str):
    stats = (
        pd.read_csv(stats_path)
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    games = pd.read_csv(games_path, encoding="ISO-8859-1")
    return stats, games


@st.cache_resource(show_spinner=False)
def _mlb_fit_model(stats: pd.DataFrame, games: pd.DataFrame):
    FEATURES = ["OPS", "OBP", "SLG", "ERA", "WHIP", "SO", "BB"]
    CLIP_MIN, CLIP_MAX = -3.0, 3.0
    HOME_LOGIT_BONUS = 0.20

    teams = sorted(pd.unique(pd.concat([games["Home"], games["Away"]], ignore_index=True).dropna()))

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

    games2 = games.copy()
    games2["home_win"] = (games2["Home Score"] > games2["Away Score"]).astype(int)

    rows, labels = [], []
    for _, g in games2.iterrows():
        home = g.get("Home")
        away = g.get("Away")
        if home not in team_stats.index or away not in team_stats.index:
            continue
        h = team_stats.loc[home]
        a = team_stats.loc[away]
        diff = [h[f] - a[f] for f in FEATURES]
        rows.append(diff)
        labels.append(g["home_win"])

    diff_features = [f"diff_{f}" for f in FEATURES]
    X_df = pd.DataFrame(rows, columns=diff_features)
    y = pd.Series(labels, name="home_win")

    X_clipped = X_df.clip(lower=CLIP_MIN, upper=CLIP_MAX)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped.values)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    return teams, team_stats, model, scaler, FEATURES, CLIP_MIN, CLIP_MAX, HOME_LOGIT_BONUS


def render_mlb():
    st.header("⚾ MLB Matchup Predictor")

    stats_path = "MLB_Run_Differential_Clean_FIXED.csv"
    games_path = "mlb-2025-asplayed.csv"

    try:
        stats, games = _mlb_load_data(stats_path, games_path)
    except Exception as e:
        st.error(f"Could not load MLB files: {e}")
        return

    teams, team_stats, model, scaler, FEATURES, CLIP_MIN, CLIP_MAX, HOME_LOGIT_BONUS = _mlb_fit_model(stats, games)

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team (MLB)", teams, index=0 if teams else 0, key="mlb_home")
    with col2:
        away_team = st.selectbox("✈️ Away Team (MLB)", teams, index=1 if len(teams) > 1 else 0, key="mlb_away")

    if home_team and away_team:
        if home_team == away_team:
            st.warning("Please select two different teams.")
            return
        if home_team not in team_stats.index or away_team not in team_stats.index:
            st.error("Could not find team stats for one or both teams.")
            return

        h = team_stats.loc[home_team]
        a = team_stats.loc[away_team]
        diff = np.array([h[f] - a[f] for f in FEATURES], dtype=float)
        clipped = np.clip(diff, CLIP_MIN, CLIP_MAX)
        scaled = scaler.transform([clipped])[0]

        home_logit = model.decision_function([scaled])[0] + HOME_LOGIT_BONUS
        prob_home = float(expit(home_logit))
        prob_away = 1.0 - prob_home
        winner = home_team if prob_home >= prob_away else away_team

        st.subheader("📈 Prediction")
        st.success(f"Predicted Winner: {winner}")
        c1, c2 = st.columns(2)
        c1.metric(f"{home_team} Win Probability", f"{prob_home * 100:.1f}%")
        c2.metric(f"{away_team} Win Probability", f"{prob_away * 100:.1f}%")

        with st.expander("Inputs (diff features)"):
            st.dataframe(pd.DataFrame([clipped], columns=[f"diff_{f}" for f in FEATURES]), use_container_width=True)


# =========================================================
# NFL (your direct edges approach)
# =========================================================
def _auto_rename_nfl(df: pd.DataFrame, is_defense: bool) -> pd.DataFrame:
    norm2orig = {_canonize_name(c): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in norm2orig:
                return norm2orig[c]
        return None

    team_col = pick("team", "teams")
    tot_perg = pick("tot_yds_perg", "total_yds_perg", "yds_g", "yds_perg", "total_yards_per_game")
    pass_perg = pick("pass_yds_perg", "passing_yds_perg", "pass_yds_g", "passing_yards_per_game")
    rush_perg = pick("rush_yds_perg", "rushing_yds_perg", "rush_yds_g", "rushing_yards_per_game")
    pts_perg = pick("pts_perg", "points_per_game", "pts_g")

    rename_map = {}
    if team_col:
        rename_map[team_col] = "Team"
    if is_defense:
        if tot_perg:
            rename_map[tot_perg] = "Tot_YDS_perG_Allowed"
        if pass_perg:
            rename_map[pass_perg] = "Pass_YDS_perG_Allowed"
        if rush_perg:
            rename_map[rush_perg] = "Rush_YDS_perG_Allowed"
        if pts_perg:
            rename_map[pts_perg] = "PTS_perG_Allowed"
    else:
        if tot_perg:
            rename_map[tot_perg] = "Tot_YDS_perG"
        if pass_perg:
            rename_map[pass_perg] = "Pass_YDS_perG"
        if rush_perg:
            rename_map[rush_perg] = "Rush_YDS_perG"
        if pts_perg:
            rename_map[pts_perg] = "PTS_perG"

    out = df.rename(columns=rename_map)

    for c in list(out.columns):
        if _canonize_name(c) == "rush_yds_perg_allowed" and c != "Rush_YDS_perG_Allowed":
            out = out.rename(columns={c: "Rush_YDS_perG_Allowed"})

    return out


@st.cache_data(show_spinner=False)
def _nfl_load_frames(off_path: str, def_path: str):
    off = pd.read_csv(off_path)
    defn = pd.read_csv(def_path)
    off = _auto_rename_nfl(off, is_defense=False)
    defn = _auto_rename_nfl(defn, is_defense=True)
    return off, defn


def render_nfl():
    st.header("🏈 NFL Matchup Predictor (No Training • Pure Stat Edges)")

    off_path = _find_one(["NFL_Offense.csv", "data/NFL_Offense.csv"])
    def_path = _find_one(["NFL_Defense.csv", "data/NFL_Defense.csv"])
    if not off_path or not def_path:
        st.error("Missing NFL CSV files (`NFL_Offense.csv`, `NFL_Defense.csv`).")
        return

    off, defn = _nfl_load_frames(off_path, def_path)

    off_req = {"Team", "Tot_YDS_perG", "Pass_YDS_perG", "Rush_YDS_perG", "PTS_perG"}
    def_req = {"Team", "Tot_YDS_perG_Allowed", "Pass_YDS_perG_Allowed", "Rush_YDS_perG_Allowed", "PTS_perG_Allowed"}
    missing = []
    if not off_req.issubset(off.columns):
        missing.append(f"Offense CSV missing: {off_req - set(off.columns)}")
    if not def_req.issubset(defn.columns):
        missing.append(f"Defense CSV missing: {def_req - set(defn.columns)}")
    if missing:
        st.error("Column mismatch:\n- " + "\n- ".join(missing))
        return

    off_means = off.drop(columns=["Team"]).mean(numeric_only=True)
    off_stds = off.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)
    def_means = defn.drop(columns=["Team"]).mean(numeric_only=True)
    def_stds = defn.drop(columns=["Team"]).std(numeric_only=True).replace(0, 1.0)

    off_d = {r.Team: r for _, r in off.iterrows()}
    def_d = {r.Team: r for _, r in defn.iterrows()}
    teams = sorted(set(off["Team"]).intersection(defn["Team"]))

    def z_off(row, col):
        return (row[col] - off_means[col]) / off_stds[col]

    def inv_def(row, col_allowed):
        return -((row[col_allowed] - def_means[col_allowed]) / def_stds[col_allowed])

    FEATURES = ["edge_pts", "edge_pass", "edge_rush", "edge_tot", "edge_pass_minus_rush"]

    def build_edges(h_off, a_off, h_def, a_def):
        home_pts_att = z_off(h_off, "PTS_perG") - inv_def(a_def, "PTS_perG_Allowed")
        away_pts_att = z_off(a_off, "PTS_perG") - inv_def(h_def, "PTS_perG_Allowed")
        home_pass_att = z_off(h_off, "Pass_YDS_perG") - inv_def(a_def, "Pass_YDS_perG_Allowed")
        away_pass_att = z_off(a_off, "Pass_YDS_perG") - inv_def(h_def, "Pass_YDS_perG_Allowed")
        home_rush_att = z_off(h_off, "Rush_YDS_perG") - inv_def(a_def, "Rush_YDS_perG_Allowed")
        away_rush_att = z_off(a_off, "Rush_YDS_perG") - inv_def(h_def, "Rush_YDS_perG_Allowed")
        home_tot_att = z_off(h_off, "Tot_YDS_perG") - inv_def(a_def, "Tot_YDS_perG_Allowed")
        away_tot_att = z_off(a_off, "Tot_YDS_perG") - inv_def(h_def, "Tot_YDS_perG_Allowed")

        edge_pts = home_pts_att - away_pts_att
        edge_pass = home_pass_att - away_pass_att
        edge_rush = home_rush_att - away_rush_att
        edge_tot = home_tot_att - away_tot_att

        x = {"edge_pts": edge_pts, "edge_pass": edge_pass, "edge_rush": edge_rush, "edge_tot": edge_tot}
        x["edge_pass_minus_rush"] = x["edge_pass"] - x["edge_rush"]
        return x

    def learn_caps_from_off_def(q=0.90):
        rows = []
        for h_team in teams:
            for a_team in teams:
                if h_team == a_team:
                    continue
                rows.append(build_edges(off_d[h_team], off_d[a_team], def_d[h_team], def_d[a_team]))
        df_caps = pd.DataFrame(rows)
        caps = {}
        for c in FEATURES:
            if df_caps[c].dropna().empty:
                caps[c] = 0.5
                continue
            cap = float(np.nanpercentile(np.abs(df_caps[c].values), q * 100))
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
        net = (0.5 * x["edge_pts"] + 0.3 * x["edge_tot"] + 0.2 * x["edge_pass_minus_rush"])
        return float(1.0 / (1.0 + np.exp(-k * net))), edges

    home = st.selectbox("Home Team", teams, key="nfl_home")
    away = st.selectbox("Away Team", teams, key="nfl_away")

    if home and away and home != away:
        if home not in off_d or away not in off_d or home not in def_d or away not in def_d:
            st.error("Missing team stats for one or both teams.")
            return

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

        with st.expander("Debug: features"):
            st.dataframe(pd.DataFrame([raw_edges]), use_container_width=True)
    elif home == away:
        st.warning("Please select two different teams.")




# =========================================================
# NCAAB (your requested emphasis-based team model -> matchup prob)
# =========================================================
def _ncaab_parse_record_to_win_pct(record: str) -> float:
    w, l = str(record).split("-")
    w, l = int(w), int(l)
    g = max(w + l, 1)
    return w / g


def _ncaab_apply_emphasis(X: pd.DataFrame, emphasis: dict) -> pd.DataFrame:
    X2 = X.copy()
    for col, factor in emphasis.items():
        if col in X2.columns:
            X2[col] = X2[col].astype(float) * float(factor)
    return X2


def _ncaab_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _ncaab_sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


@st.cache_data(show_spinner=False)
def _ncaab_load_data(csv_path: str, team_col: str, record_col: str, feature_cols: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if team_col not in df.columns:
        raise ValueError(f"Expected '{team_col}' column in CSV.")
    if record_col not in df.columns:
        raise ValueError(f"Expected '{record_col}' column like '22-5' in CSV.")

    df["win_pct"] = df[record_col].astype(str).apply(_ncaab_parse_record_to_win_pct)

    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing expected feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=feature_cols + ["win_pct"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def _ncaab_fit_model(df: pd.DataFrame, feature_cols: list, emphasis: dict, ridge_alpha: float) -> Pipeline:
    X = df[feature_cols].copy()
    y = df["win_pct"].astype(float).values
    X = _ncaab_apply_emphasis(X, emphasis)
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(ridge_alpha), random_state=42))])
    pipe.fit(X, y)
    return pipe


def render_ncaab():
    import numpy as np
    import pandas as pd
    import streamlit as st
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    st.header("🏀 College Basketball Predictor (NCAAB)")
    st.caption("Pick teams → get probabilities. Extra weight on conference strength, DRtg, ORtg, Pyth wins, and 3PPG.")

    CSV_PATH = "ncaab.csv"
    TEAM_COL = "team"
    RECORD_COL = "record"

    # ✅ DRtg is inverted so higher = better
    FEATURE_COLS = [
        "conference strength",
        "ortg",
        "drtg_inv",
        "pythagorean wins",
        "3ppg",
        "ppg",
        "fg%",
        "3p%",
        "ft%",
        "reb",
        "ast",
    ]

    # ✅ bigger conference weight
    EMPHASIS = {
        "conference strength": 3.60,
        "drtg_inv": 1.55,
        "ortg": 1.55,
        "pythagorean wins": 1.45,
        "3ppg": 1.35,
    }

    RIDGE_ALPHA = 2.0
    HOME_ADV_LOGIT = 0.12
    CLAMP_MIN, CLAMP_MAX = 0.10, 0.90

    # ✅ optional: guaranteed-direction conference bump on final probability
    # Try 0.10–0.30. Higher = more conference-driven.
    CONF_LOGIT_BONUS = 0.18

    def parse_record_to_win_pct(record: str) -> float:
        w, l = str(record).split("-")
        w, l = int(w), int(l)
        g = max(w + l, 1)
        return w / g

    def apply_emphasis(X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()
        for col, factor in EMPHASIS.items():
            if col in X2.columns:
                X2[col] = X2[col].astype(float) * float(factor)
        return X2

    def logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def sigmoid(z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-z)))

    @st.cache_data(show_spinner=False)
    def load_df():
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]

        if TEAM_COL not in df.columns:
            raise ValueError(f"Missing '{TEAM_COL}' column.")
        if RECORD_COL not in df.columns:
            raise ValueError(f"Missing '{RECORD_COL}' column (like '22-5').")

        df["win_pct"] = df[RECORD_COL].astype(str).apply(parse_record_to_win_pct)

        # make sure base columns are numeric
        needed_raw = ["conference strength", "ortg", "drtg", "pythagorean wins", "3ppg", "ppg", "fg%", "3p%", "ft%", "reb", "ast"]
        for c in needed_raw:
            if c not in df.columns:
                raise ValueError(f"Missing feature column: {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ✅ invert DRtg (lower=better -> higher=better)
        df["drtg_inv"] = -df["drtg"]

        # now require final features
        df = df.dropna(subset=FEATURE_COLS + ["win_pct"]).reset_index(drop=True)
        return df

    @st.cache_resource(show_spinner=False)
    def fit_model(df: pd.DataFrame):
        X = apply_emphasis(df[FEATURE_COLS].copy())
        y = df["win_pct"].astype(float).values

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                # ✅ positive=True prevents "conference strength" from flipping negative
                ("ridge", Ridge(alpha=float(RIDGE_ALPHA), random_state=42, positive=True)),
            ]
        )
        pipe.fit(X, y)
        return pipe

    try:
        df = load_df()
    except Exception as e:
        st.error(f"NCAAB: Could not load {CSV_PATH}: {e}")
        return

    model = fit_model(df)
    teams = sorted(df[TEAM_COL].astype(str).unique().tolist())

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0, key="ncaab_home_team")
    with col2:
        away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], index=0, key="ncaab_away_team")
    with col3:
        site = st.selectbox("Site", ["Home", "Neutral"], index=0, key="ncaab_site")

    if home_team == away_team:
        st.warning("Pick two different teams.")
        return

    hrow = df.loc[df[TEAM_COL].astype(str).str.lower().eq(home_team.lower())].iloc[0]
    arow = df.loc[df[TEAM_COL].astype(str).str.lower().eq(away_team.lower())].iloc[0]

    home_stats = {c: float(hrow[c]) for c in FEATURE_COLS}
    away_stats = {c: float(arow[c]) for c in FEATURE_COLS}

    pH = float(model.predict(apply_emphasis(pd.DataFrame([home_stats], columns=FEATURE_COLS)))[0])
    pA = float(model.predict(apply_emphasis(pd.DataFrame([away_stats], columns=FEATURE_COLS)))[0])

    pH = float(np.clip(pH, 0.05, 0.95))
    pA = float(np.clip(pA, 0.05, 0.95))

    adv = HOME_ADV_LOGIT if site == "Home" else 0.0

    # ✅ guaranteed-direction conference adjustment (home conf - away conf)
    conf_delta = home_stats["conference strength"] - away_stats["conference strength"]
    conf_adj = CONF_LOGIT_BONUS * conf_delta

    prob_home = sigmoid((logit(pH) - logit(pA)) + adv + conf_adj)

    prob_home = float(np.clip(prob_home, CLAMP_MIN, CLAMP_MAX))
    prob_away = 1.0 - prob_home
    winner = home_team if prob_home >= prob_away else away_team

    st.subheader("Prediction")
    st.success(f"Predicted Winner: {winner}")
    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} Win Probability", f"{prob_home * 100:.1f}%")
    c2.metric(f"{away_team} Win Probability", f"{prob_away * 100:.1f}%")

    with st.expander("Debug (optional)"):
        st.write("Conference delta (home - away):", conf_delta)
        st.write("Conference logit adj:", conf_adj)


# =========================================================
# Main App UI (clean selector)
# =========================================================
st.title("Sports Predictor App")
sport = st.selectbox(
    "Choose a sport",
    ["NBA", "NHL", "MLB", "NFL", "NCAAB"],
    index=0,
    key="sport_selector",
)

st.divider()

if sport == "NBA":
    render_nba()
elif sport == "NHL":
    render_nhl()
elif sport == "MLB":
    render_mlb()
elif sport == "NFL":
    render_nfl()
elif sport == "NCAAB":
    render_ncaab()
