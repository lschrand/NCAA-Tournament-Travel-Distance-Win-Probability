"""
🏀 NCAA Tournament: Travel Distance Analyzer
Streamlit application for exploring how travel distance affects NCAA tournament outcomes.

Usage:
    pip install streamlit pandas numpy scipy scikit-learn matplotlib
    streamlit run ncaa_travel_app.py

Place the following CSV files in the same directory:
    - Tournament Locations.csv
    - Tournament Matchups.csv
    - TeamRankings.csv
    - KenPom Preseason.csv
    - Conference Stats.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NCAA Tournament Analytics Hub",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
#  THEME COLORS & STYLING
# ═══════════════════════════════════════════════════════════════════════════

BG      = '#0f1117'
CARD    = '#161b22'
ORANGE  = '#f97316'
GREEN   = '#22c55e'
RED     = '#ef4444'
BLUE    = '#3b82f6'
CYAN    = '#06b6d4'
MUTED   = '#64748b'
LIGHT   = '#e2e8f0'
GRID_C  = '#1e2530'

def dark_style():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG,
        'axes.edgecolor': '#2a2f3a', 'axes.labelcolor': '#a0aec0',
        'text.color': LIGHT, 'xtick.color': '#a0aec0', 'ytick.color': '#a0aec0',
        'grid.color': GRID_C, 'grid.alpha': 0.5,
        'font.family': 'monospace', 'font.size': 10,
    })

# Inject custom CSS for dark theme cards
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { max-width: 1200px; padding-top: 1.5rem; }
    h1, h2, h3, h4, h5, h6, p, li, span, label, .stMarkdown {
        color: #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8 !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f97316 !important;
        color: #ffffff !important;
    }
    .stSlider > div > div > div { color: #e2e8f0 !important; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f97316 !important; }
    .stat-card {
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .stButton > button {
        background-color: #f97316 !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #ea580c !important;
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #161b22;
        border-color: #2a2f3a;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_process_data():
    """Load CSVs, merge, compute winners, train models."""
    # Find data files — look in current dir, then fallback
    data_dir = '.'
    for candidate in ['.', 'data', os.path.dirname(__file__) if '__file__' in dir() else '.']:
        if os.path.exists(os.path.join(candidate, 'Tournament Locations.csv')):
            data_dir = candidate
            break

    locations = pd.read_csv(os.path.join(data_dir, 'Tournament Locations.csv'))
    matchups  = pd.read_csv(os.path.join(data_dir, 'Tournament Matchups.csv'))

    # Merge
    merged = matchups.merge(
        locations[['YEAR', 'BY YEAR NO', 'TEAM', 'DISTANCE (MI)', 'DISTANCE (KM)',
                   'TIME ZONES CROSSED', 'DIRECTION', 'COLLEGE LATITUDE',
                   'COLLEGE LONGITUDE', 'GAME LATITUDE', 'GAME LONGITUDE',
                   'GAME CITY', 'GAME STATE']],
        on=['YEAR', 'BY YEAR NO', 'TEAM'], how='inner'
    )
    merged['WON'] = (merged['ROUND'] < merged['CURRENT ROUND']).astype(int)

    # Game pairs
    games_list = []
    for (year, cr), grp in merged.groupby(['YEAR', 'CURRENT ROUND']):
        grp = grp.sort_values('BY YEAR NO', ascending=False).reset_index(drop=True)
        for i in range(0, len(grp) - 1, 2):
            t1, t2 = grp.iloc[i], grp.iloc[i + 1]
            games_list.append({
                'YEAR': year, 'CURRENT ROUND': cr,
                'TEAM1': t1['TEAM'], 'SEED1': t1['SEED'],
                'DIST1': t1['DISTANCE (MI)'], 'WON1': t1['WON'], 'SCORE1': t1['SCORE'],
                'TEAM2': t2['TEAM'], 'SEED2': t2['SEED'],
                'DIST2': t2['DISTANCE (MI)'], 'WON2': t2['WON'], 'SCORE2': t2['SCORE'],
                'GAME_CITY': t1['GAME CITY'], 'GAME_STATE': t1['GAME STATE'],
            })
    games_df = pd.DataFrame(games_list)
    games_df['CLOSER_WON'] = np.where(
        games_df['DIST1'] < games_df['DIST2'], games_df['WON1'], games_df['WON2']
    )

    # Train models
    team_data = merged.copy()
    y_all = team_data['WON'].values

    sc1 = StandardScaler()
    X1s = sc1.fit_transform(team_data[['DISTANCE (MI)']].values)
    lr1 = LogisticRegression(random_state=42).fit(X1s, y_all)
    cv1 = cross_val_score(lr1, X1s, y_all, cv=5, scoring='roc_auc')

    sc2 = StandardScaler()
    X2s = sc2.fit_transform(team_data[['DISTANCE (MI)', 'SEED']].values)
    lr2 = LogisticRegression(random_state=42).fit(X2s, y_all)
    cv2 = cross_val_score(lr2, X2s, y_all, cv=5, scoring='roc_auc')

    sc3 = StandardScaler()
    X3s = sc3.fit_transform(team_data[['DISTANCE (MI)', 'SEED', 'TIME ZONES CROSSED']].values)
    lr3 = LogisticRegression(random_state=42).fit(X3s, y_all)
    cv3 = cross_val_score(lr3, X3s, y_all, cv=5, scoring='roc_auc')

    all_teams = sorted(team_data['TEAM'].unique())
    all_years = sorted(team_data['YEAR'].unique())

    # ── Load additional datasets ─────────────────────────────────────────
    team_rankings = pd.read_csv(os.path.join(data_dir, 'TeamRankings.csv'))
    kenpom = pd.read_csv(os.path.join(data_dir, 'KenPom Preseason.csv'))
    conf_stats = pd.read_csv(os.path.join(data_dir, 'Conference Stats.csv'))

    # ── Build enhanced merged dataset ────────────────────────────────────
    enhanced = team_data.copy()
    enhanced = enhanced.merge(
        team_rankings[['YEAR', 'TEAM NO', 'TR RANK', 'TR RATING', 'SOS RANK', 'SOS RATING',
                        'V 1-25 WINS', 'V 1-25 LOSS', 'V 26-50 WINS', 'V 26-50 LOSS',
                        'CONSISTENCY RANK', 'CONSISTENCY TR RATING',
                        'LUCK RANK', 'LUCK RATING']],
        on=['YEAR', 'TEAM NO'], how='left'
    )
    enhanced = enhanced.merge(
        kenpom[['YEAR', 'TEAM NO', 'PRESEASON KADJ EM', 'PRESEASON KADJ EM RANK',
                'PRESEASON KADJ O', 'PRESEASON KADJ D',
                'KADJ EM RANK CHANGE', 'KADJ EM CHANGE']],
        on=['YEAR', 'TEAM NO'], how='left'
    )

    # ── Train enhanced models ────────────────────────────────────────────
    # Model 4: TeamRankings features (available all years 2008-2025)
    features_tr = ['DISTANCE (MI)', 'SEED', 'TR RATING', 'SOS RATING']
    df_tr = enhanced.dropna(subset=features_tr)
    sc4 = StandardScaler()
    X4s = sc4.fit_transform(df_tr[features_tr].values)
    y4 = df_tr['WON'].values
    lr4 = LogisticRegression(random_state=42).fit(X4s, y4)
    cv4 = cross_val_score(lr4, X4s, y4, cv=5, scoring='roc_auc')

    # Model 5: Full model with KenPom (2012+ only)
    features_full = ['DISTANCE (MI)', 'SEED', 'TR RATING', 'SOS RATING',
                     'PRESEASON KADJ EM', 'KADJ EM CHANGE', 'V 1-25 WINS']
    df_full = enhanced.dropna(subset=features_full)
    sc5 = StandardScaler()
    X5s = sc5.fit_transform(df_full[features_full].values)
    y5 = df_full['WON'].values
    lr5 = LogisticRegression(random_state=42).fit(X5s, y5)
    cv5 = cross_val_score(lr5, X5s, y5, cv=5, scoring='roc_auc')

    return {
        'team_data': team_data, 'games_df': games_df,
        'y_all': y_all,
        'sc1': sc1, 'X1s': X1s, 'lr1': lr1, 'cv1': cv1,
        'sc2': sc2, 'X2s': X2s, 'lr2': lr2, 'cv2': cv2,
        'sc3': sc3, 'X3s': X3s, 'lr3': lr3, 'cv3': cv3,
        'all_teams': all_teams, 'all_years': all_years,
        # New data
        'enhanced': enhanced, 'team_rankings': team_rankings,
        'kenpom': kenpom, 'conf_stats': conf_stats,
        # Enhanced models
        'features_tr': features_tr, 'sc4': sc4, 'X4s': X4s, 'lr4': lr4, 'cv4': cv4, 'y4': y4,
        'features_full': features_full, 'sc5': sc5, 'X5s': X5s, 'lr5': lr5, 'cv5': cv5, 'y5': y5,
    }


# Load data
try:
    data = load_and_process_data()
    team_data = data['team_data']
    games_df = data['games_df']
    y_all = data['y_all']
    sc1, X1s, lr1, cv1 = data['sc1'], data['X1s'], data['lr1'], data['cv1']
    sc2, X2s, lr2, cv2 = data['sc2'], data['X2s'], data['lr2'], data['cv2']
    sc3, X3s, lr3, cv3 = data['sc3'], data['X3s'], data['lr3'], data['cv3']
    all_teams = data['all_teams']
    all_years = data['all_years']
    # New data
    enhanced = data['enhanced']
    team_rankings = data['team_rankings']
    kenpom = data['kenpom']
    conf_stats = data['conf_stats']
    # Enhanced models
    features_tr = data['features_tr']
    sc4, X4s, lr4, cv4, y4 = data['sc4'], data['X4s'], data['lr4'], data['cv4'], data['y4']
    features_full = data['features_full']
    sc5, X5s, lr5, cv5, y5 = data['sc5'], data['X5s'], data['lr5'], data['cv5'], data['y5']
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"**Error loading data:** {e}\n\nMake sure `Tournament Locations.csv`, "
             f"`Tournament Matchups.csv`, `TeamRankings.csv`, `KenPom Preseason.csv`, "
             f"and `Conference Stats.csv` are in the same directory as this script.")
    st.stop()

round_map = {64: 'Round of 64', 32: 'Round of 32', 16: 'Sweet 16',
             8: 'Elite 8', 4: 'Final 4', 2: 'Championship'}



# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🏀 NCAA Tournament Analytics Hub")
st.markdown(
    f"**Data-driven tools for predicting and analyzing March Madness matchups**  \n"
    f"`{len(team_data):,} observations · {len(games_df):,} games · "
    f"{min(all_years)}–{max(all_years)} · 5 ML models · Team, Conference & Distance analytics`"
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER TABS
# ═══════════════════════════════════════════════════════════════════════════

master_predict, master_travel, master_conf, master_models = st.tabs([
    "🎯 Predictions & Matchups",
    "📊 Travel & Distance Analysis",
    "🏛️ Conference Intelligence",
    "🤖 Models & Methodology",
])


# ─── MASTER TAB 1: PREDICTIONS & MATCHUPS ────────────────────────────────

with master_predict:
    tab_pred, tab_enhanced, tab_h2h, tab_team = st.tabs([
        "🎯 Win Predictor",
        "🔬 Enhanced Predictor",
        "⚔️ Head-to-Head",
        "🏫 Team Lookup",
    ])

    with tab_pred:
        st.subheader("Win Probability Predictor")
        st.caption("Adjust distance, seed, and round to see the predicted win probability.")

        c1, c2, c3 = st.columns(3)
        with c1:
            distance = st.slider("Distance Traveled (miles)", 0, 2500, 500, step=25, key="pred_dist")
        with c2:
            seed = st.slider("Seed", 1, 16, 4, step=1, key="pred_seed")
        with c3:
            tournament_round = st.selectbox("Tournament Round",
                ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship'],
                key="pred_round")

        rnd_val = {'Round of 64': 64, 'Round of 32': 32, 'Sweet 16': 16,
                   'Elite 8': 8, 'Final 4': 4, 'Championship': 2}[tournament_round]

        # Predict
        x_input = sc2.transform([[distance, seed]])
        prob = lr2.predict_proba(x_input)[0][1]
        baseline = lr2.predict_proba(sc2.transform([[0, seed]]))[0][1]
        delta = prob - baseline

        hist = team_data[(team_data['CURRENT ROUND'] == rnd_val) & (team_data['SEED'] == seed)]
        hist_wr = hist['WON'].mean() if len(hist) > 0 else None

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win Probability", f"{prob:.1%}")
        m2.metric("Baseline (0 mi)", f"{baseline:.1%}")
        m3.metric("Distance Penalty", f"{delta:+.1%}")
        if hist_wr is not None:
            m4.metric(f"Historical WR (Seed {seed})", f"{hist_wr:.1%}", help=f"n={len(hist)}")

        # Charts
        dark_style()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Gauge
        ax = axes[0]
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.3, 1.6); ax.set_aspect('equal'); ax.axis('off')
        theta_bg = np.linspace(np.pi, 0, 100)
        r_outer, r_inner = 1.2, 0.85
        for t in theta_bg:
            frac = (np.pi - t) / np.pi
            c = RED if frac < 0.4 else (ORANGE if frac < 0.6 else GREEN)
            ax.plot([r_inner * np.cos(t), r_outer * np.cos(t)],
                    [r_inner * np.sin(t), r_outer * np.sin(t)],
                    color=c, linewidth=3, alpha=0.2)
        needle_angle = np.pi * (1 - prob)
        ax.annotate('', xy=(1.0 * np.cos(needle_angle), 1.0 * np.sin(needle_angle)),
                    xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
        ax.plot(0, 0, 'o', color=ORANGE, markersize=8, zorder=5)
        color = GREEN if prob >= 0.55 else (ORANGE if prob >= 0.45 else RED)
        ax.text(0, 0.55, f'{prob:.1%}', ha='center', va='center',
                fontsize=36, fontweight='bold', color=color, fontfamily='monospace')
        ax.text(0, 0.25, 'Win Probability', ha='center', fontsize=11, color=MUTED)
        ax.text(0, -0.15, f'Seed {seed} · {distance:.0f} mi · {tournament_round}',
                ha='center', fontsize=9, color=MUTED)

        # Panel 2: Win curve
        ax = axes[1]
        dist_range = np.linspace(0, 2500, 200)
        probs_range = [lr2.predict_proba(sc2.transform([[d, seed]]))[0][1] for d in dist_range]
        ax.fill_between(dist_range, [p * 100 for p in probs_range], 50,
                        where=[p * 100 >= 50 for p in probs_range],
                        alpha=0.12, color=GREEN, interpolate=True)
        ax.fill_between(dist_range, [p * 100 for p in probs_range], 50,
                        where=[p * 100 < 50 for p in probs_range],
                        alpha=0.12, color=RED, interpolate=True)
        ax.plot(dist_range, [p * 100 for p in probs_range], color=ORANGE, linewidth=2.5)
        ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(distance, color=CYAN, linestyle='-', linewidth=2, alpha=0.8)
        ax.plot(distance, prob * 100, 'o', color=CYAN, markersize=10, zorder=5)
        ax.annotate(f'  {prob:.1%}', (distance, prob * 100), fontsize=11,
                    fontweight='bold', color=CYAN)
        ax.set_xlabel('Distance (miles)', fontsize=10)
        ax.set_ylabel('Win Probability (%)', fontsize=10)
        ax.set_title(f'Seed {seed} Win Curve', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(0, 2500); ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        # Panel 3: Historical
        ax = axes[2]
        if len(hist) > 5:
            bins = np.linspace(0, max(hist['DISTANCE (MI)'].max(), distance + 100), 20)
            w_hist = hist[hist['WON'] == 1]['DISTANCE (MI)']
            l_hist = hist[hist['WON'] == 0]['DISTANCE (MI)']
            ax.hist(w_hist, bins=bins, alpha=0.6, color=GREEN, label=f'Wins (n={len(w_hist)})', density=True)
            ax.hist(l_hist, bins=bins, alpha=0.5, color=RED, label=f'Losses (n={len(l_hist)})', density=True)
            ax.axvline(distance, color=CYAN, linewidth=2, linestyle='-', label='Your distance')
            ax.set_xlabel('Distance (miles)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8, framealpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Insufficient data\nfor seed {seed} in {tournament_round}\n(n={len(hist)})',
                    ha='center', va='center', transform=ax.transAxes, color=MUTED, fontsize=11)
        ax.set_title(f'Historical: Seed {seed} in {tournament_round}', fontsize=13, fontweight='bold', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



    with tab_enhanced:
        st.subheader("Enhanced Game Predictor")
        st.caption("Uses TeamRankings power ratings, KenPom efficiency, strength of schedule, "
                   "and more to predict matchup outcomes. Pick two real tournament teams or enter custom stats.")

        pred_mode = st.radio("Mode", ["Pick Tournament Teams", "Custom Input"], horizontal=True, key="enh_mode")

        if pred_mode == "Pick Tournament Teams":
            # Let user pick year and two teams
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                enh_year = st.selectbox("Tournament Year",
                                         sorted(enhanced['YEAR'].unique(), reverse=True), key="enh_year")
            year_teams = sorted(enhanced[enhanced['YEAR'] == enh_year]['TEAM'].unique())
            with c2:
                team_a = st.selectbox("Team A", year_teams,
                                       index=0, key="enh_team_a")
            with c3:
                team_b = st.selectbox("Team B", year_teams,
                                       index=min(1, len(year_teams) - 1), key="enh_team_b")

            if team_a == team_b:
                st.warning("Please select two different teams.")
            else:
                # Get the most advanced round for each team that year (use their entry stats)
                row_a = enhanced[(enhanced['YEAR'] == enh_year) & (enhanced['TEAM'] == team_a)].iloc[0]
                row_b = enhanced[(enhanced['YEAR'] == enh_year) & (enhanced['TEAM'] == team_b)].iloc[0]

                # Show team profiles side by side
                st.markdown("#### Team Profiles")
                col_a, col_b = st.columns(2)

                def show_team_profile(col, row, label):
                    with col:
                        st.markdown(f"**{label}: {row['TEAM']}**")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Seed", f"#{int(row['SEED'])}")
                        p2.metric("TR Rating", f"{row.get('TR RATING', 'N/A'):.1f}" if pd.notna(row.get('TR RATING')) else "N/A")
                        p3.metric("TR Rank", f"#{int(row.get('TR RANK', 0))}" if pd.notna(row.get('TR RANK')) else "N/A")
                        p4, p5, p6 = st.columns(3)
                        p4.metric("SOS Rating", f"{row.get('SOS RATING', 'N/A'):.1f}" if pd.notna(row.get('SOS RATING')) else "N/A")
                        if pd.notna(row.get('PRESEASON KADJ EM')):
                            p5.metric("KenPom EM", f"{row['PRESEASON KADJ EM']:.1f}")
                            p6.metric("EM Improvement", f"{row.get('KADJ EM CHANGE', 0):+.1f}" if pd.notna(row.get('KADJ EM CHANGE')) else "N/A")
                        else:
                            p5.metric("KenPom EM", "N/A")
                            p6.metric("EM Improvement", "N/A")

                show_team_profile(col_a, row_a, "Team A")
                show_team_profile(col_b, row_b, "Team B")

                # ── Predictions from all models ──────────────────────────────
                st.markdown("#### Model Predictions")

                # Simple model (distance + seed)
                prob_a_simple = lr2.predict_proba(sc2.transform([[row_a['DISTANCE (MI)'], row_a['SEED']]]))[0][1]
                prob_b_simple = lr2.predict_proba(sc2.transform([[row_b['DISTANCE (MI)'], row_b['SEED']]]))[0][1]
                total_simple = prob_a_simple + prob_b_simple
                rel_a_simple = prob_a_simple / total_simple
                rel_b_simple = prob_b_simple / total_simple

                # Enhanced model (TR features)
                try:
                    vals_a = [row_a[f] for f in features_tr]
                    vals_b = [row_b[f] for f in features_tr]
                    if all(pd.notna(v) for v in vals_a) and all(pd.notna(v) for v in vals_b):
                        prob_a_enh = lr4.predict_proba(sc4.transform([vals_a]))[0][1]
                        prob_b_enh = lr4.predict_proba(sc4.transform([vals_b]))[0][1]
                        total_enh = prob_a_enh + prob_b_enh
                        rel_a_enh = prob_a_enh / total_enh
                        rel_b_enh = prob_b_enh / total_enh
                        has_enhanced = True
                    else:
                        has_enhanced = False
                except Exception:
                    has_enhanced = False

                # Full model (TR + KenPom)
                try:
                    vals_a_full = [row_a[f] for f in features_full]
                    vals_b_full = [row_b[f] for f in features_full]
                    if all(pd.notna(v) for v in vals_a_full) and all(pd.notna(v) for v in vals_b_full):
                        prob_a_full = lr5.predict_proba(sc5.transform([vals_a_full]))[0][1]
                        prob_b_full = lr5.predict_proba(sc5.transform([vals_b_full]))[0][1]
                        total_full = prob_a_full + prob_b_full
                        rel_a_full = prob_a_full / total_full
                        rel_b_full = prob_b_full / total_full
                        has_full = True
                    else:
                        has_full = False
                except Exception:
                    has_full = False

                # Display predictions
                dark_style()
                n_panels = 1 + int(has_enhanced) + int(has_full)
                fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
                if n_panels == 1:
                    axes = [axes]

                def draw_h2h_bar(ax, rel_a, rel_b, title, team_a_name, team_b_name):
                    ax.barh([1], [rel_a * 100], height=0.5, color=GREEN, alpha=0.85, zorder=3)
                    ax.barh([0], [rel_b * 100], height=0.5, color=RED, alpha=0.85, zorder=3)
                    ax.axvline(50, color=MUTED, linestyle='--', linewidth=0.8)
                    ax.text(rel_a * 100 + 1, 1, f'{rel_a:.1%}', va='center', fontsize=14, fontweight='bold', color=GREEN)
                    ax.text(rel_b * 100 + 1, 0, f'{rel_b:.1%}', va='center', fontsize=14, fontweight='bold', color=RED)
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels([team_b_name, team_a_name], fontsize=10)
                    ax.set_xlabel('Win Probability (%)', fontsize=10)
                    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
                    ax.set_xlim(0, 105)
                    ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

                panel_idx = 0
                draw_h2h_bar(axes[panel_idx], rel_a_simple, rel_b_simple,
                             f'Simple Model\n(AUC={cv2.mean():.3f})', team_a, team_b)
                panel_idx += 1

                if has_enhanced:
                    draw_h2h_bar(axes[panel_idx], rel_a_enh, rel_b_enh,
                                 f'Enhanced Model\n(AUC={cv4.mean():.3f})', team_a, team_b)
                    panel_idx += 1

                if has_full:
                    draw_h2h_bar(axes[panel_idx], rel_a_full, rel_b_full,
                                 f'Full Model\n(AUC={cv5.mean():.3f})', team_a, team_b)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Summary table
                st.markdown("#### Prediction Summary")
                pred_rows = [{'Model': 'Simple (Dist+Seed)',
                              'AUC': f'{cv2.mean():.3f}',
                              f'{team_a} Win%': f'{rel_a_simple:.1%}',
                              f'{team_b} Win%': f'{rel_b_simple:.1%}',
                              'Predicted Winner': team_a if rel_a_simple > rel_b_simple else team_b,
                              'Edge': f'{abs(rel_a_simple - rel_b_simple):.1%}'}]
                if has_enhanced:
                    pred_rows.append({'Model': 'Enhanced (TR Ratings)',
                                      'AUC': f'{cv4.mean():.3f}',
                                      f'{team_a} Win%': f'{rel_a_enh:.1%}',
                                      f'{team_b} Win%': f'{rel_b_enh:.1%}',
                                      'Predicted Winner': team_a if rel_a_enh > rel_b_enh else team_b,
                                      'Edge': f'{abs(rel_a_enh - rel_b_enh):.1%}'})
                if has_full:
                    pred_rows.append({'Model': 'Full (TR+KenPom)',
                                      'AUC': f'{cv5.mean():.3f}',
                                      f'{team_a} Win%': f'{rel_a_full:.1%}',
                                      f'{team_b} Win%': f'{rel_b_full:.1%}',
                                      'Predicted Winner': team_a if rel_a_full > rel_b_full else team_b,
                                      'Edge': f'{abs(rel_a_full - rel_b_full):.1%}'})
                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

                # Check if models agree
                winners = [r['Predicted Winner'] for r in pred_rows]
                if len(set(winners)) == 1:
                    st.success(f"All models agree: **{winners[0]}** is the predicted winner.")
                else:
                    st.warning("Models disagree on the winner — this could be a close game or potential upset!")

        else:
            # Custom input mode
            st.markdown("#### Enter Custom Team Stats")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Team A**")
                c_seed_a = st.slider("Seed", 1, 16, 3, key="enh_c_sa")
                c_dist_a = st.slider("Distance (miles)", 0, 2500, 200, step=25, key="enh_c_da")
                c_tr_a = st.number_input("TR Rating", value=15.0, step=0.5, key="enh_c_tra",
                                          help="TeamRankings power rating (top teams ~20+, avg ~0)")
                c_sos_a = st.number_input("SOS Rating", value=8.0, step=0.5, key="enh_c_sosa",
                                           help="Strength of schedule (higher = tougher)")
                c_kem_a = st.number_input("KenPom Adj EM", value=20.0, step=0.5, key="enh_c_kema",
                                           help="KenPom preseason adjusted efficiency margin")
                c_kchg_a = st.number_input("KenPom EM Change", value=2.0, step=0.5, key="enh_c_kchga",
                                            help="How much team improved vs preseason (+positive = improved)")
                c_v25_a = st.number_input("Wins vs Top 25", value=5, step=1, key="enh_c_v25a")
            with col_b:
                st.markdown("**Team B**")
                c_seed_b = st.slider("Seed", 1, 16, 11, key="enh_c_sb")
                c_dist_b = st.slider("Distance (miles)", 0, 2500, 800, step=25, key="enh_c_db")
                c_tr_b = st.number_input("TR Rating", value=6.0, step=0.5, key="enh_c_trb")
                c_sos_b = st.number_input("SOS Rating", value=2.0, step=0.5, key="enh_c_sosb")
                c_kem_b = st.number_input("KenPom Adj EM", value=10.0, step=0.5, key="enh_c_kemb")
                c_kchg_b = st.number_input("KenPom EM Change", value=5.0, step=0.5, key="enh_c_kchgb")
                c_v25_b = st.number_input("Wins vs Top 25", value=1, step=1, key="enh_c_v25b")

            # Run all three models
            prob_a_s = lr2.predict_proba(sc2.transform([[c_dist_a, c_seed_a]]))[0][1]
            prob_b_s = lr2.predict_proba(sc2.transform([[c_dist_b, c_seed_b]]))[0][1]
            rel_a_s = prob_a_s / (prob_a_s + prob_b_s)
            rel_b_s = 1 - rel_a_s

            prob_a_e = lr4.predict_proba(sc4.transform([[c_dist_a, c_seed_a, c_tr_a, c_sos_a]]))[0][1]
            prob_b_e = lr4.predict_proba(sc4.transform([[c_dist_b, c_seed_b, c_tr_b, c_sos_b]]))[0][1]
            rel_a_e = prob_a_e / (prob_a_e + prob_b_e)
            rel_b_e = 1 - rel_a_e

            prob_a_f = lr5.predict_proba(sc5.transform([[c_dist_a, c_seed_a, c_tr_a, c_sos_a,
                                                          c_kem_a, c_kchg_a, c_v25_a]]))[0][1]
            prob_b_f = lr5.predict_proba(sc5.transform([[c_dist_b, c_seed_b, c_tr_b, c_sos_b,
                                                          c_kem_b, c_kchg_b, c_v25_b]]))[0][1]
            rel_a_f = prob_a_f / (prob_a_f + prob_b_f)
            rel_b_f = 1 - rel_a_f

            st.markdown("#### Predictions Across Models")
            dark_style()
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))

            for ax, (ra, rb, title, auc) in zip(axes, [
                (rel_a_s, rel_b_s, 'Simple\n(Dist+Seed)', cv2.mean()),
                (rel_a_e, rel_b_e, 'Enhanced\n(+TR Ratings)', cv4.mean()),
                (rel_a_f, rel_b_f, 'Full Model\n(+KenPom)', cv5.mean()),
            ]):
                ax.barh([1], [ra * 100], height=0.5, color=GREEN, alpha=0.85, zorder=3)
                ax.barh([0], [rb * 100], height=0.5, color=RED, alpha=0.85, zorder=3)
                ax.axvline(50, color=MUTED, linestyle='--', linewidth=0.8)
                ax.text(ra * 100 + 1, 1, f'{ra:.1%}', va='center', fontsize=14, fontweight='bold', color=GREEN)
                ax.text(rb * 100 + 1, 0, f'{rb:.1%}', va='center', fontsize=14, fontweight='bold', color=RED)
                ax.set_yticks([0, 1])
                ax.set_yticklabels([f'Team B (#{c_seed_b})', f'Team A (#{c_seed_a})'], fontsize=10)
                ax.set_xlabel('Win Probability (%)', fontsize=10)
                ax.set_title(f'{title}\n(AUC={auc:.3f})', fontsize=13, fontweight='bold', pad=10)
                ax.set_xlim(0, 105)
                ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Feature importance for the full model
            st.markdown("#### What Matters Most?")
            st.caption("Standardized coefficients from the full model — larger magnitude = more predictive power.")

            dark_style()
            fig, ax = plt.subplots(figsize=(10, 4))
            feat_labels = ['Distance', 'Seed', 'TR Rating', 'SOS Rating',
                           'KenPom EM', 'EM Change', 'Top-25 Wins']
            coefs = list(lr5.coef_[0])
            fc = [RED if c < 0 else GREEN for c in coefs]
            sorted_idx = np.argsort(np.abs(coefs))
            ax.barh([feat_labels[i] for i in sorted_idx], [coefs[i] for i in sorted_idx],
                    color=[fc[i] for i in sorted_idx], alpha=0.85, height=0.5, zorder=3)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            for i, idx in enumerate(sorted_idx):
                c = coefs[idx]
                ax.text(c + (0.02 if c >= 0 else -0.02), i, f'{c:+.3f}',
                        ha='left' if c >= 0 else 'right', va='center', fontsize=10, color=LIGHT)
            ax.set_xlabel('Standardized Coefficient', fontsize=10)
            ax.set_title('Feature Importance (Full Model)', fontsize=14, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.info("**How to read this:** Positive coefficients (green) increase win probability. "
                    "Negative coefficients (red) decrease it. KenPom efficiency margin and improvement "
                    "from preseason are typically the strongest predictors.")



    with tab_h2h:
        st.subheader("Head-to-Head Simulator")
        st.caption("Simulate a matchup between two teams with custom seeds and distances.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Team A**")
            seed_a = st.slider("Seed A", 1, 16, 3, step=1, key="h2h_sa")
            dist_a = st.slider("Distance A (miles)", 0, 2500, 200, step=25, key="h2h_da")
        with col_b:
            st.markdown("**Team B**")
            seed_b = st.slider("Seed B", 1, 16, 11, step=1, key="h2h_sb")
            dist_b = st.slider("Distance B (miles)", 0, 2500, 1200, step=25, key="h2h_db")

        prob_a = lr2.predict_proba(sc2.transform([[dist_a, seed_a]]))[0][1]
        prob_b = lr2.predict_proba(sc2.transform([[dist_b, seed_b]]))[0][1]
        total = prob_a + prob_b
        rel_a = prob_a / total
        rel_b = prob_b / total

        winner = 'Team A' if rel_a > rel_b else 'Team B'
        edge = abs(rel_a - rel_b)

        m1, m2, m3 = st.columns(3)
        m1.metric("Team A Win Prob", f"{rel_a:.1%}")
        m2.metric("Team B Win Prob", f"{rel_b:.1%}")
        m3.metric("Predicted Winner", f"{winner} (+{edge:.1%})")

        dark_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: H2H bar
        ax = axes[0]
        ax.barh([1], [rel_a * 100], height=0.5, color=GREEN, alpha=0.85, label=f'Team A (#{seed_a})', zorder=3)
        ax.barh([0], [rel_b * 100], height=0.5, color=RED, alpha=0.85, label=f'Team B (#{seed_b})', zorder=3)
        ax.axvline(50, color=MUTED, linestyle='--', linewidth=0.8)
        ax.text(rel_a * 100 + 1, 1, f'{rel_a:.1%}', va='center', fontsize=16, fontweight='bold', color=GREEN)
        ax.text(rel_b * 100 + 1, 0, f'{rel_b:.1%}', va='center', fontsize=16, fontweight='bold', color=RED)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f'Team B\n#{seed_b}, {dist_b:.0f} mi', f'Team A\n#{seed_a}, {dist_a:.0f} mi'], fontsize=10)
        ax.set_xlabel('Relative Win Probability (%)', fontsize=10)
        ax.set_title('Head-to-Head Prediction', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        # Panel 2: Sensitivity
        ax = axes[1]
        d_range = np.linspace(0, 2500, 100)
        probs_a_vary = [lr2.predict_proba(sc2.transform([[d, seed_a]]))[0][1] for d in d_range]
        rel_a_vary = [pa / (pa + prob_b) for pa in probs_a_vary]
        ax.plot(d_range, [r * 100 for r in rel_a_vary], color=GREEN, linewidth=2.5, label='Team A edge')
        ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8)
        ax.axvline(dist_a, color=CYAN, linewidth=2, alpha=0.8, label=f'Current ({dist_a:.0f} mi)')
        ax.fill_between(d_range, [r * 100 for r in rel_a_vary], 50,
                        where=[r * 100 >= 50 for r in rel_a_vary],
                        alpha=0.1, color=GREEN, interpolate=True)
        ax.fill_between(d_range, [r * 100 for r in rel_a_vary], 50,
                        where=[r * 100 < 50 for r in rel_a_vary],
                        alpha=0.1, color=RED, interpolate=True)
        ax.set_xlabel('Team A Distance (miles)', fontsize=10)
        ax.set_ylabel('Team A Relative Win Prob (%)', fontsize=10)
        ax.set_title('Sensitivity: Team A Distance', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=9, framealpha=0.3)
        ax.set_xlim(0, 2500)
        ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



    with tab_team:
        st.subheader("Team Lookup")
        st.caption("Look up any team's tournament travel history and outcomes.")

        team_name = st.selectbox("Select Team", all_teams,
                                  index=all_teams.index('Duke') if 'Duke' in all_teams else 0,
                                  key="team_select")

        df = team_data[team_data['TEAM'] == team_name]

        if len(df) == 0:
            st.warning(f"No tournament data for {team_name}.")
        else:
            med = df['DISTANCE (MI)'].median()
            close = df[df['DISTANCE (MI)'] <= med]
            far = df[df['DISTANCE (MI)'] > med]
            years = sorted(df['YEAR'].unique())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Appearances", f"{len(years)}")
            m2.metric("Total Games", f"{len(df)}")
            m3.metric("Win Rate", f"{df['WON'].mean():.1%}")
            m4.metric("Avg Distance", f"{df['DISTANCE (MI)'].mean():.0f} mi")

            dark_style()
            fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

            # Panel 1: Distance vs Outcome
            ax = axes[0]
            wins = df[df['WON'] == 1]
            losses = df[df['WON'] == 0]
            ax.scatter(wins['DISTANCE (MI)'], wins['YEAR'], color=GREEN, s=60,
                       alpha=0.8, label=f'Wins (n={len(wins)})', zorder=3, edgecolors='white', linewidths=0.3)
            ax.scatter(losses['DISTANCE (MI)'], losses['YEAR'], color=RED, s=60, marker='X',
                       alpha=0.8, label=f'Losses (n={len(losses)})', zorder=3, edgecolors='white', linewidths=0.3)
            ax.set_xlabel('Distance Traveled (miles)', fontsize=10)
            ax.set_ylabel('Year', fontsize=10)
            ax.set_title(f'{team_name} Games\nby Distance & Outcome', fontsize=14, fontweight='bold', pad=10)
            ax.legend(fontsize=9, framealpha=0.3)
            ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

            # Panel 2: Close vs Far
            ax = axes[1]
            vals = [close['WON'].mean() * 100 if len(close) > 0 else 0,
                    far['WON'].mean() * 100 if len(far) > 0 else 0]
            bars = ax.bar([f'Close\n(<{med:.0f} mi)', f'Far\n(>{med:.0f} mi)'],
                          vals, color=[GREEN, RED], alpha=0.85, width=0.5, zorder=3)
            ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f'{v:.1f}%',
                        ha='center', color=LIGHT, fontsize=12, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=10)
            ax.set_title(f'{team_name}\nClose vs Far', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim(0, max(vals) + 15)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            # Panel 3: Distance histogram
            ax = axes[2]
            ax.hist(df['DISTANCE (MI)'], bins=15, color=BLUE, alpha=0.7, edgecolor=BG, zorder=3)
            ax.axvline(df['DISTANCE (MI)'].mean(), color=ORANGE, linewidth=2, linestyle='--',
                       label=f'Mean: {df["DISTANCE (MI)"].mean():.0f} mi')
            ax.set_xlabel('Distance (miles)', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{team_name}\nTravel Distribution', fontsize=14, fontweight='bold', pad=10)
            ax.legend(fontsize=9, framealpha=0.3)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(f"Tournament years: {', '.join(str(y) for y in years)}")



# ─── MASTER TAB 2: TRAVEL & DISTANCE ANALYSIS ────────────────────────────

with master_travel:
    tab_exp, tab_rnd, tab_seed = st.tabs([
        "📊 Distance Explorer",
        "🔄 By Round",
        "🌱 By Seed",
    ])

    with tab_exp:
        st.subheader("Distance vs Win Rate Explorer")
        st.caption("Filter by round, seed range, and years to explore the distance–win relationship.")

        c1, c2, c3 = st.columns(3)
        with c1:
            round_filter = st.selectbox("Round",
                ['All Rounds', 'Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship'],
                key="exp_round")
        with c2:
            seed_range = st.slider("Seed Range", 1, 16, (1, 16), step=1, key="exp_seeds")
        with c3:
            year_range_exp = st.slider("Year Range", int(min(all_years)), int(max(all_years)),
                                       (int(min(all_years)), int(max(all_years))), step=1, key="exp_years")

        rnd_map = {'All Rounds': None, 'Round of 64': 64, 'Round of 32': 32,
                   'Sweet 16': 16, 'Elite 8': 8, 'Final 4': 4, 'Championship': 2}
        rnd_val = rnd_map[round_filter]

        df = team_data.copy()
        if rnd_val is not None:
            df = df[df['CURRENT ROUND'] == rnd_val]
        df = df[(df['SEED'] >= seed_range[0]) & (df['SEED'] <= seed_range[1])]
        df = df[(df['YEAR'] >= year_range_exp[0]) & (df['YEAR'] <= year_range_exp[1])]

        if len(df) < 20:
            st.warning(f"Not enough data for these filters (n={len(df)}). Try broadening your filters.")
        else:
            # Stats
            corr, pval = stats.pointbiserialr(df['WON'], df['DISTANCE (MI)'])
            w_mean = df[df['WON'] == 1]['DISTANCE (MI)'].mean()
            l_mean = df[df['WON'] == 0]['DISTANCE (MI)'].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Observations", f"{len(df):,}")
            m2.metric("Correlation (r)", f"{corr:+.4f}")
            m3.metric("p-value", f"{pval:.6f}")
            m4.metric("Losers Travel More", f"{l_mean - w_mean:+.0f} mi")

            # Bucket
            buckets = [0, 100, 250, 500, 750, 1000, 1500, 3500]
            labels = ['0–100', '100–250', '250–500', '500–750', '750–1K', '1K–1.5K', '1.5K+']
            df_plot = df.copy()
            df_plot['BUCKET'] = pd.cut(df_plot['DISTANCE (MI)'], bins=buckets, labels=labels)
            bstats = df_plot.groupby('BUCKET', observed=True).agg(
                wr=('WON', 'mean'), n=('WON', 'count'), avg_seed=('SEED', 'mean'),
                avg_dist=('DISTANCE (MI)', 'mean')
            ).reset_index()
            bstats = bstats[bstats['n'] >= 5]

            dark_style()
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

            # LEFT: Bar chart
            ax = axes[0]
            colors = [GREEN if w >= 0.5 else RED for w in bstats['wr']]
            bars = ax.bar(range(len(bstats)), bstats['wr'] * 100, color=colors, alpha=0.85,
                          width=0.6, zorder=3)
            ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5)
            for i, (bar, row) in enumerate(zip(bars, bstats.itertuples())):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f'{row.wr:.1%}', ha='center', color=LIGHT, fontsize=10, fontweight='bold')
                ax.text(bar.get_x() + bar.get_width() / 2, -4,
                        f'n={row.n}', ha='center', color=MUTED, fontsize=8)
            ax.set_xticks(range(len(bstats)))
            ax.set_xticklabels(bstats['BUCKET'], fontsize=9)
            ax.set_xlabel('Distance Traveled (miles)', fontsize=10, labelpad=18)
            ax.set_ylabel('Win Rate (%)', fontsize=10)
            ax.set_title('Win Rate by Distance Bucket', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim(0, max(bstats['wr'].max() * 100 + 12, 60))
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            # RIGHT: Scatter with trend
            ax = axes[1]
            jitter_y = df['WON'] + np.random.normal(0, 0.03, len(df))
            c = [GREEN if w == 1 else RED for w in df['WON']]
            ax.scatter(df['DISTANCE (MI)'], jitter_y, c=c, alpha=0.15, s=12, zorder=2)
            if df['WON'].nunique() > 1:
                sc_tmp = StandardScaler()
                X_tmp = sc_tmp.fit_transform(df[['DISTANCE (MI)']].values)
                lr_tmp = LogisticRegression(random_state=42).fit(X_tmp, df['WON'].values)
                xr = np.linspace(df['DISTANCE (MI)'].min(), df['DISTANCE (MI)'].max(), 200)
                pr = lr_tmp.predict_proba(sc_tmp.transform(xr.reshape(-1, 1)))[:, 1]
                ax.plot(xr, pr, color=ORANGE, linewidth=2.5, zorder=4, label='Logistic fit')
                ax.legend(fontsize=9, framealpha=0.3)
            ax.set_xlabel('Distance (miles)', fontsize=10)
            ax.set_ylabel('Outcome (0=Loss, 1=Win)', fontsize=10)
            ax.set_title('Individual Outcomes with Trend', fontsize=14, fontweight='bold', pad=10)
            ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['Loss', '', 'Win'])
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)



    with tab_rnd:
        st.subheader("Round-by-Round Analysis")
        st.caption("See how the distance effect changes across tournament rounds.")

        year_range_rnd = st.slider("Year Range", int(min(all_years)), int(max(all_years)),
                                   (int(min(all_years)), int(max(all_years))), step=1, key="rnd_years")

        df = team_data[(team_data['YEAR'] >= year_range_rnd[0]) & (team_data['YEAR'] <= year_range_rnd[1])]

        rounds = [64, 32, 16, 8, 4, 2]
        r_labels = ['R64', 'R32', 'S16', 'E8', 'F4', 'Champ']
        results = []
        for rnd in rounds:
            rd = df[df['CURRENT ROUND'] == rnd]
            if len(rd) > 10:
                r, p = stats.pointbiserialr(rd['WON'], rd['DISTANCE (MI)'])
                med = rd['DISTANCE (MI)'].median()
                cw = rd[rd['DISTANCE (MI)'] <= med]['WON'].mean()
                fw = rd[rd['DISTANCE (MI)'] > med]['WON'].mean()
                results.append({'rnd': rnd, 'r': r, 'p': p, 'cw': cw, 'fw': fw, 'n': len(rd), 'sig': p < 0.05})

        rdf = pd.DataFrame(results)

        if len(rdf) == 0:
            st.warning("Not enough data for any round. Try widening the year range.")
        else:
            dark_style()
            fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

            # Panel 1: Close vs Far grouped bars
            ax = axes[0]
            x = np.arange(len(rdf))
            w = 0.32
            ax.bar(x - w / 2, rdf['cw'] * 100, w, color=GREEN, alpha=0.85, label='Closer', zorder=3)
            ax.bar(x + w / 2, rdf['fw'] * 100, w, color=RED, alpha=0.85, label='Farther', zorder=3)
            ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8)
            for i, row in rdf.iterrows():
                if row['sig']:
                    ax.text(i, max(row['cw'], row['fw']) * 100 + 2, '★', ha='center', color=ORANGE, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(r_labels[:len(rdf)], fontsize=10)
            ax.set_ylabel('Win Rate (%)', fontsize=10)
            ax.set_title('Close vs Far Win Rate', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim(30, 70)
            ax.legend(fontsize=9, framealpha=0.3)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            # Panel 2: Correlation strength
            ax = axes[1]
            bar_colors = [GREEN if r < 0 else RED for r in rdf['r']]
            for i, (_, row) in enumerate(rdf.iterrows()):
                alpha = 0.9 if row['sig'] else 0.35
                ax.bar(i, row['r'], color=bar_colors[i], alpha=alpha, width=0.55, zorder=3)
                ax.text(i, row['r'] + (0.015 if row['r'] >= 0 else -0.025),
                        f"{row['r']:+.3f}", ha='center', fontsize=8, color=LIGHT)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xticks(range(len(rdf)))
            ax.set_xticklabels(r_labels[:len(rdf)], fontsize=10)
            ax.set_ylabel('Correlation (r)', fontsize=10)
            ax.set_title('Correlation by Round\n(faded = not significant)', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylim(-0.3, 0.35)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            # Panel 3: Sample size
            ax = axes[2]
            ax.bar(range(len(rdf)), rdf['n'], color=BLUE, alpha=0.7, width=0.55, zorder=3)
            for i, row in rdf.iterrows():
                ax.text(i, row['n'] + 15, str(row['n']), ha='center', fontsize=9, color=LIGHT)
            ax.set_xticks(range(len(rdf)))
            ax.set_xticklabels(r_labels[:len(rdf)], fontsize=10)
            ax.set_ylabel('Observations', fontsize=10)
            ax.set_title('Sample Size by Round', fontsize=14, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Summary table
            st.markdown("#### Round-by-Round Summary")
            summary_rows = []
            for _, row in rdf.iterrows():
                name = round_map.get(int(row['rnd']), str(int(row['rnd'])))
                summary_rows.append({
                    'Round': name,
                    'Correlation (r)': f"{row['r']:+.4f}",
                    'p-value': f"{row['p']:.4f}",
                    'Significant': '✓' if row['sig'] else '✗',
                    'Close WR': f"{row['cw']:.1%}",
                    'Far WR': f"{row['fw']:.1%}",
                    'n': int(row['n']),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.caption("★ = statistically significant (p < 0.05)")



    with tab_seed:
        st.subheader("Seed Deep Dive")
        st.caption("Deep-dive into how distance affects specific seeds.")

        c1, c2 = st.columns([2, 1])
        with c1:
            selected_seeds = st.multiselect("Select Seeds",
                [f'Seed {i}' for i in range(1, 17)],
                default=[f'Seed {i}' for i in range(1, 9)],
                key="seed_select")
        with c2:
            year_range_seed = st.slider("Year Range", int(min(all_years)), int(max(all_years)),
                                        (int(min(all_years)), int(max(all_years))), step=1, key="seed_years")

        if not selected_seeds:
            st.info("Select at least one seed to analyze.")
        else:
            seed_nums = [int(s.replace('Seed ', '')) for s in selected_seeds]
            df = team_data[(team_data['YEAR'] >= year_range_seed[0]) & (team_data['YEAR'] <= year_range_seed[1])]
            df = df[df['SEED'].isin(seed_nums)]

            if len(df) < 20:
                st.warning("Not enough data for selected seeds/years.")
            else:
                results = []
                for s in sorted(seed_nums):
                    sd = df[df['SEED'] == s]
                    if len(sd) >= 10:
                        med = sd['DISTANCE (MI)'].median()
                        cw = sd[sd['DISTANCE (MI)'] <= med]['WON'].mean()
                        fw = sd[sd['DISTANCE (MI)'] > med]['WON'].mean()
                        results.append({
                            'seed': s, 'cw': cw, 'fw': fw, 'diff': cw - fw,
                            'n': len(sd), 'wr': sd['WON'].mean(),
                            'avg_close': sd[sd['DISTANCE (MI)'] <= med]['DISTANCE (MI)'].mean(),
                            'avg_far': sd[sd['DISTANCE (MI)'] > med]['DISTANCE (MI)'].mean()
                        })

                sdf = pd.DataFrame(results)

                if len(sdf) == 0:
                    st.warning("Not enough data for any of the selected seeds (min 10 obs each).")
                else:
                    dark_style()
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                    # LEFT: Grouped bars
                    ax = axes[0]
                    x = np.arange(len(sdf))
                    w = 0.35
                    ax.bar(x - w / 2, sdf['cw'] * 100, w, color=GREEN, alpha=0.85, label='Close (<median)', zorder=3)
                    ax.bar(x + w / 2, sdf['fw'] * 100, w, color=RED, alpha=0.85, label='Far (>median)', zorder=3)
                    ax.axhline(50, color=MUTED, linestyle='--', linewidth=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'#{s}' for s in sdf['seed']], fontsize=10)
                    ax.set_xlabel('Seed', fontsize=10)
                    ax.set_ylabel('Win Rate (%)', fontsize=10)
                    ax.set_title('Win Rate by Proximity\n(per Seed)', fontsize=14, fontweight='bold', pad=10)
                    ax.legend(fontsize=9, framealpha=0.3)
                    ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

                    # RIGHT: Difference
                    ax = axes[1]
                    diff_colors = [GREEN if d >= 0 else RED for d in sdf['diff']]
                    ax.bar(x, sdf['diff'] * 100, color=diff_colors, alpha=0.85, width=0.55, zorder=3)
                    ax.axhline(0, color=MUTED, linewidth=0.8)
                    for i, (_, row) in enumerate(sdf.iterrows()):
                        val = row['diff'] * 100
                        ax.text(i, val + (1.5 if val >= 0 else -3), f"{val:+.1f}pp",
                                ha='center', fontsize=9, color=LIGHT, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'#{s}' for s in sdf['seed']], fontsize=10)
                    ax.set_xlabel('Seed', fontsize=10)
                    ax.set_ylabel('Win Rate Δ (pp)', fontsize=10)
                    ax.set_title('Close − Far Advantage\n(positive = distance helps)', fontsize=14, fontweight='bold', pad=10)
                    ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Summary table
                    st.markdown("#### Seed-Level Distance Effects")
                    tbl = []
                    for _, row in sdf.iterrows():
                        tbl.append({
                            'Seed': f"#{int(row['seed'])}",
                            'Close WR': f"{row['cw']:.1%}",
                            'Far WR': f"{row['fw']:.1%}",
                            'Δ (pp)': f"{row['diff'] * 100:+.1f}",
                            'Avg Close': f"{row['avg_close']:.0f} mi",
                            'Avg Far': f"{row['avg_far']:.0f} mi",
                            'n': int(row['n']),
                        })
                    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
                    pos = (sdf['diff'] > 0).sum()
                    st.info(f"**{pos} of {len(sdf)}** selected seeds show a proximity advantage.")



# ─── MASTER TAB 3: CONFERENCE INTELLIGENCE ───────────────────────────────

with master_conf:
    st.subheader("Conference Strength Impact")
    st.caption("Compare conferences side by side and see how conference strength relates to tournament success.")

    # ── Year and conference selection ────────────────────────────────────
    c1, c2 = st.columns([1, 2])
    with c1:
        conf_year = st.selectbox("Season", sorted(conf_stats['YEAR'].unique(), reverse=True), key="conf_year")
    with c2:
        available_confs = sorted(conf_stats[conf_stats['YEAR'] == conf_year]['CONF'].unique())
        # Default to major conferences if available
        major_defaults = [c for c in ['SEC', 'B10', 'B12', 'ACC', 'BE', 'MWC'] if c in available_confs]
        if not major_defaults:
            major_defaults = available_confs[:4]
        selected_confs = st.multiselect("Select Conferences to Compare",
                                         available_confs, default=major_defaults, key="conf_select")

    if not selected_confs:
        st.info("Select at least one conference to analyze.")
    else:
        cs_yr = conf_stats[(conf_stats['YEAR'] == conf_year) & (conf_stats['CONF'].isin(selected_confs))]
        cs_yr = cs_yr.sort_values('BADJ EM', ascending=False)

        # ── Metrics overview ─────────────────────────────────────────────
        st.markdown("#### Conference Overview")
        overview_df = cs_yr[['CONF', 'BADJ EM', 'BADJ O', 'BADJ D', 'BARTHAG',
                             'TALENT', 'ELITE SOS', 'WAB', 'EXP', 'AVG HGT']].copy()
        overview_df.columns = ['Conference', 'Adj EM', 'Adj Off', 'Adj Def', 'BARTHAG',
                               'Talent', 'Elite SOS', 'WAB', 'Experience', 'Avg Height']
        for col in ['Adj EM', 'Adj Off', 'Adj Def', 'BARTHAG', 'Talent', 'Elite SOS', 'WAB']:
            overview_df[col] = overview_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        overview_df['Experience'] = overview_df['Experience'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        overview_df['Avg Height'] = overview_df['Avg Height'].apply(lambda x: f'{x:.1f}"' if pd.notna(x) else "—")
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

        # ── Visualization: Multi-metric radar-style comparison ───────────
        dark_style()
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

        # Panel 1: Adjusted Efficiency Margin comparison
        ax = axes[0]
        confs = cs_yr['CONF'].values
        x = np.arange(len(confs))
        em_vals = cs_yr['BADJ EM'].values
        em_colors = [GREEN if v >= 10 else (ORANGE if v >= 5 else (BLUE if v >= 0 else RED)) for v in em_vals]
        bars = ax.bar(x, em_vals, color=em_colors, alpha=0.85, width=0.6, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for i, (bar, v) in enumerate(zip(bars, em_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.5 if v >= 0 else -1.5),
                    f'{v:.1f}', ha='center', color=LIGHT, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(confs, fontsize=10)
        ax.set_ylabel('Adjusted Efficiency Margin', fontsize=10)
        ax.set_title('Conference Strength\n(Adj. Efficiency Margin)', fontsize=14, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        # Panel 2: Offense vs Defense scatter
        ax = axes[1]
        for _, row in cs_yr.iterrows():
            ax.scatter(row['BADJ O'], row['BADJ D'], s=120, zorder=3,
                       edgecolors='white', linewidths=0.5, alpha=0.9)
            ax.annotate(row['CONF'], (row['BADJ O'], row['BADJ D']),
                        textcoords="offset points", xytext=(8, -3),
                        fontsize=10, fontweight='bold', color=LIGHT)
        # Add quadrant lines at medians
        all_yr = conf_stats[conf_stats['YEAR'] == conf_year]
        ax.axhline(all_yr['BADJ D'].median(), color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(all_yr['BADJ O'].median(), color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Adjusted Offense', fontsize=10)
        ax.set_ylabel('Adjusted Defense (lower = better)', fontsize=10)
        ax.set_title('Offense vs Defense', fontsize=14, fontweight='bold', pad=10)
        ax.invert_yaxis()
        ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)
        # Label quadrants
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.text(xlim[1], ylim[0], ' Elite', fontsize=8, color=GREEN, alpha=0.6, va='top')
        ax.text(xlim[0], ylim[1], ' Weak', fontsize=8, color=RED, alpha=0.6, va='bottom')

        # Panel 3: Talent vs WAB (Wins Above Bubble)
        ax = axes[2]
        for _, row in cs_yr.iterrows():
            color = GREEN if row['WAB'] >= 0 else RED
            ax.scatter(row['TALENT'], row['WAB'], s=120, color=color, zorder=3,
                       edgecolors='white', linewidths=0.5, alpha=0.9)
            ax.annotate(row['CONF'], (row['TALENT'], row['WAB']),
                        textcoords="offset points", xytext=(8, -3),
                        fontsize=10, fontweight='bold', color=LIGHT)
        ax.axhline(0, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Talent Rating', fontsize=10)
        ax.set_ylabel('Wins Above Bubble (WAB)', fontsize=10)
        ax.set_title('Talent vs Tournament Readiness', fontsize=14, fontweight='bold', pad=10)
        ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Four Factors Comparison ──────────────────────────────────────
        st.markdown("#### Four Factors Comparison")
        st.caption("The 'four factors' of basketball success: shooting, turnovers, rebounding, and free throws.")

        dark_style()
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        factor_pairs = [
            ('EFG%', 'EFGD%', 'Effective FG%', 'Off EFG%', 'Def EFG%'),
            ('TOV%', 'TOV%D', 'Turnover Rate', 'Own TOV%', 'Forced TOV%'),
            ('OREB%', 'DREB%', 'Rebounding', 'Off Reb%', 'Def Reb%'),
            ('FTR', 'FTRD', 'Free Throw Rate', 'Own FTR', 'Opp FTR'),
        ]
        for ax, (off_col, def_col, title, off_label, def_label) in zip(axes, factor_pairs):
            x = np.arange(len(cs_yr))
            w = 0.32
            ax.bar(x - w / 2, cs_yr[off_col].values, w, color=GREEN, alpha=0.85, label=off_label, zorder=3)
            ax.bar(x + w / 2, cs_yr[def_col].values, w, color=RED, alpha=0.85, label=def_label, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(cs_yr['CONF'].values, fontsize=8)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.legend(fontsize=7, framealpha=0.3)
            ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Historical Conference Strength Trends ────────────────────────
        st.markdown("#### Historical Strength Trends")
        st.caption("Track how conference strength has changed over the years.")

        cs_hist = conf_stats[conf_stats['CONF'].isin(selected_confs)].sort_values('YEAR')

        dark_style()
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # Adj EM over time
        ax = axes[0]
        conf_colors = plt.cm.Set2(np.linspace(0, 1, len(selected_confs)))
        for i, conf in enumerate(selected_confs):
            cd = cs_hist[cs_hist['CONF'] == conf]
            ax.plot(cd['YEAR'], cd['BADJ EM'], marker='o', markersize=4, linewidth=2,
                    label=conf, color=conf_colors[i], zorder=3)
        ax.set_xlabel('Year', fontsize=10); ax.set_ylabel('Adj. Efficiency Margin', fontsize=10)
        ax.set_title('Conference Strength Over Time', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        # Talent over time
        ax = axes[1]
        for i, conf in enumerate(selected_confs):
            cd = cs_hist[cs_hist['CONF'] == conf]
            ax.plot(cd['YEAR'], cd['TALENT'], marker='s', markersize=4, linewidth=2,
                    label=conf, color=conf_colors[i], zorder=3)
        ax.set_xlabel('Year', fontsize=10); ax.set_ylabel('Talent Rating', fontsize=10)
        ax.set_title('Talent Trends', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        # WAB over time
        ax = axes[2]
        for i, conf in enumerate(selected_confs):
            cd = cs_hist[cs_hist['CONF'] == conf]
            ax.plot(cd['YEAR'], cd['WAB'], marker='^', markersize=4, linewidth=2,
                    label=conf, color=conf_colors[i], zorder=3)
        ax.axhline(0, color=MUTED, linestyle='--', linewidth=0.8)
        ax.set_xlabel('Year', fontsize=10); ax.set_ylabel('Wins Above Bubble', fontsize=10)
        ax.set_title('WAB Trends', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Shooting Profile ─────────────────────────────────────────────
        st.markdown("#### Shooting Profile")
        shooting_df = cs_yr[['CONF', '2PT%', '3PT%', '2PT%D', '3PT%D', 'FT%', '2PTR', '3PTR']].copy()
        shooting_df.columns = ['Conference', '2PT%', '3PT%', '2PT% Def', '3PT% Def', 'FT%',
                                '2PT Rate', '3PT Rate']
        for col in shooting_df.columns[1:]:
            shooting_df[col] = shooting_df[col].apply(lambda x: f"{x:.1f}%")
        st.dataframe(shooting_df, use_container_width=True, hide_index=True)



# ─── MASTER TAB 4: MODELS & METHODOLOGY ──────────────────────────────────

with master_models:
    st.subheader("Model Comparison")
    st.caption("Compare all five logistic regression models and their feature importance.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Distance Only", f"{cv1.mean():.3f}")
    m2.metric("Dist + Seed", f"{cv2.mean():.3f}")
    m3.metric("Dist+Seed+TZ", f"{cv3.mean():.3f}")
    m4.metric("Enhanced (TR)", f"{cv4.mean():.3f}")
    m5.metric("Full (TR+KP)", f"{cv5.mean():.3f}")

    dark_style()

    # ── Row 1: ROC Curves and AUC Bars ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: ROC Curves for all 5 models
    ax = axes[0]
    PURPLE = '#a855f7'
    PINK = '#ec4899'
    for Xs, model, y_true, name, color, ls in [
        (X1s, lr1, y_all, f'Distance (AUC={cv1.mean():.3f})', MUTED, ':'),
        (X2s, lr2, y_all, f'Dist+Seed (AUC={cv2.mean():.3f})', ORANGE, '-'),
        (X3s, lr3, y_all, f'Dist+Seed+TZ (AUC={cv3.mean():.3f})', CYAN, '--'),
        (X4s, lr4, y4, f'Enhanced (AUC={cv4.mean():.3f})', GREEN, '-'),
        (X5s, lr5, y5, f'Full (AUC={cv5.mean():.3f})', PURPLE, '-'),
    ]:
        yp = model.predict_proba(Xs)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, yp)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls, label=name, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=':', linewidth=1, alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.3)
    ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: AUC bars
    ax = axes[1]
    names = ['Distance\nOnly', 'Dist\n+Seed', 'Dist+Seed\n+TZ', 'Enhanced\n(TR)', 'Full\n(TR+KP)']
    aucs = [cv1.mean(), cv2.mean(), cv3.mean(), cv4.mean(), cv5.mean()]
    stds = [cv1.std(), cv2.std(), cv3.std(), cv4.std(), cv5.std()]
    bc = [MUTED, ORANGE, CYAN, GREEN, PURPLE]
    bars = ax.bar(range(5), aucs, yerr=stds, capsize=5, color=bc, alpha=0.85,
                  width=0.55, error_kw={'color': MUTED, 'linewidth': 1.5}, zorder=3)
    for i, (b, a) in enumerate(zip(bars, aucs)):
        ax.text(b.get_x() + b.get_width() / 2, a + stds[i] + 0.008,
                f'{a:.3f}', ha='center', fontsize=10, fontweight='bold', color=LIGHT)
    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('AUC-ROC', fontsize=10)
    ax.set_title('Model AUC (5-fold CV)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0.45, max(aucs) + 0.08)
    ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Row 2: Feature coefficients for Enhanced and Full models ─────────
    st.markdown("#### Feature Importance")
    st.caption("Standardized coefficients — larger magnitude means more predictive power. "
               "Green = increases win probability, Red = decreases it.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Enhanced model coefficients
    ax = axes[0]
    feat_labels_tr = ['Distance', 'Seed', 'TR Rating', 'SOS Rating']
    coefs_tr = list(lr4.coef_[0])
    fc_tr = [RED if c < 0 else GREEN for c in coefs_tr]
    sorted_idx = np.argsort(np.abs(coefs_tr))
    ax.barh([feat_labels_tr[i] for i in sorted_idx], [coefs_tr[i] for i in sorted_idx],
            color=[fc_tr[i] for i in sorted_idx], alpha=0.85, height=0.5, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for i, idx in enumerate(sorted_idx):
        c = coefs_tr[idx]
        ax.text(c + (0.02 if c >= 0 else -0.02), i, f'{c:+.3f}',
                ha='left' if c >= 0 else 'right', va='center', fontsize=10, color=LIGHT)
    ax.set_xlabel('Standardized Coefficient', fontsize=10)
    ax.set_title(f'Enhanced Model (AUC={cv4.mean():.3f})', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

    # Full model coefficients
    ax = axes[1]
    feat_labels_full = ['Distance', 'Seed', 'TR Rating', 'SOS Rating',
                        'KenPom EM', 'EM Change', 'Top-25 Wins']
    coefs_full = list(lr5.coef_[0])
    fc_full = [RED if c < 0 else GREEN for c in coefs_full]
    sorted_idx = np.argsort(np.abs(coefs_full))
    ax.barh([feat_labels_full[i] for i in sorted_idx], [coefs_full[i] for i in sorted_idx],
            color=[fc_full[i] for i in sorted_idx], alpha=0.85, height=0.45, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for i, idx in enumerate(sorted_idx):
        c = coefs_full[idx]
        ax.text(c + (0.02 if c >= 0 else -0.02), i, f'{c:+.3f}',
                ha='left' if c >= 0 else 'right', va='center', fontsize=10, color=LIGHT)
    ax.set_xlabel('Standardized Coefficient', fontsize=10)
    ax.set_title(f'Full Model (AUC={cv5.mean():.3f})', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    st.markdown("#### Model Summary")
    model_summary = pd.DataFrame([
        {'Model': '1. Distance Only', 'Features': 'Distance', 'AUC': f'{cv1.mean():.4f}',
         '± Std': f'{cv1.std():.4f}', 'Observations': f'{len(y_all):,}', 'Years': '2008–2025'},
        {'Model': '2. Distance + Seed', 'Features': 'Distance, Seed', 'AUC': f'{cv2.mean():.4f}',
         '± Std': f'{cv2.std():.4f}', 'Observations': f'{len(y_all):,}', 'Years': '2008–2025'},
        {'Model': '3. Dist+Seed+TZ', 'Features': 'Distance, Seed, Time Zones', 'AUC': f'{cv3.mean():.4f}',
         '± Std': f'{cv3.std():.4f}', 'Observations': f'{len(y_all):,}', 'Years': '2008–2025'},
        {'Model': '4. Enhanced (TR)', 'Features': 'Distance, Seed, TR Rating, SOS',
         'AUC': f'{cv4.mean():.4f}', '± Std': f'{cv4.std():.4f}',
         'Observations': f'{len(y4):,}', 'Years': '2008–2025'},
        {'Model': '5. Full (TR+KenPom)', 'Features': 'Distance, Seed, TR, SOS, KenPom EM, EM Change, Top-25 W',
         'AUC': f'{cv5.mean():.4f}', '± Std': f'{cv5.std():.4f}',
         'Observations': f'{len(y5):,}', 'Years': '2012–2025'},
    ])
    st.dataframe(model_summary, use_container_width=True, hide_index=True)

    best_model = 'Full (TR+KenPom)' if cv5.mean() > cv4.mean() else 'Enhanced (TR)'
    improvement = max(cv4.mean(), cv5.mean()) - cv2.mean()
    st.info(f"**Key insight:** The **{best_model}** model achieves the highest AUC, "
            f"a **{improvement:.3f}** improvement over the simple Distance+Seed model. "
            f"KenPom efficiency margin is the single strongest predictor of tournament success.")



# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("NCAA Tournament Analytics Hub · Data: 2008–2025 · "
           "Built with Streamlit + scikit-learn + matplotlib")
