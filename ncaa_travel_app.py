"""
🏀 NCAA Tournament: Travel Distance Analyzer
Streamlit application for exploring how travel distance affects NCAA tournament outcomes.

Usage:
    pip install streamlit pandas numpy scipy scikit-learn matplotlib
    streamlit run ncaa_travel_app.py

Place the following CSV files in the same directory:
    - Tournament Locations.csv
    - Tournament Matchups.csv
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
    page_title="NCAA Travel Distance Analyzer",
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

    return {
        'team_data': team_data, 'games_df': games_df,
        'y_all': y_all,
        'sc1': sc1, 'X1s': X1s, 'lr1': lr1, 'cv1': cv1,
        'sc2': sc2, 'X2s': X2s, 'lr2': lr2, 'cv2': cv2,
        'sc3': sc3, 'X3s': X3s, 'lr3': lr3, 'cv3': cv3,
        'all_teams': all_teams, 'all_years': all_years,
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
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"**Error loading data:** {e}\n\nMake sure `Tournament Locations.csv` and "
             f"`Tournament Matchups.csv` are in the same directory as this script.")
    st.stop()

round_map = {64: 'Round of 64', 32: 'Round of 32', 16: 'Sweet 16',
             8: 'Elite 8', 4: 'Final 4', 2: 'Championship'}


# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🏀 NCAA Tournament: Travel Distance Analyzer")
st.markdown(
    f"**Machine learning analysis of how travel distance affects tournament outcomes**  \n"
    f"`{len(team_data):,} observations · {len(games_df):,} games · "
    f"{min(all_years)}–{max(all_years)} · Logistic Regression + Correlation Analysis`"
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_pred, tab_exp, tab_rnd, tab_seed, tab_team, tab_h2h, tab_model = st.tabs([
    "🎯 Win Predictor",
    "📊 Distance Explorer",
    "🔄 By Round",
    "🌱 By Seed",
    "🏫 Team Lookup",
    "⚔️ Head-to-Head",
    "🤖 Models",
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1: WIN PROBABILITY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2: DISTANCE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3: ROUND-BY-ROUND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4: SEED DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5: TEAM LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 6: HEAD-TO-HEAD SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 7: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

with tab_model:
    st.subheader("Model Comparison")
    st.caption("Compare the three logistic regression models and their feature importance.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Distance Only AUC", f"{cv1.mean():.4f} ± {cv1.std():.4f}")
    m2.metric("Distance + Seed AUC", f"{cv2.mean():.4f} ± {cv2.std():.4f}")
    m3.metric("Full Model AUC", f"{cv3.mean():.4f} ± {cv3.std():.4f}")

    dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel 1: ROC Curves
    ax = axes[0]
    for Xs, model, name, color, ls in [
        (X1s, lr1, f'Distance (AUC={cv1.mean():.3f})', ORANGE, '-'),
        (X2s, lr2, f'Dist+Seed (AUC={cv2.mean():.3f})', GREEN, '-'),
        (X3s, lr3, f'Full (AUC={cv3.mean():.3f})', BLUE, '--'),
    ]:
        yp = model.predict_proba(Xs)[:, 1]
        fpr, tpr, _ = roc_curve(y_all, yp)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls, label=name, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=':', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.3)
    ax.grid(alpha=0.2); ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: AUC bars
    ax = axes[1]
    names = ['Distance\nOnly', 'Distance\n+ Seed', 'Full\nModel']
    aucs = [cv1.mean(), cv2.mean(), cv3.mean()]
    stds = [cv1.std(), cv2.std(), cv3.std()]
    bc = [ORANGE, GREEN, BLUE]
    bars = ax.bar(range(3), aucs, yerr=stds, capsize=6, color=bc, alpha=0.85,
                  width=0.5, error_kw={'color': MUTED, 'linewidth': 1.5}, zorder=3)
    for i, (b, a) in enumerate(zip(bars, aucs)):
        ax.text(b.get_x() + b.get_width() / 2, a + stds[i] + 0.01,
                f'{a:.3f}', ha='center', fontsize=12, fontweight='bold', color=LIGHT)
    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('AUC-ROC', fontsize=10)
    ax.set_title('Model AUC (5-fold CV)', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0.45, 0.82)
    ax.grid(axis='y', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

    # Panel 3: Feature coefficients
    ax = axes[2]
    feats = ['Distance', 'Seed', 'Time Zones']
    coefs = list(lr3.coef_[0])
    fc = [RED if c < 0 else GREEN for c in coefs]
    bars = ax.barh(range(len(feats)), coefs, color=fc, alpha=0.85, height=0.45, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for i, (b, c) in enumerate(zip(bars, coefs)):
        ax.text(c + (0.02 if c >= 0 else -0.02), i, f'{c:+.4f}',
                ha='left' if c >= 0 else 'right', va='center', fontsize=10, color=LIGHT)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=11)
    ax.set_xlabel('Standardized Coefficient', fontsize=10)
    ax.set_title('Feature Importance\n(Full Model)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3); ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Key insight
    st.info(f"**Key insight:** Seed is ~{abs(coefs[1] / coefs[0]):.0f}× more predictive than distance. "
            f"Time zones add no independent value.")


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("NCAA Tournament Travel Distance Analyzer · Data: 2008–2025 · "
           "Built with Streamlit + scikit-learn + matplotlib")
