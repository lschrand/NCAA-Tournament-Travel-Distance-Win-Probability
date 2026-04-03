"""
🏀 NCAA Tournament Analytics Hub
Streamlit application for predicting and analyzing March Madness matchups.

Usage:
    pip install streamlit pandas numpy scipy scikit-learn plotly
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="NCAA Tournament Analytics Hub", page_icon="🏀",
                    layout="wide", initial_sidebar_state="expanded")

# ── Colors ────────────────────────────────────────────────────────────────
ORANGE, GREEN, RED, BLUE, CYAN = '#f97316', '#22c55e', '#ef4444', '#3b82f6', '#06b6d4'
MUTED, LIGHT, PURPLE, BG, CARD = '#64748b', '#e2e8f0', '#a855f7', '#0f1117', '#161b22'

def plotly_layout(**kw):
    base = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=LIGHT, size=12),
                xaxis=dict(gridcolor='#1e2530', zerolinecolor='#2a2f3a'),
                yaxis=dict(gridcolor='#1e2530', zerolinecolor='#2a2f3a'),
                margin=dict(l=60, r=30, t=50, b=50),
                hoverlabel=dict(bgcolor=CARD, font_size=12, font_color=LIGHT),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)))
    base.update(kw)
    return base

# ── CSS: #3 – differentiate master vs sub-tab styling ─────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
.block-container { max-width: 1250px; padding-top: 1rem; }
h1,h2,h3,h4,h5,h6,p,li,span,label,.stMarkdown { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] { background-color: #161b22; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f97316 !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background-color: #161b22; border-radius: 10px; padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #94a3b8 !important;
    border-radius: 8px; padding: 10px 20px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: #ffffff !important;
}
.stSlider > div > div > div { color: #e2e8f0 !important; }
div[data-testid="stMetric"] {
    background-color: #161b22; border: 1px solid #2a2f3a;
    border-radius: 8px; padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f97316 !important; }
.stButton>button { background-color: #f97316 !important; color: white !important;
    border: none; border-radius: 6px; font-weight: 600; }
.stButton>button:hover { background-color: #ea580c !important; }
.stSelectbox>div>div, .stMultiSelect>div>div {
    background-color: #161b22; border-color: #2a2f3a; }
.landing-card {
    background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
    border: 1px solid #2a2f3a; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 16px;
}
.landing-card h4 { margin: 0 0 6px 0; color: #f97316 !important; }
.landing-card p { margin: 0; color: #94a3b8 !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_and_process_data():
    data_dir = '.'
    for c in ['.', 'data', os.path.dirname(__file__) if '__file__' in dir() else '.']:
        if os.path.exists(os.path.join(c, 'Tournament Locations.csv')):
            data_dir = c; break
    locations = pd.read_csv(os.path.join(data_dir, 'Tournament Locations.csv'))
    matchups  = pd.read_csv(os.path.join(data_dir, 'Tournament Matchups.csv'))
    merged = matchups.merge(
        locations[['YEAR','BY YEAR NO','TEAM','DISTANCE (MI)','DISTANCE (KM)',
                   'TIME ZONES CROSSED','DIRECTION','COLLEGE LATITUDE',
                   'COLLEGE LONGITUDE','GAME LATITUDE','GAME LONGITUDE',
                   'GAME CITY','GAME STATE']],
        on=['YEAR','BY YEAR NO','TEAM'], how='inner')
    merged['WON'] = (merged['ROUND'] < merged['CURRENT ROUND']).astype(int)
    gl = []
    for (yr, cr), g in merged.groupby(['YEAR','CURRENT ROUND']):
        g = g.sort_values('BY YEAR NO', ascending=False).reset_index(drop=True)
        for i in range(0, len(g)-1, 2):
            t1, t2 = g.iloc[i], g.iloc[i+1]
            gl.append({'YEAR': yr, 'CURRENT ROUND': cr,
                'TEAM1': t1['TEAM'], 'SEED1': t1['SEED'], 'DIST1': t1['DISTANCE (MI)'],
                'WON1': t1['WON'], 'SCORE1': t1['SCORE'],
                'TEAM2': t2['TEAM'], 'SEED2': t2['SEED'], 'DIST2': t2['DISTANCE (MI)'],
                'WON2': t2['WON'], 'SCORE2': t2['SCORE'],
                'GAME_CITY': t1['GAME CITY'], 'GAME_STATE': t1['GAME STATE']})
    games_df = pd.DataFrame(gl)
    games_df['CLOSER_WON'] = np.where(games_df['DIST1']<games_df['DIST2'], games_df['WON1'], games_df['WON2'])
    td = merged.copy(); ya = td['WON'].values
    sc1 = StandardScaler(); X1 = sc1.fit_transform(td[['DISTANCE (MI)']].values)
    lr1 = LogisticRegression(random_state=42).fit(X1, ya)
    cv1 = cross_val_score(lr1, X1, ya, cv=5, scoring='roc_auc')
    sc2 = StandardScaler(); X2 = sc2.fit_transform(td[['DISTANCE (MI)','SEED']].values)
    lr2 = LogisticRegression(random_state=42).fit(X2, ya)
    cv2 = cross_val_score(lr2, X2, ya, cv=5, scoring='roc_auc')
    sc3 = StandardScaler(); X3 = sc3.fit_transform(td[['DISTANCE (MI)','SEED','TIME ZONES CROSSED']].values)
    lr3 = LogisticRegression(random_state=42).fit(X3, ya)
    cv3 = cross_val_score(lr3, X3, ya, cv=5, scoring='roc_auc')
    trk = pd.read_csv(os.path.join(data_dir, 'TeamRankings.csv'))
    kp  = pd.read_csv(os.path.join(data_dir, 'KenPom Preseason.csv'))
    cs  = pd.read_csv(os.path.join(data_dir, 'Conference Stats.csv'))
    enh = td.copy()
    enh = enh.merge(trk[['YEAR','TEAM NO','TR RANK','TR RATING','SOS RANK','SOS RATING',
        'V 1-25 WINS','V 1-25 LOSS','V 26-50 WINS','V 26-50 LOSS',
        'CONSISTENCY RANK','CONSISTENCY TR RATING','LUCK RANK','LUCK RATING']],
        on=['YEAR','TEAM NO'], how='left')
    enh = enh.merge(kp[['YEAR','TEAM NO','PRESEASON KADJ EM','PRESEASON KADJ EM RANK',
        'PRESEASON KADJ O','PRESEASON KADJ D','KADJ EM RANK CHANGE','KADJ EM CHANGE']],
        on=['YEAR','TEAM NO'], how='left')
    f_tr = ['DISTANCE (MI)','SEED','TR RATING','SOS RATING']
    dt = enh.dropna(subset=f_tr); sc4 = StandardScaler()
    X4 = sc4.fit_transform(dt[f_tr].values); y4 = dt['WON'].values
    lr4 = LogisticRegression(random_state=42).fit(X4, y4)
    cv4 = cross_val_score(lr4, X4, y4, cv=5, scoring='roc_auc')
    f_fl = ['DISTANCE (MI)','SEED','TR RATING','SOS RATING','PRESEASON KADJ EM','KADJ EM CHANGE','V 1-25 WINS']
    df = enh.dropna(subset=f_fl); sc5 = StandardScaler()
    X5 = sc5.fit_transform(df[f_fl].values); y5 = df['WON'].values
    lr5 = LogisticRegression(random_state=42).fit(X5, y5)
    cv5 = cross_val_score(lr5, X5, y5, cv=5, scoring='roc_auc')
    return dict(team_data=td, games_df=games_df, y_all=ya,
        sc1=sc1, X1s=X1, lr1=lr1, cv1=cv1, sc2=sc2, X2s=X2, lr2=lr2, cv2=cv2,
        sc3=sc3, X3s=X3, lr3=lr3, cv3=cv3, all_teams=sorted(td['TEAM'].unique()),
        all_years=sorted(td['YEAR'].unique()), enhanced=enh, team_rankings=trk,
        kenpom=kp, conf_stats=cs, features_tr=f_tr, sc4=sc4, X4s=X4, lr4=lr4, cv4=cv4, y4=y4,
        features_full=f_fl, sc5=sc5, X5s=X5, lr5=lr5, cv5=cv5, y5=y5)

try:
    D = load_and_process_data()
    team_data, games_df, y_all = D['team_data'], D['games_df'], D['y_all']
    sc1, X1s, lr1, cv1 = D['sc1'], D['X1s'], D['lr1'], D['cv1']
    sc2, X2s, lr2, cv2 = D['sc2'], D['X2s'], D['lr2'], D['cv2']
    sc3, X3s, lr3, cv3 = D['sc3'], D['X3s'], D['lr3'], D['cv3']
    all_teams, all_years = D['all_teams'], D['all_years']
    enhanced, conf_stats = D['enhanced'], D['conf_stats']
    features_tr, sc4, X4s, lr4, cv4, y4 = D['features_tr'], D['sc4'], D['X4s'], D['lr4'], D['cv4'], D['y4']
    features_full, sc5, X5s, lr5, cv5, y5 = D['features_full'], D['sc5'], D['X5s'], D['lr5'], D['cv5'], D['y5']
except Exception as e:
    st.error(f"**Error loading data:** {e}\n\nMake sure `Tournament Locations.csv`, "
             "`Tournament Matchups.csv`, `TeamRankings.csv`, `KenPom Preseason.csv`, "
             "and `Conference Stats.csv` are in the same directory as this script.")
    st.stop()
round_map = {64:'Round of 64',32:'Round of 32',16:'Sweet 16',8:'Elite 8',4:'Final 4',2:'Championship'}

# ── #2: Global sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏀 Global Filters")
    st.caption("These apply across all tabs.")
    g_years = st.slider("Year Range", int(min(all_years)), int(max(all_years)),
                         (int(min(all_years)), int(max(all_years))), key="g_yr")
    g_seeds = st.slider("Seed Range", 1, 16, (1, 16), key="g_sd")
    g_fav = st.selectbox("Favorite Team", ["—"] + all_teams, key="g_fav")
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    _f = team_data[(team_data['YEAR'].between(*g_years)) & (team_data['SEED'].between(*g_seeds))]
    st.metric("Filtered Obs", f"{len(_f):,}")
    st.metric("Win Rate", f"{_f['WON'].mean():.1%}" if len(_f) else "—")
    st.metric("Avg Distance", f"{_f['DISTANCE (MI)'].mean():.0f} mi" if len(_f) else "—")
    if g_fav != "—":
        _fv = _f[_f['TEAM']==g_fav]
        if len(_fv):
            st.markdown(f"### 🌟 {g_fav}")
            st.metric("Games", f"{len(_fv)}"); st.metric("Win Rate", f"{_fv['WON'].mean():.1%}")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("# 🏀 NCAA Tournament Analytics Hub")
st.markdown(f"**Data-driven tools for predicting and analyzing March Madness matchups**  \n"
    f"`{len(team_data):,} obs · {len(games_df):,} games · {min(all_years)}–{max(all_years)} · "
    f"5 ML models · Team, Conference & Distance analytics`")
st.markdown("---")

# ── Master tabs ───────────────────────────────────────────────────────────
master_predict, master_travel, master_conf, master_models = st.tabs([
    "🎯 Predictions & Matchups", "📊 Travel & Distance Analysis",
    "🏛️ Conference Intelligence", "🤖 Models & Methodology"])

# ═══════════════════════════════════════════════════════════════════════════
#  MASTER 1: PREDICTIONS & MATCHUPS
# ═══════════════════════════════════════════════════════════════════════════
with master_predict:
    st.markdown('<div class="landing-card"><h4>🎯 Predictions & Matchups</h4>'
        '<p>Predict game outcomes using 5 ML models trained on 17 seasons of tournament data. '
        'Compare teams head-to-head, look up any team\'s travel history, or use the enhanced '
        'predictor with KenPom efficiency and power ratings.</p></div>', unsafe_allow_html=True)
    tab_pred, tab_enh, tab_h2h, tab_team = st.tabs([
        "🎯 Win Predictor","🔬 Enhanced Predictor","⚔️ Head-to-Head","🏫 Team Lookup"])

    with tab_pred:
        st.subheader("Win Probability Predictor")
        st.caption("Adjust distance, seed, and round to see the predicted win probability.")
        c1,c2,c3 = st.columns(3)
        with c1: distance = st.slider("Distance (mi)", 0, 2500, 500, 25, key="p_d")
        with c2: seed = st.slider("Seed", 1, 16, 4, key="p_s")
        with c3: tround = st.selectbox("Round", list(round_map.values()), key="p_r")
        rv = {v:k for k,v in round_map.items()}[tround]
        prob = lr2.predict_proba(sc2.transform([[distance, seed]]))[0][1]
        base = lr2.predict_proba(sc2.transform([[0, seed]]))[0][1]
        hist = team_data[(team_data['CURRENT ROUND']==rv)&(team_data['SEED']==seed)]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Win Probability", f"{prob:.1%}")
        m2.metric("Baseline (0 mi)", f"{base:.1%}")
        m3.metric("Distance Penalty", f"{prob-base:+.1%}")
        if len(hist): m4.metric(f"Historical WR", f"{hist['WON'].mean():.1%}")
        fig = make_subplots(1, 3, specs=[[{"type":"indicator"},{"type":"xy"},{"type":"xy"}]],
            subplot_titles=[f"Win Probability: {prob:.1%}", f"Seed {seed} Win Curve",
                            f"Historical: Seed {seed} in {tround}"])
        c = GREEN if prob>=.55 else (ORANGE if prob>=.45 else RED)
        fig.add_trace(go.Indicator(mode="gauge+number", value=prob*100,
            number=dict(suffix="%", font=dict(size=36, color=c)),
            gauge=dict(axis=dict(range=[0,100], tickcolor=MUTED), bar=dict(color=c),
                bgcolor='rgba(0,0,0,0)',
                steps=[dict(range=[0,40],color='rgba(239,68,68,0.15)'),
                       dict(range=[40,60],color='rgba(249,115,22,0.15)'),
                       dict(range=[60,100],color='rgba(34,197,94,0.15)')])), row=1, col=1)
        dr = np.linspace(0,2500,200)
        pr = [lr2.predict_proba(sc2.transform([[d,seed]]))[0][1]*100 for d in dr]
        fig.add_trace(go.Scatter(x=dr, y=pr, mode='lines', line=dict(color=ORANGE,width=3),
            name='Win %', hovertemplate='%{x:.0f} mi → %{y:.1f}%'), row=1, col=2)
        # Use add_shape instead of add_hline/add_vline (indicator subplot breaks row/col refs)
        fig.add_shape(type="line", x0=0, x1=2500, y0=50, y1=50,
            line=dict(dash="dash", color=MUTED), xref="x2", yref="y2")
        fig.add_shape(type="line", x0=distance, x1=distance, y0=0, y1=100,
            line=dict(color=CYAN, width=2), xref="x2", yref="y2")
        fig.add_trace(go.Scatter(x=[distance], y=[prob*100], mode='markers+text',
            marker=dict(size=12, color=CYAN), text=[f" {prob:.1%}"], textposition="middle right",
            textfont=dict(color=CYAN, size=13), showlegend=False), row=1, col=2)
        if len(hist)>5:
            fig.add_trace(go.Histogram(x=hist[hist['WON']==1]['DISTANCE (MI)'],
                name=f'Wins', marker_color=GREEN, opacity=0.6, histnorm='probability density'), row=1, col=3)
            fig.add_trace(go.Histogram(x=hist[hist['WON']==0]['DISTANCE (MI)'],
                name=f'Losses', marker_color=RED, opacity=0.5, histnorm='probability density'), row=1, col=3)
            fig.add_shape(type="line", x0=distance, x1=distance, y0=0, y1=1,
                line=dict(color=CYAN, width=2), xref="x3", yref="y3 domain")
        fig.update_layout(**plotly_layout(height=400, showlegend=True, barmode='overlay'))
        fig.update_xaxes(title_text="Distance (mi)", row=1, col=2)
        fig.update_yaxes(title_text="Win Prob (%)", row=1, col=2)
        fig.update_xaxes(title_text="Distance (mi)", row=1, col=3)
        st.plotly_chart(fig, use_container_width=True)

    with tab_enh:
        st.subheader("Enhanced Game Predictor")
        st.caption("Uses TR power ratings, KenPom efficiency, and SOS to predict matchups.")
        mode = st.radio("Mode", ["Pick Tournament Teams","Custom Input"], horizontal=True, key="em")
        if mode == "Pick Tournament Teams":
            c1,c2,c3 = st.columns([1,2,2])
            with c1: ey = st.selectbox("Year", sorted(enhanced['YEAR'].unique(), reverse=True), key="ey")
            yt = sorted(enhanced[enhanced['YEAR']==ey]['TEAM'].unique())
            with c2: ta = st.selectbox("Team A", yt, 0, key="eta")
            with c3: tb = st.selectbox("Team B", yt, min(1,len(yt)-1), key="etb")
            if ta==tb: st.warning("Pick two different teams.")
            else:
                ra = enhanced[(enhanced['YEAR']==ey)&(enhanced['TEAM']==ta)].iloc[0]
                rb = enhanced[(enhanced['YEAR']==ey)&(enhanced['TEAM']==tb)].iloc[0]
                st.markdown("#### Team Profiles")
                ca, cb = st.columns(2)
                for col, r, lbl in [(ca,ra,"A"),(cb,rb,"B")]:
                    with col:
                        st.markdown(f"**Team {lbl}: {r['TEAM']}**")
                        p1,p2,p3=st.columns(3)
                        p1.metric("Seed",f"#{int(r['SEED'])}")
                        p2.metric("TR Rating", f"{r['TR RATING']:.1f}" if pd.notna(r.get('TR RATING')) else "N/A")
                        p3.metric("SOS", f"{r['SOS RATING']:.1f}" if pd.notna(r.get('SOS RATING')) else "N/A")
                preds = []
                pa=lr2.predict_proba(sc2.transform([[ra['DISTANCE (MI)'],ra['SEED']]]))[0][1]
                pb=lr2.predict_proba(sc2.transform([[rb['DISTANCE (MI)'],rb['SEED']]]))[0][1]
                preds.append(("Simple", pa/(pa+pb), pb/(pa+pb), cv2.mean()))
                try:
                    va=[ra[f] for f in features_tr]; vb=[rb[f] for f in features_tr]
                    if all(pd.notna(v) for v in va+vb):
                        pa=lr4.predict_proba(sc4.transform([va]))[0][1]
                        pb=lr4.predict_proba(sc4.transform([vb]))[0][1]
                        preds.append(("Enhanced", pa/(pa+pb), pb/(pa+pb), cv4.mean()))
                except: pass
                try:
                    va=[ra[f] for f in features_full]; vb=[rb[f] for f in features_full]
                    if all(pd.notna(v) for v in va+vb):
                        pa=lr5.predict_proba(sc5.transform([va]))[0][1]
                        pb=lr5.predict_proba(sc5.transform([vb]))[0][1]
                        preds.append(("Full", pa/(pa+pb), pb/(pa+pb), cv5.mean()))
                except: pass
                st.markdown("#### Model Predictions")
                fig = make_subplots(1, len(preds), subplot_titles=[f"{n} (AUC={a:.3f})" for n,_,_,a in preds])
                for i,(n,rra,rrb,a) in enumerate(preds,1):
                    fig.add_trace(go.Bar(y=[ta],x=[rra*100],orientation='h',marker_color=GREEN,
                        showlegend=(i==1),name=ta,text=[f"{rra:.1%}"],textposition='outside'),row=1,col=i)
                    fig.add_trace(go.Bar(y=[tb],x=[rrb*100],orientation='h',marker_color=RED,
                        showlegend=(i==1),name=tb,text=[f"{rrb:.1%}"],textposition='outside'),row=1,col=i)
                    fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=i)
                fig.update_layout(**plotly_layout(height=300)); fig.update_xaxes(range=[0,105])
                st.plotly_chart(fig, use_container_width=True)
                ws = [ta if a>b else tb for _,a,b,_ in preds]
                if len(set(ws))==1: st.success(f"All models agree: **{ws[0]}** wins.")
                else: st.warning("Models disagree — potential upset territory!")
        else:
            st.markdown("#### Enter Custom Team Stats")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Team A**")
                csa=st.slider("Seed",1,16,3,key="csa"); cda=st.slider("Dist (mi)",0,2500,200,25,key="cda")
                ctra=st.number_input("TR Rating",value=15.0,step=0.5,key="ctra")
                csosa=st.number_input("SOS Rating",value=8.0,step=0.5,key="csosa")
                ckema=st.number_input("KenPom EM",value=20.0,step=0.5,key="ckema")
                ckchga=st.number_input("EM Change",value=2.0,step=0.5,key="ckchga")
                cv25a=st.number_input("Top-25 Wins",value=5,step=1,key="cv25a")
            with cb:
                st.markdown("**Team B**")
                csb=st.slider("Seed",1,16,11,key="csb"); cdb=st.slider("Dist (mi)",0,2500,800,25,key="cdb")
                ctrb=st.number_input("TR Rating",value=6.0,step=0.5,key="ctrb")
                csosb=st.number_input("SOS Rating",value=2.0,step=0.5,key="csosb")
                ckemb=st.number_input("KenPom EM",value=10.0,step=0.5,key="ckemb")
                ckchgb=st.number_input("EM Change",value=5.0,step=0.5,key="ckchgb")
                cv25b=st.number_input("Top-25 Wins",value=1,step=1,key="cv25b")
            mc = []
            pa=lr2.predict_proba(sc2.transform([[cda,csa]]))[0][1]
            pb=lr2.predict_proba(sc2.transform([[cdb,csb]]))[0][1]; mc.append(("Simple",pa/(pa+pb),pb/(pa+pb)))
            pa=lr4.predict_proba(sc4.transform([[cda,csa,ctra,csosa]]))[0][1]
            pb=lr4.predict_proba(sc4.transform([[cdb,csb,ctrb,csosb]]))[0][1]; mc.append(("Enhanced",pa/(pa+pb),pb/(pa+pb)))
            pa=lr5.predict_proba(sc5.transform([[cda,csa,ctra,csosa,ckema,ckchga,cv25a]]))[0][1]
            pb=lr5.predict_proba(sc5.transform([[cdb,csb,ctrb,csosb,ckemb,ckchgb,cv25b]]))[0][1]; mc.append(("Full",pa/(pa+pb),pb/(pa+pb)))
            fig = make_subplots(1,3,subplot_titles=[n for n,_,_ in mc])
            for i,(n,ra,rb) in enumerate(mc,1):
                fig.add_trace(go.Bar(y=[f'A (#{csa})'],x=[ra*100],orientation='h',marker_color=GREEN,
                    showlegend=False,text=[f"{ra:.1%}"],textposition='outside'),row=1,col=i)
                fig.add_trace(go.Bar(y=[f'B (#{csb})'],x=[rb*100],orientation='h',marker_color=RED,
                    showlegend=False,text=[f"{rb:.1%}"],textposition='outside'),row=1,col=i)
                fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=i)
            fig.update_layout(**plotly_layout(height=280)); fig.update_xaxes(range=[0,105])
            st.plotly_chart(fig, use_container_width=True)

    with tab_h2h:
        st.subheader("Head-to-Head Simulator")
        ca, cb = st.columns(2)
        with ca: st.markdown("**Team A**"); sa=st.slider("Seed A",1,16,3,key="ha"); da=st.slider("Dist A (mi)",0,2500,200,25,key="hda")
        with cb: st.markdown("**Team B**"); sb=st.slider("Seed B",1,16,11,key="hb"); db=st.slider("Dist B (mi)",0,2500,1200,25,key="hdb")
        pa=lr2.predict_proba(sc2.transform([[da,sa]]))[0][1]; pb=lr2.predict_proba(sc2.transform([[db,sb]]))[0][1]
        ra,rb = pa/(pa+pb), pb/(pa+pb)
        m1,m2,m3=st.columns(3); m1.metric("Team A",f"{ra:.1%}"); m2.metric("Team B",f"{rb:.1%}")
        m3.metric("Winner", f"{'A' if ra>rb else 'B'} (+{abs(ra-rb):.1%})")
        fig = make_subplots(1,2,subplot_titles=["Head-to-Head","Sensitivity: Team A Distance"])
        fig.add_trace(go.Bar(y=[f'A (#{sa})'],x=[ra*100],orientation='h',marker_color=GREEN,showlegend=False,text=[f"{ra:.1%}"],textposition='outside'),row=1,col=1)
        fig.add_trace(go.Bar(y=[f'B (#{sb})'],x=[rb*100],orientation='h',marker_color=RED,showlegend=False,text=[f"{rb:.1%}"],textposition='outside'),row=1,col=1)
        fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=1)
        dr=np.linspace(0,2500,100); pv=[lr2.predict_proba(sc2.transform([[d,sa]]))[0][1] for d in dr]
        rv=[p/(p+pb)*100 for p in pv]
        fig.add_trace(go.Scatter(x=dr,y=rv,mode='lines',line=dict(color=GREEN,width=3),showlegend=False),row=1,col=2)
        fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=2)
        fig.add_vline(x=da,line_color=CYAN,line_width=2,row=1,col=2)
        fig.update_layout(**plotly_layout(height=350)); fig.update_xaxes(range=[0,105],row=1,col=1)
        fig.update_xaxes(title_text="Team A Distance (mi)",row=1,col=2)
        fig.update_yaxes(title_text="Team A Win Prob (%)",row=1,col=2)
        st.plotly_chart(fig, use_container_width=True)

    with tab_team:
        st.subheader("Team Lookup")
        defteam = g_fav if g_fav!="—" else ('Duke' if 'Duke' in all_teams else all_teams[0])
        tn = st.selectbox("Select Team", all_teams, index=all_teams.index(defteam) if defteam in all_teams else 0, key="tl")
        df = team_data[team_data['TEAM']==tn]
        if not len(df): st.warning(f"No data for {tn}.")
        else:
            med=df['DISTANCE (MI)'].median(); cl=df[df['DISTANCE (MI)']<=med]; fa=df[df['DISTANCE (MI)']>med]
            yrs=sorted(df['YEAR'].unique())
            m1,m2,m3,m4=st.columns(4); m1.metric("Appearances",f"{len(yrs)}"); m2.metric("Games",f"{len(df)}")
            m3.metric("Win Rate",f"{df['WON'].mean():.1%}"); m4.metric("Avg Dist",f"{df['DISTANCE (MI)'].mean():.0f} mi")
            fig=make_subplots(1,3,subplot_titles=[f"{tn} Games","Close vs Far","Travel Distribution"])
            w=df[df['WON']==1]; l=df[df['WON']==0]
            fig.add_trace(go.Scatter(x=w['DISTANCE (MI)'],y=w['YEAR'],mode='markers',
                marker=dict(size=10,color=GREEN),name=f'Wins ({len(w)})'),row=1,col=1)
            fig.add_trace(go.Scatter(x=l['DISTANCE (MI)'],y=l['YEAR'],mode='markers',
                marker=dict(size=10,color=RED,symbol='x'),name=f'Losses ({len(l)})'),row=1,col=1)
            cwr=cl['WON'].mean()*100 if len(cl) else 0; fwr=fa['WON'].mean()*100 if len(fa) else 0
            fig.add_trace(go.Bar(x=[f'Close (<{med:.0f}mi)',f'Far (>{med:.0f}mi)'],y=[cwr,fwr],
                marker_color=[GREEN,RED],text=[f"{cwr:.1f}%",f"{fwr:.1f}%"],textposition='outside',showlegend=False),row=1,col=2)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=2)
            fig.add_trace(go.Histogram(x=df['DISTANCE (MI)'],marker_color=BLUE,opacity=0.7,showlegend=False),row=1,col=3)
            fig.add_vline(x=df['DISTANCE (MI)'].mean(),line_dash="dash",line_color=ORANGE,line_width=2,row=1,col=3)
            fig.update_layout(**plotly_layout(height=400,showlegend=True))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Years: {', '.join(str(y) for y in yrs)}")

# ═══════════════════════════════════════════════════════════════════════════
#  MASTER 2: TRAVEL & DISTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with master_travel:
    st.markdown('<div class="landing-card"><h4>📊 Travel & Distance Analysis</h4>'
        '<p>Explore the relationship between travel distance and tournament outcomes. '
        'Does playing closer to home give teams an edge? Break down the data by distance buckets, '
        'tournament rounds, and seed lines. Uses global filters from the sidebar.</p></div>', unsafe_allow_html=True)
    tab_exp, tab_rnd, tab_seed = st.tabs(["📊 Distance Explorer","🔄 By Round","🌱 By Seed"])

    with tab_exp:
        st.subheader("Distance vs Win Rate Explorer")
        rf = st.selectbox("Round", ['All Rounds']+list(round_map.values()), key="er")
        rv = {v:k for k,v in round_map.items()}.get(rf)
        df=team_data.copy()
        if rv: df=df[df['CURRENT ROUND']==rv]
        df=df[(df['SEED'].between(*g_seeds))&(df['YEAR'].between(*g_years))]
        if len(df)<20: st.warning(f"Not enough data (n={len(df)}). Broaden sidebar filters.")
        else:
            corr,pv=stats.pointbiserialr(df['WON'],df['DISTANCE (MI)'])
            wm=df[df['WON']==1]['DISTANCE (MI)'].mean(); lm=df[df['WON']==0]['DISTANCE (MI)'].mean()
            m1,m2,m3,m4=st.columns(4); m1.metric("Obs",f"{len(df):,}"); m2.metric("r",f"{corr:+.4f}")
            m3.metric("p",f"{pv:.6f}"); m4.metric("Losers +",f"{lm-wm:+.0f} mi")
            bk=[0,100,250,500,750,1000,1500,3500]; lb=['0–100','100–250','250–500','500–750','750–1K','1K–1.5K','1.5K+']
            dp=df.copy(); dp['BK']=pd.cut(dp['DISTANCE (MI)'],bins=bk,labels=lb)
            bs=dp.groupby('BK',observed=True).agg(wr=('WON','mean'),n=('WON','count')).reset_index()
            bs=bs[bs['n']>=5]
            fig=make_subplots(1,2,subplot_titles=["Win Rate by Distance Bucket","Outcomes with Trend"])
            cs2=[GREEN if w>=0.5 else RED for w in bs['wr']]
            fig.add_trace(go.Bar(x=bs['BK'],y=bs['wr']*100,marker_color=cs2,
                text=[f"{w:.1%}<br>n={n}" for w,n in zip(bs['wr'],bs['n'])],textposition='outside',showlegend=False),row=1,col=1)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
            if df['WON'].nunique()>1:
                st2=StandardScaler(); xt=st2.fit_transform(df[['DISTANCE (MI)']].values)
                lt=LogisticRegression(random_state=42).fit(xt,df['WON'].values)
                xr=np.linspace(df['DISTANCE (MI)'].min(),df['DISTANCE (MI)'].max(),200)
                pr2=lt.predict_proba(st2.transform(xr.reshape(-1,1)))[:,1]
                fig.add_trace(go.Scatter(x=xr,y=pr2,mode='lines',line=dict(color=ORANGE,width=3),name='Logistic fit'),row=1,col=2)
            jy=df['WON']+np.random.normal(0,0.03,len(df))
            fig.add_trace(go.Scatter(x=df['DISTANCE (MI)'],y=jy,mode='markers',
                marker=dict(size=4,color=[GREEN if w else RED for w in df['WON']],opacity=0.2),
                showlegend=False,hoverinfo='skip'),row=1,col=2)
            fig.update_layout(**plotly_layout(height=420,showlegend=True))
            fig.update_xaxes(title_text="Distance (mi)",row=1,col=1); fig.update_yaxes(title_text="Win Rate (%)",row=1,col=1)
            st.plotly_chart(fig, use_container_width=True)

    with tab_rnd:
        st.subheader("Round-by-Round Analysis")
        st.caption("Uses year/seed range from sidebar.")
        df=team_data[(team_data['YEAR'].between(*g_years))&(team_data['SEED'].between(*g_seeds))]
        rnds=[64,32,16,8,4,2]; rlb=['R64','R32','S16','E8','F4','Champ']; res=[]
        for r in rnds:
            rd=df[df['CURRENT ROUND']==r]
            if len(rd)>10:
                cr,p=stats.pointbiserialr(rd['WON'],rd['DISTANCE (MI)']); md=rd['DISTANCE (MI)'].median()
                cw=rd[rd['DISTANCE (MI)']<=md]['WON'].mean(); fw=rd[rd['DISTANCE (MI)']>md]['WON'].mean()
                res.append(dict(rnd=r,label=rlb[rnds.index(r)],r=cr,p=p,cw=cw,fw=fw,n=len(rd),sig=p<0.05))
        if not res: st.warning("Not enough data.")
        else:
            rdf=pd.DataFrame(res)
            fig=make_subplots(1,3,subplot_titles=["Close vs Far Win Rate","Correlation by Round","Sample Size"])
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['cw']*100,name='Closer',marker_color=GREEN),row=1,col=1)
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['fw']*100,name='Farther',marker_color=RED),row=1,col=1)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
            bc2=[GREEN if r<0 else RED for r in rdf['r']]; bo=[0.9 if s else 0.35 for s in rdf['sig']]
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['r'],showlegend=False,
                marker=dict(color=bc2,opacity=bo),text=[f"{v:+.3f}" for v in rdf['r']],textposition='outside'),row=1,col=2)
            fig.add_hline(y=0,line_color=MUTED,row=1,col=2)
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['n'],showlegend=False,marker_color=BLUE,
                text=[str(n) for n in rdf['n']],textposition='outside'),row=1,col=3)
            fig.update_layout(**plotly_layout(height=420,barmode='group'))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Solid bars = significant (p<0.05). Faded = not significant.")

    with tab_seed:
        st.subheader("Seed Deep Dive")
        ss=st.multiselect("Seeds", [f'Seed {i}' for i in range(1,17)],
            default=[f'Seed {i}' for i in range(g_seeds[0],min(g_seeds[1]+1,9))], key="ss")
        if not ss: st.info("Select at least one seed.")
        else:
            sn=[int(s.replace('Seed ','')) for s in ss]
            df=team_data[(team_data['YEAR'].between(*g_years))&(team_data['SEED'].isin(sn))]
            if len(df)<20: st.warning("Not enough data.")
            else:
                res=[]
                for s in sorted(sn):
                    sd=df[df['SEED']==s]
                    if len(sd)>=10:
                        md=sd['DISTANCE (MI)'].median()
                        cw=sd[sd['DISTANCE (MI)']<=md]['WON'].mean(); fw=sd[sd['DISTANCE (MI)']>md]['WON'].mean()
                        res.append(dict(seed=s,cw=cw,fw=fw,diff=cw-fw,n=len(sd)))
                if not res: st.warning("Min 10 obs per seed needed.")
                else:
                    sdf=pd.DataFrame(res)
                    fig=make_subplots(1,2,subplot_titles=["Win Rate by Proximity","Close − Far Advantage"])
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['cw']*100,name='Close',marker_color=GREEN),row=1,col=1)
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['fw']*100,name='Far',marker_color=RED),row=1,col=1)
                    fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
                    dc=[GREEN if d>=0 else RED for d in sdf['diff']]
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['diff']*100,marker_color=dc,
                        showlegend=False,text=[f"{d*100:+.1f}pp" for d in sdf['diff']],textposition='outside'),row=1,col=2)
                    fig.add_hline(y=0,line_color=MUTED,row=1,col=2)
                    fig.update_layout(**plotly_layout(height=420,barmode='group'))
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"**{(sdf['diff']>0).sum()} of {len(sdf)}** seeds show a proximity advantage.")

# ═══════════════════════════════════════════════════════════════════════════
#  MASTER 3: CONFERENCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
with master_conf:
    st.markdown('<div class="landing-card"><h4>🏛️ Conference Intelligence</h4>'
        '<p>Compare conferences side by side across efficiency, talent, SOS, and the four factors. '
        'Track how conference power has shifted over the years.</p></div>', unsafe_allow_html=True)
    st.subheader("Conference Strength Impact")
    c1,c2=st.columns([1,2])
    with c1: cy=st.selectbox("Season",sorted(conf_stats['YEAR'].unique(),reverse=True),key="cy")
    with c2:
        ac=sorted(conf_stats[conf_stats['YEAR']==cy]['CONF'].unique())
        md=[c for c in ['SEC','B10','B12','ACC','BE','MWC'] if c in ac]
        sc=st.multiselect("Conferences",ac,default=md or ac[:4],key="cs")
    if not sc: st.info("Select at least one conference.")
    else:
        csy=conf_stats[(conf_stats['YEAR']==cy)&(conf_stats['CONF'].isin(sc))].sort_values('BADJ EM',ascending=False)
        st.markdown("#### Overview")
        ov=csy[['CONF','BADJ EM','BADJ O','BADJ D','BARTHAG','TALENT','ELITE SOS','WAB']].copy()
        ov.columns=['Conf','Adj EM','Adj Off','Adj Def','BARTHAG','Talent','Elite SOS','WAB']
        st.dataframe(ov.style.format({'Adj EM':'{:.1f}','Adj Off':'{:.1f}','Adj Def':'{:.1f}',
            'BARTHAG':'{:.3f}','Talent':'{:.1f}','Elite SOS':'{:.1f}','WAB':'{:.1f}'}
        ).background_gradient(subset=['Adj EM'],cmap='RdYlGn').background_gradient(subset=['WAB'],cmap='RdYlGn'),
        use_container_width=True,hide_index=True)

        fig=make_subplots(1,3,subplot_titles=["Conference Strength","Offense vs Defense","Talent vs WAB"])
        ec=[GREEN if v>=10 else (ORANGE if v>=5 else (BLUE if v>=0 else RED)) for v in csy['BADJ EM']]
        fig.add_trace(go.Bar(x=csy['CONF'],y=csy['BADJ EM'],marker_color=ec,
            text=[f"{v:.1f}" for v in csy['BADJ EM']],textposition='outside',showlegend=False),row=1,col=1)
        fig.add_hline(y=0,line_color=MUTED,row=1,col=1)
        for _,r in csy.iterrows():
            fig.add_trace(go.Scatter(x=[r['BADJ O']],y=[r['BADJ D']],mode='markers+text',
                marker=dict(size=14,color=ORANGE),text=[r['CONF']],textposition='top center',
                textfont=dict(size=11,color=LIGHT),showlegend=False),row=1,col=2)
        fig.update_yaxes(autorange='reversed',title_text="Adj Def (↓ better)",row=1,col=2)
        for _,r in csy.iterrows():
            fig.add_trace(go.Scatter(x=[r['TALENT']],y=[r['WAB']],mode='markers+text',
                marker=dict(size=14,color=GREEN if r['WAB']>=0 else RED),text=[r['CONF']],
                textposition='top center',textfont=dict(size=11,color=LIGHT),showlegend=False),row=1,col=3)
        fig.add_hline(y=0,line_dash="dash",line_color=MUTED,row=1,col=3)
        fig.update_layout(**plotly_layout(height=420))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Four Factors")
        fig=make_subplots(1,4,subplot_titles=["Eff FG%","Turnover%","Rebounding","FT Rate"])
        for i,(o,d) in enumerate([('EFG%','EFGD%'),('TOV%','TOV%D'),('OREB%','DREB%'),('FTR','FTRD')],1):
            fig.add_trace(go.Bar(x=csy['CONF'],y=csy[o],name='Off' if i==1 else None,
                marker_color=GREEN,showlegend=(i==1)),row=1,col=i)
            fig.add_trace(go.Bar(x=csy['CONF'],y=csy[d],name='Def' if i==1 else None,
                marker_color=RED,showlegend=(i==1)),row=1,col=i)
        fig.update_layout(**plotly_layout(height=350,barmode='group'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Historical Trends")
        ch=conf_stats[conf_stats['CONF'].isin(sc)]; cc=[ORANGE,GREEN,BLUE,CYAN,PURPLE,RED,'#ec4899','#84cc16']
        fig=make_subplots(1,3,subplot_titles=["Adj EM","Talent","WAB"])
        for i,cf in enumerate(sc):
            cd=ch[ch['CONF']==cf].sort_values('YEAR'); cl=cc[i%len(cc)]
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['BADJ EM'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf),row=1,col=1)
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['TALENT'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf,showlegend=False),row=1,col=2)
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['WAB'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf,showlegend=False),row=1,col=3)
        fig.add_hline(y=0,line_dash="dash",line_color=MUTED,row=1,col=3)
        fig.update_layout(**plotly_layout(height=400))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  MASTER 4: MODELS & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with master_models:
    st.markdown('<div class="landing-card"><h4>🤖 Models & Methodology</h4>'
        '<p>Under the hood, this app uses 5 logistic regression models of increasing complexity. '
        'Compare their predictive power (AUC-ROC), examine feature importance, and understand '
        'what drives tournament outcomes.</p></div>', unsafe_allow_html=True)
    st.subheader("Model Comparison")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Dist Only",f"{cv1.mean():.3f}"); m2.metric("Dist+Seed",f"{cv2.mean():.3f}")
    m3.metric("D+S+TZ",f"{cv3.mean():.3f}"); m4.metric("Enhanced",f"{cv4.mean():.3f}")
    m5.metric("Full",f"{cv5.mean():.3f}")

    fig=make_subplots(1,2,subplot_titles=["ROC Curves","Model AUC (5-fold CV)"])
    for Xs,md,yt,nm,cl,ds in [(X1s,lr1,y_all,f'Dist ({cv1.mean():.3f})',MUTED,'dot'),
        (X2s,lr2,y_all,f'D+S ({cv2.mean():.3f})',ORANGE,'solid'),
        (X3s,lr3,y_all,f'D+S+TZ ({cv3.mean():.3f})',CYAN,'dash'),
        (X4s,lr4,y4,f'Enh ({cv4.mean():.3f})',GREEN,'solid'),
        (X5s,lr5,y5,f'Full ({cv5.mean():.3f})',PURPLE,'solid')]:
        yp=md.predict_proba(Xs)[:,1]; fpr,tpr,_=roc_curve(yt,yp)
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',line=dict(color=cl,width=2.5,dash=ds),name=nm),row=1,col=1)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color=MUTED,dash='dot',width=1),name='Random'),row=1,col=1)
    nms=['Dist','D+S','D+S+TZ','Enh','Full']; aucs=[cv1.mean(),cv2.mean(),cv3.mean(),cv4.mean(),cv5.mean()]
    sds=[cv1.std(),cv2.std(),cv3.std(),cv4.std(),cv5.std()]; bcs=[MUTED,ORANGE,CYAN,GREEN,PURPLE]
    fig.add_trace(go.Bar(x=nms,y=aucs,marker_color=bcs,error_y=dict(type='data',array=sds,visible=True,color=MUTED),
        text=[f"{a:.3f}" for a in aucs],textposition='outside',showlegend=False),row=1,col=2)
    fig.add_hline(y=0.5,line_dash="dash",line_color=MUTED,row=1,col=2)
    fig.update_layout(**plotly_layout(height=420))
    fig.update_yaxes(range=[0.45,max(aucs)+0.08],row=1,col=2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature Importance")
    fig=make_subplots(1,2,subplot_titles=[f"Enhanced (AUC={cv4.mean():.3f})",f"Full (AUC={cv5.mean():.3f})"])
    ft=['Distance','Seed','TR Rating','SOS']; c4=list(lr4.coef_[0]); si=np.argsort(np.abs(c4))
    fig.add_trace(go.Bar(y=[ft[i] for i in si],x=[c4[i] for i in si],orientation='h',
        marker_color=[RED if c<0 else GREEN for c in [c4[i] for i in si]],
        text=[f"{c4[i]:+.3f}" for i in si],textposition='outside',showlegend=False),row=1,col=1)
    ff=['Distance','Seed','TR Rating','SOS','KenPom EM','EM Δ','Top-25 W']; c5=list(lr5.coef_[0]); si=np.argsort(np.abs(c5))
    fig.add_trace(go.Bar(y=[ff[i] for i in si],x=[c5[i] for i in si],orientation='h',
        marker_color=[RED if c<0 else GREEN for c in [c5[i] for i in si]],
        text=[f"{c5[i]:+.3f}" for i in si],textposition='outside',showlegend=False),row=1,col=2)
    fig.add_vline(x=0,line_color=MUTED,row=1,col=1); fig.add_vline(x=0,line_color=MUTED,row=1,col=2)
    fig.update_layout(**plotly_layout(height=380))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model Summary")
    ms=pd.DataFrame([
        dict(Model='1. Distance',Features='Distance',AUC=cv1.mean(),Std=cv1.std(),Obs=len(y_all),Years='2008–2025'),
        dict(Model='2. Dist+Seed',Features='Distance, Seed',AUC=cv2.mean(),Std=cv2.std(),Obs=len(y_all),Years='2008–2025'),
        dict(Model='3. D+S+TZ',Features='Dist, Seed, TZ',AUC=cv3.mean(),Std=cv3.std(),Obs=len(y_all),Years='2008–2025'),
        dict(Model='4. Enhanced',Features='Dist, Seed, TR, SOS',AUC=cv4.mean(),Std=cv4.std(),Obs=len(y4),Years='2008–2025'),
        dict(Model='5. Full',Features='Dist, Seed, TR, SOS, KP, EMΔ, T25W',AUC=cv5.mean(),Std=cv5.std(),Obs=len(y5),Years='2012–2025')])
    st.dataframe(ms.style.format({'AUC':'{:.4f}','Std':'{:.4f}','Obs':'{:,}'}).background_gradient(subset=['AUC'],cmap='RdYlGn'),
        use_container_width=True,hide_index=True)
    best='Full' if cv5.mean()>cv4.mean() else 'Enhanced'
    imp=max(cv4.mean(),cv5.mean())-cv2.mean()
    st.info(f"**Key insight:** The **{best}** model achieves the highest AUC, a **{imp:.3f}** improvement "
            f"over Dist+Seed. KenPom efficiency margin is the single strongest predictor.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("NCAA Tournament Analytics Hub · Data: 2008–2025 · Built with Streamlit + Plotly + scikit-learn")
