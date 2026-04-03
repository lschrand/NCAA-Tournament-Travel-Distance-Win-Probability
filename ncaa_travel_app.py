"""
🏀 NCAA Tournament Analytics Hub
Usage: pip install streamlit pandas numpy scipy scikit-learn plotly
       streamlit run ncaa_travel_app.py
Files needed: Tournament Locations.csv, Tournament Matchups.csv, TeamRankings.csv,
              KenPom Preseason.csv, Conference Stats.csv, Shooting Splits.csv
"""
import streamlit as st, pandas as pd, numpy as np, warnings, os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
import plotly.graph_objects as go, plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NCAA Tournament Analytics Hub", page_icon="🏀",
                    layout="wide", initial_sidebar_state="expanded")

# ── Colors & Plotly helper ────────────────────────────────────────────────
ORANGE,GREEN,RED,BLUE,CYAN='#f97316','#22c55e','#ef4444','#3b82f6','#06b6d4'
MUTED,LIGHT,PURPLE,BG,CARD='#64748b','#e2e8f0','#a855f7','#0f1117','#161b22'
GOLD='#fbbf24'
def plotly_layout(**kw):
    b=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=LIGHT,size=12),xaxis=dict(gridcolor='#1e2530',zerolinecolor='#2a2f3a'),
        yaxis=dict(gridcolor='#1e2530',zerolinecolor='#2a2f3a'),margin=dict(l=60,r=30,t=50,b=50),
        hoverlabel=dict(bgcolor=CARD,font_size=12,font_color=LIGHT),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=10)))
    b.update(kw); return b
def bottom_line(text):
    st.markdown(f'<div style="background:rgba(251,191,36,0.10);border-left:3px solid {GOLD};'
        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:8px 0 16px 0;">'
        f'<p style="margin:0;color:#e2e8f0 !important;font-size:0.9rem;">💡 <strong>Bottom line:</strong> {text}</p>'
        f'</div>', unsafe_allow_html=True)
def guide_box(text):
    st.markdown(f'<div class="chart-guide"><p>{text}</p></div>', unsafe_allow_html=True)
def read_box(text):
    st.markdown(f'<div class="how-to-read"><p>{text}</p></div>', unsafe_allow_html=True)
def landing(icon, title, desc):
    st.markdown(f'<div class="landing-card"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background-color:#0f1117}.block-container{max-width:1250px;padding-top:1rem}
h1,h2,h3,h4,h5,h6,p,li,span,label,.stMarkdown{color:#e2e8f0 !important}
section[data-testid="stSidebar"]{background-color:#161b22}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#f97316 !important}
.stTabs [data-baseweb="tab-list"]{gap:4px;background-color:#161b22;border-radius:10px;padding:6px}
.stTabs [data-baseweb="tab"]{background-color:transparent;color:#94a3b8 !important;border-radius:8px;padding:10px 20px;font-weight:600}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#f97316,#ea580c) !important;color:#fff !important}
.stSlider>div>div>div{color:#e2e8f0 !important}
div[data-testid="stMetric"]{background-color:#161b22;border:1px solid #2a2f3a;border-radius:8px;padding:12px 16px}
div[data-testid="stMetric"] label{color:#94a3b8 !important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f97316 !important}
.stButton>button{background-color:#f97316 !important;color:white !important;border:none;border-radius:6px;font-weight:600}
.stButton>button:hover{background-color:#ea580c !important}
.stSelectbox>div>div,.stMultiSelect>div>div{background-color:#161b22;border-color:#2a2f3a}
.landing-card{background:linear-gradient(135deg,#161b22 0%,#1a1f2e 100%);border:1px solid #2a2f3a;border-radius:10px;padding:16px 20px;margin-bottom:16px}
.landing-card h4{margin:0 0 6px 0;color:#f97316 !important}.landing-card p{margin:0;color:#94a3b8 !important;font-size:0.9rem}
.chart-guide{background-color:rgba(59,130,246,0.08);border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;padding:10px 14px;margin:8px 0 12px 0}
.chart-guide p{margin:0;color:#94a3b8 !important;font-size:0.85rem;line-height:1.5}.chart-guide strong{color:#e2e8f0 !important}
.how-to-read{background-color:rgba(34,197,94,0.08);border-left:3px solid #22c55e;border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0 12px 0}
.how-to-read p{margin:0;color:#94a3b8 !important;font-size:0.85rem;line-height:1.5}.how-to-read strong{color:#e2e8f0 !important}
[data-testid="stImage"] img{border-radius:50%;border:3px solid #f97316;box-shadow:0 2px 12px rgba(249,115,22,0.2)}
.nav-card{background:#161b22;border:1px solid #2a2f3a;border-radius:10px;padding:18px;text-align:center;min-height:130px}
.nav-card h5{color:#f97316 !important;margin:0 0 8px 0}.nav-card p{color:#94a3b8 !important;font-size:0.82rem;margin:0}
</style>""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d='.'
    for c in ['.','data',os.path.dirname(__file__) if '__file__' in dir() else '.']:
        if os.path.exists(os.path.join(c,'Tournament Locations.csv')): d=c; break
    loc=pd.read_csv(os.path.join(d,'Tournament Locations.csv'))
    mat=pd.read_csv(os.path.join(d,'Tournament Matchups.csv'))
    mg=mat.merge(loc[['YEAR','BY YEAR NO','TEAM','DISTANCE (MI)','DISTANCE (KM)','TIME ZONES CROSSED',
        'DIRECTION','COLLEGE LATITUDE','COLLEGE LONGITUDE','GAME LATITUDE','GAME LONGITUDE',
        'GAME CITY','GAME STATE']],on=['YEAR','BY YEAR NO','TEAM'],how='inner')
    mg['WON']=(mg['ROUND']<mg['CURRENT ROUND']).astype(int)
    gl=[]
    for (yr,cr),g in mg.groupby(['YEAR','CURRENT ROUND']):
        g=g.sort_values('BY YEAR NO',ascending=False).reset_index(drop=True)
        for i in range(0,len(g)-1,2):
            t1,t2=g.iloc[i],g.iloc[i+1]
            gl.append(dict(YEAR=yr,CURRENT_ROUND=cr,TEAM1=t1['TEAM'],SEED1=t1['SEED'],
                DIST1=t1['DISTANCE (MI)'],WON1=t1['WON'],SCORE1=t1['SCORE'],
                TEAM2=t2['TEAM'],SEED2=t2['SEED'],DIST2=t2['DISTANCE (MI)'],
                WON2=t2['WON'],SCORE2=t2['SCORE'],GAME_CITY=t1['GAME CITY'],GAME_STATE=t1['GAME STATE']))
    gdf=pd.DataFrame(gl)
    td=mg.copy(); ya=td['WON'].values
    sc1=StandardScaler();X1=sc1.fit_transform(td[['DISTANCE (MI)']].values)
    lr1=LogisticRegression(random_state=42).fit(X1,ya);cv1=cross_val_score(lr1,X1,ya,cv=5,scoring='roc_auc')
    sc2=StandardScaler();X2=sc2.fit_transform(td[['DISTANCE (MI)','SEED']].values)
    lr2=LogisticRegression(random_state=42).fit(X2,ya);cv2=cross_val_score(lr2,X2,ya,cv=5,scoring='roc_auc')
    sc3=StandardScaler();X3=sc3.fit_transform(td[['DISTANCE (MI)','SEED','TIME ZONES CROSSED']].values)
    lr3=LogisticRegression(random_state=42).fit(X3,ya);cv3=cross_val_score(lr3,X3,ya,cv=5,scoring='roc_auc')
    trk=pd.read_csv(os.path.join(d,'TeamRankings.csv'))
    kp=pd.read_csv(os.path.join(d,'KenPom Preseason.csv'))
    cs=pd.read_csv(os.path.join(d,'Conference Stats.csv'))
    ss=pd.read_csv(os.path.join(d,'Shooting Splits.csv'))
    enh=td.copy()
    enh=enh.merge(trk[['YEAR','TEAM NO','TR RANK','TR RATING','SOS RANK','SOS RATING',
        'V 1-25 WINS','V 1-25 LOSS','V 26-50 WINS','V 26-50 LOSS',
        'CONSISTENCY RANK','CONSISTENCY TR RATING','LUCK RANK','LUCK RATING']],on=['YEAR','TEAM NO'],how='left')
    enh=enh.merge(kp[['YEAR','TEAM NO','PRESEASON KADJ EM','PRESEASON KADJ EM RANK',
        'PRESEASON KADJ O','PRESEASON KADJ D','KADJ EM RANK CHANGE','KADJ EM CHANGE']],on=['YEAR','TEAM NO'],how='left')
    f4=['DISTANCE (MI)','SEED','TR RATING','SOS RATING']
    d4=enh.dropna(subset=f4);sc4=StandardScaler();X4=sc4.fit_transform(d4[f4].values);y4=d4['WON'].values
    lr4=LogisticRegression(random_state=42).fit(X4,y4);cv4=cross_val_score(lr4,X4,y4,cv=5,scoring='roc_auc')
    f5=['DISTANCE (MI)','SEED','TR RATING','SOS RATING','PRESEASON KADJ EM','KADJ EM CHANGE','V 1-25 WINS']
    d5=enh.dropna(subset=f5);sc5=StandardScaler();X5=sc5.fit_transform(d5[f5].values);y5=d5['WON'].values
    lr5=LogisticRegression(random_state=42).fit(X5,y5);cv5=cross_val_score(lr5,X5,y5,cv=5,scoring='roc_auc')
    return dict(td=td,gdf=gdf,ya=ya,sc1=sc1,X1=X1,lr1=lr1,cv1=cv1,sc2=sc2,X2=X2,lr2=lr2,cv2=cv2,
        sc3=sc3,X3=X3,lr3=lr3,cv3=cv3,teams=sorted(td['TEAM'].unique()),years=sorted(td['YEAR'].unique()),
        enh=enh,trk=trk,kp=kp,cs=cs,ss=ss,f4=f4,sc4=sc4,X4=X4,lr4=lr4,cv4=cv4,y4=y4,
        f5=f5,sc5=sc5,X5=X5,lr5=lr5,cv5=cv5,y5=y5)
try:
    D=load_data()
    td,gdf,ya=D['td'],D['gdf'],D['ya']
    sc1,X1s,lr1,cv1=D['sc1'],D['X1'],D['lr1'],D['cv1']
    sc2,X2s,lr2,cv2=D['sc2'],D['X2'],D['lr2'],D['cv2']
    sc3,X3s,lr3,cv3=D['sc3'],D['X3'],D['lr3'],D['cv3']
    teams,years=D['teams'],D['years']
    enh,cs,ss=D['enh'],D['cs'],D['ss']
    f4,sc4,X4s,lr4,cv4,y4=D['f4'],D['sc4'],D['X4'],D['lr4'],D['cv4'],D['y4']
    f5,sc5,X5s,lr5,cv5,y5=D['f5'],D['sc5'],D['X5'],D['lr5'],D['cv5'],D['y5']
except Exception as e:
    st.error(f"**Error:** {e}"); st.stop()
rmap={64:'Round of 64',32:'Round of 32',16:'Sweet 16',8:'Elite 8',4:'Final 4',2:'Championship'}
# Precompute model accuracies as plain Python floats (avoids numpy formatting issues)
acc1=cv1.mean().item()*100; acc2=cv2.mean().item()*100; acc3=cv3.mean().item()*100
acc4=cv4.mean().item()*100; acc5=cv5.mean().item()*100
std1=cv1.std().item()*100; std2=cv2.std().item()*100; std3=cv3.std().item()*100
std4=cv4.std().item()*100; std5=cv5.std().item()*100

def best_model_predict(row_a, row_b):
    """Return (team_a_prob, team_b_prob, model_name, accuracy%) using best available model."""
    try:
        va=[row_a[f] for f in f5]; vb=[row_b[f] for f in f5]
        if all(pd.notna(v) for v in va+vb):
            pa=lr5.predict_proba(sc5.transform([va]))[0][1]
            pb=lr5.predict_proba(sc5.transform([vb]))[0][1]
            return pa/(pa+pb),pb/(pa+pb),'Full model',acc5
    except: pass
    try:
        va=[row_a[f] for f in f4]; vb=[row_b[f] for f in f4]
        if all(pd.notna(v) for v in va+vb):
            pa=lr4.predict_proba(sc4.transform([va]))[0][1]
            pb=lr4.predict_proba(sc4.transform([vb]))[0][1]
            return pa/(pa+pb),pb/(pa+pb),'Enhanced model',acc4
    except: pass
    pa=lr2.predict_proba(sc2.transform([[row_a['DISTANCE (MI)'],row_a['SEED']]]))[0][1]
    pb=lr2.predict_proba(sc2.transform([[row_b['DISTANCE (MI)'],row_b['SEED']]]))[0][1]
    return pa/(pa+pb),pb/(pa+pb),'Simple model',acc2

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏀 Filters")
    st.caption("Adjust these to filter the Travel & Distance tabs.")
    g_years=st.slider("Years",int(min(years)),int(max(years)),(int(min(years)),int(max(years))),key="gy")
    g_seeds=st.slider("Seeds",1,16,(1,16),key="gs")
    g_fav=st.selectbox("Favorite Team",["—"]+teams,key="gf")
    st.markdown("---")
    _f=td[(td['YEAR'].between(*g_years))&(td['SEED'].between(*g_seeds))]
    st.metric("Games in Filter",f"{len(_f):,}")
    st.metric("Win Rate",f"{_f['WON'].mean():.1%}" if len(_f) else "—")
    if g_fav!="—":
        _fv=_f[_f['TEAM']==g_fav]
        if len(_fv):
            st.markdown(f"### 🌟 {g_fav}")
            st.metric("Games",f"{len(_fv)}"); st.metric("Win Rate",f"{_fv['WON'].mean():.1%}")

# ── Header ────────────────────────────────────────────────────────────────
hl,hr=st.columns([4,1])
with hl:
    st.markdown("# 🏀 NCAA Tournament Analytics Hub")
    st.markdown(f"Your bracket's secret weapon — powered by **{len(td):,} games** of real tournament data "
        f"across **{len(years)} seasons** ({min(years)}–{max(years)})")
with hr:
    try:
        _p=os.path.join(os.path.dirname(__file__),'teams_pic.jpg') if '__file__' in dir() else 'teams_pic.jpg'
        if os.path.exists(_p): st.image(_p,width=110)
        elif os.path.exists('teams_pic.jpg'): st.image('teams_pic.jpg',width=110)
    except: pass
    st.markdown('<p style="text-align:center;color:#94a3b8;font-size:0.8rem;margin:-8px 0 0 0;line-height:1.4;">'
        '<strong style="color:#f97316;">Levi Schrandt</strong><br>Purdue University</p>',
        unsafe_allow_html=True)
st.markdown("---")

# ── Master tabs ───────────────────────────────────────────────────────────
m_home,m_pred,m_travel,m_shot,m_conf,m_models=st.tabs([
    "🏠 Home","🎯 Predict Games","📊 Travel Analysis",
    "🏀 Shot Profiles","🏛️ Conferences","🤖 Under the Hood"])

# ═══════════════════════════════════════════════════════════════════════════
#  HOME TAB
# ═══════════════════════════════════════════════════════════════════════════
with m_home:
    st.markdown("### Welcome! Here's how to use this app.")
    guide_box(
        'This app helps you <strong>make smarter March Madness picks</strong> using real data. '
        'Below is a <strong>Quick Predict</strong> tool — just pick two teams and get an instant answer. '
        'Or explore the tabs above for deeper analysis.')

    # ── Quick Predict ─────────────────────────────────────────────────
    st.markdown("### ⚡ Quick Predict")
    st.markdown("Pick two teams from any tournament year and instantly see who our best model favors.")
    qc1,qc2,qc3=st.columns([1,2,2])
    with qc1: qy=st.selectbox("Year",sorted(enh['YEAR'].unique(),reverse=True),key="qy")
    qt=sorted(enh[enh['YEAR']==qy]['TEAM'].unique())
    with qc2: qa=st.selectbox("Team A",qt,0,key="qa")
    with qc3:
        default_b=min(1,len(qt)-1)
        qb=st.selectbox("Team B",qt,default_b,key="qb")
    if qa==qb:
        st.info("Pick two different teams to see a prediction.")
    else:
        ra=enh[(enh['YEAR']==qy)&(enh['TEAM']==qa)].iloc[0]
        rb=enh[(enh['YEAR']==qy)&(enh['TEAM']==qb)].iloc[0]
        pa,pb,mname,macc=best_model_predict(ra,rb)
        winner=qa if pa>pb else qb
        loser=qb if winner==qa else qa
        conf=max(pa,pb)
        # Big result display
        rc1,rc2,rc3=st.columns([2,1,2])
        with rc1:
            st.metric(qa,f"{pa:.0%}")
        with rc2:
            st.markdown(f'<div style="text-align:center;padding-top:10px;">'
                f'<span style="color:{MUTED};font-size:1.5rem;">vs</span></div>',unsafe_allow_html=True)
        with rc3:
            st.metric(qb,f"{pb:.0%}")

        # Plain English result
        if conf>=0.65:
            verdict=f"**{winner}** is the clear favorite over {loser}."
        elif conf>=0.55:
            verdict=f"**{winner}** has an edge, but this could go either way."
        else:
            verdict=f"This is essentially a **coin flip** — pick with your gut."
        st.markdown(f'<div style="background:#161b22;border:1px solid #2a2f3a;border-radius:10px;'
            f'padding:16px 20px;text-align:center;margin:8px 0;">'
            f'<p style="font-size:1.1rem;margin:0;color:#e2e8f0 !important;">{verdict}</p>'
            f'<p style="font-size:0.8rem;margin:6px 0 0 0;color:#64748b !important;">'
            f'Based on our {mname} ({macc:.0f}% accurate) · Seed #{int(ra["SEED"])} vs #{int(rb["SEED"])}</p>'
            f'</div>',unsafe_allow_html=True)

    # ── Navigation Guide ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ What would you like to do?")
    nc1,nc2,nc3=st.columns(3)
    with nc1:
        st.markdown('<div class="nav-card"><h5>🎯 Predict a Game</h5>'
            '<p>Use our 5 models to predict any matchup. Compare simple vs advanced predictions '
            'and see where the models agree or disagree.</p></div>',unsafe_allow_html=True)
    with nc2:
        st.markdown('<div class="nav-card"><h5>🏀 Research a Team</h5>'
            '<p>Look up a team\'s shooting DNA, travel history, and how they compare to '
            'past champions. Great for spotting sleepers.</p></div>',unsafe_allow_html=True)
    with nc3:
        st.markdown('<div class="nav-card"><h5>📊 Find an Edge</h5>'
            '<p>Explore which teams travel too far, which conferences are overrated, and '
            'which seeds historically over- or under-perform.</p></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 At a Glance")
    g1,g2,g3,g4=st.columns(4)
    g1.metric("Seasons Covered",f"{len(years)}")
    g2.metric("Total Games",f"{len(td):,}")
    g3.metric("Best Model Accuracy",f"{max(acc4,acc5):.0f}%")
    closer_wr=td[td['DISTANCE (MI)']<=td['DISTANCE (MI)'].median()]['WON'].mean()
    g4.metric("Close Teams Win Rate",f"{closer_wr:.1%}")
    bottom_line("Teams that play closer to home win more often, but seed and team quality "
                "matter far more than distance. Use the Predict Games tab for the full picture.")

# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT GAMES
# ═══════════════════════════════════════════════════════════════════════════
with m_pred:
    landing("🎯","Predict Games",
        "Compare two teams using our prediction models. Start with the simple predictor "
        "or jump to the Enhanced Predictor for our most accurate analysis.")
    tab_pred,tab_enh,tab_h2h,tab_team=st.tabs(["🎯 Win Predictor","🔬 Enhanced Predictor",
        "⚔️ Head-to-Head","🏫 Team Lookup"])

    with tab_pred:
        st.subheader("Win Probability Predictor")
        guide_box('How much does <strong>travel distance</strong> affect a team\'s chances? '
            'Adjust the sliders to see — for example, try changing a 4-seed from 200 miles to 1,500 miles.')
        c1,c2,c3=st.columns(3)
        with c1: dist=st.slider("Distance (mi)",0,2500,500,25,key="pd")
        with c2: seed=st.slider("Seed",1,16,4,key="ps")
        with c3: tr=st.selectbox("Round",list(rmap.values()),key="pr")
        rv={v:k for k,v in rmap.items()}[tr]
        prob=lr2.predict_proba(sc2.transform([[dist,seed]]))[0][1]
        base=lr2.predict_proba(sc2.transform([[0,seed]]))[0][1]
        hist=td[(td['CURRENT ROUND']==rv)&(td['SEED']==seed)]
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Win Chance",f"{prob:.0%}")
        m2.metric("If They Played at Home",f"{base:.0%}")
        m3.metric("Distance Cost",f"{(prob-base)*100:+.1f} pts")
        if len(hist): m4.metric(f"Seed {seed} Historical WR",f"{hist['WON'].mean():.0%}")
        read_box('📊 <strong>Left:</strong> Win probability gauge. <strong>Center:</strong> How win chance '
            'drops as distance increases (cyan dot = your selection). <strong>Right:</strong> '
            'How far past winners vs losers actually traveled for this seed/round.')
        fig=make_subplots(1,3,specs=[[{"type":"indicator"},{"type":"xy"},{"type":"xy"}]],
            subplot_titles=[f"Win Chance: {prob:.0%}",f"Seed {seed} — Distance Effect",
                f"Seed {seed} in {tr}"])
        c=GREEN if prob>=.55 else(ORANGE if prob>=.45 else RED)
        fig.add_trace(go.Indicator(mode="gauge+number",value=prob*100,
            number=dict(suffix="%",font=dict(size=36,color=c)),
            gauge=dict(axis=dict(range=[0,100],tickcolor=MUTED),bar=dict(color=c),bgcolor='rgba(0,0,0,0)',
                steps=[dict(range=[0,40],color='rgba(239,68,68,0.15)'),
                    dict(range=[40,60],color='rgba(249,115,22,0.15)'),
                    dict(range=[60,100],color='rgba(34,197,94,0.15)')])),row=1,col=1)
        dr=np.linspace(0,2500,200)
        pr=[lr2.predict_proba(sc2.transform([[d,seed]]))[0][1]*100 for d in dr]
        fig.add_trace(go.Scatter(x=dr,y=pr,mode='lines',line=dict(color=ORANGE,width=3),
            name='Win %',hovertemplate='%{x:.0f} mi → %{y:.1f}%'),row=1,col=2)
        fig.add_shape(type="line",x0=0,x1=2500,y0=50,y1=50,line=dict(dash="dash",color=MUTED),xref="x2",yref="y2")
        fig.add_shape(type="line",x0=dist,x1=dist,y0=0,y1=100,line=dict(color=CYAN,width=2),xref="x2",yref="y2")
        fig.add_trace(go.Scatter(x=[dist],y=[prob*100],mode='markers+text',marker=dict(size=12,color=CYAN),
            text=[f" {prob:.0%}"],textposition="middle right",textfont=dict(color=CYAN,size=13),
            showlegend=False),row=1,col=2)
        if len(hist)>5:
            fig.add_trace(go.Histogram(x=hist[hist['WON']==1]['DISTANCE (MI)'],name='Winners',
                marker_color=GREEN,opacity=0.6,histnorm='probability density'),row=1,col=3)
            fig.add_trace(go.Histogram(x=hist[hist['WON']==0]['DISTANCE (MI)'],name='Losers',
                marker_color=RED,opacity=0.5,histnorm='probability density'),row=1,col=3)
            fig.add_shape(type="line",x0=dist,x1=dist,y0=0,y1=1,line=dict(color=CYAN,width=2),
                xref="x3",yref="y3 domain")
        fig.update_layout(**plotly_layout(height=400,showlegend=True,barmode='overlay'))
        fig.update_xaxes(title_text="Distance (mi)",row=1,col=2)
        fig.update_yaxes(title_text="Win Chance (%)",row=1,col=2)
        st.plotly_chart(fig,use_container_width=True)
        penalty=abs(prob-base)*100
        if penalty<1: bottom_line("Distance barely matters for this seed — focus on matchup quality instead.")
        elif penalty<3: bottom_line(f"Travel costs this team about {penalty:.1f} percentage points. A small but real disadvantage.")
        else: bottom_line(f"Travel is a real factor here — {penalty:.1f} points of win probability lost to distance alone.")

    with tab_enh:
        st.subheader("Enhanced Game Predictor")
        guide_box('Our <strong>most accurate predictor</strong> — uses team power ratings, strength of schedule, '
            'and KenPom efficiency data (not just seed and distance). Pick two real tournament teams or enter custom stats.')
        mode=st.radio("Mode",["Pick Tournament Teams","Custom Input"],horizontal=True,key="em")
        if mode=="Pick Tournament Teams":
            c1,c2,c3=st.columns([1,2,2])
            with c1: ey=st.selectbox("Year",sorted(enh['YEAR'].unique(),reverse=True),key="ey")
            yt=sorted(enh[enh['YEAR']==ey]['TEAM'].unique())
            with c2: ta=st.selectbox("Team A",yt,0,key="ea")
            with c3: tb=st.selectbox("Team B",yt,min(1,len(yt)-1),key="eb")
            if ta==tb: st.info("Pick two different teams.")
            else:
                ra=enh[(enh['YEAR']==ey)&(enh['TEAM']==ta)].iloc[0]
                rb=enh[(enh['YEAR']==ey)&(enh['TEAM']==tb)].iloc[0]
                st.markdown("#### Team Profiles")
                ca,cb=st.columns(2)
                for col,r,l in[(ca,ra,"A"),(cb,rb,"B")]:
                    with col:
                        st.markdown(f"**Team {l}: {r['TEAM']}**")
                        p1,p2,p3=st.columns(3)
                        p1.metric("Seed",f"#{int(r['SEED'])}")
                        p2.metric("Power Rating",f"{r['TR RATING']:.1f}" if pd.notna(r.get('TR RATING')) else "N/A")
                        p3.metric("Schedule Strength",f"{r['SOS RATING']:.1f}" if pd.notna(r.get('SOS RATING')) else "N/A")
                preds=[]
                pa=lr2.predict_proba(sc2.transform([[ra['DISTANCE (MI)'],ra['SEED']]]))[0][1]
                pb=lr2.predict_proba(sc2.transform([[rb['DISTANCE (MI)'],rb['SEED']]]))[0][1]
                preds.append(("Basic",pa/(pa+pb),pb/(pa+pb),acc2/100))
                try:
                    va=[ra[f] for f in f4];vb=[rb[f] for f in f4]
                    if all(pd.notna(v) for v in va+vb):
                        pa=lr4.predict_proba(sc4.transform([va]))[0][1]
                        pb=lr4.predict_proba(sc4.transform([vb]))[0][1]
                        preds.append(("+ Power Ratings",pa/(pa+pb),pb/(pa+pb),acc4/100))
                except: pass
                try:
                    va=[ra[f] for f in f5];vb=[rb[f] for f in f5]
                    if all(pd.notna(v) for v in va+vb):
                        pa=lr5.predict_proba(sc5.transform([va]))[0][1]
                        pb=lr5.predict_proba(sc5.transform([vb]))[0][1]
                        preds.append(("+ KenPom (best)",pa/(pa+pb),pb/(pa+pb),acc5/100))
                except: pass
                st.markdown("#### Model Predictions")
                read_box('📊 Each bar shows win probability from a different model. '
                    '<strong>Basic</strong> = seed + distance only. '
                    '<strong>+ Power Ratings</strong> adds team quality. '
                    '<strong>+ KenPom</strong> is our most accurate. When they all agree, you can be confident.')
                fig=make_subplots(1,len(preds),subplot_titles=[f"{n} ({a*100:.0f}% acc.)" for n,_,_,a in preds])
                for i,(n,rra,rrb,a) in enumerate(preds,1):
                    fig.add_trace(go.Bar(y=[ta],x=[rra*100],orientation='h',marker_color=GREEN,
                        showlegend=(i==1),name=ta,text=[f"{rra:.0%}"],textposition='outside'),row=1,col=i)
                    fig.add_trace(go.Bar(y=[tb],x=[rrb*100],orientation='h',marker_color=RED,
                        showlegend=(i==1),name=tb,text=[f"{rrb:.0%}"],textposition='outside'),row=1,col=i)
                    fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=i)
                fig.update_layout(**plotly_layout(height=300));fig.update_xaxes(range=[0,105])
                st.plotly_chart(fig,use_container_width=True)
                ws=[ta if a>b else tb for _,a,b,_ in preds]
                best_p=preds[-1]; w=ta if best_p[1]>best_p[2] else tb; conf=max(best_p[1],best_p[2])
                if len(set(ws))==1:
                    bottom_line(f"All models agree — pick <strong>{ws[0]}</strong>. "
                        f"Our best model gives them a {conf:.0%} chance.")
                else:
                    bottom_line(f"The models disagree — this is a <strong>potential upset game</strong>. "
                        f"Our best model slightly favors {w} ({conf:.0%}), but tread carefully.")
        else:
            st.markdown("#### Enter Custom Team Stats")
            guide_box('Don\'t know these numbers? Check <strong>TeamRankings.com</strong> for TR/SOS ratings '
                'and <strong>KenPom.com</strong> for efficiency margins. Higher TR Rating = better team.')
            ca,cb=st.columns(2)
            with ca:
                st.markdown("**Team A**")
                csa=st.slider("Seed",1,16,3,key="csa");cda=st.slider("Dist (mi)",0,2500,200,25,key="cda")
                ctra=st.number_input("Power Rating (TR)",value=15.0,step=0.5,key="ctra")
                csosa=st.number_input("Schedule Strength (SOS)",value=8.0,step=0.5,key="csosa")
                ckema=st.number_input("KenPom Efficiency",value=20.0,step=0.5,key="ckema")
                ckchga=st.number_input("Season Improvement",value=2.0,step=0.5,key="ckchga")
                cv25a=st.number_input("Wins vs Top 25",value=5,step=1,key="cv25a")
            with cb:
                st.markdown("**Team B**")
                csb=st.slider("Seed",1,16,11,key="csb");cdb=st.slider("Dist (mi)",0,2500,800,25,key="cdb")
                ctrb=st.number_input("Power Rating (TR)",value=6.0,step=0.5,key="ctrb")
                csosb=st.number_input("Schedule Strength (SOS)",value=2.0,step=0.5,key="csosb")
                ckemb=st.number_input("KenPom Efficiency",value=10.0,step=0.5,key="ckemb")
                ckchgb=st.number_input("Season Improvement",value=5.0,step=0.5,key="ckchgb")
                cv25b=st.number_input("Wins vs Top 25",value=1,step=1,key="cv25b")
            mc=[]
            pa=lr2.predict_proba(sc2.transform([[cda,csa]]))[0][1]
            pb=lr2.predict_proba(sc2.transform([[cdb,csb]]))[0][1];mc.append(("Basic",pa/(pa+pb),pb/(pa+pb)))
            pa=lr4.predict_proba(sc4.transform([[cda,csa,ctra,csosa]]))[0][1]
            pb=lr4.predict_proba(sc4.transform([[cdb,csb,ctrb,csosb]]))[0][1];mc.append(("Enhanced",pa/(pa+pb),pb/(pa+pb)))
            pa=lr5.predict_proba(sc5.transform([[cda,csa,ctra,csosa,ckema,ckchga,cv25a]]))[0][1]
            pb=lr5.predict_proba(sc5.transform([[cdb,csb,ctrb,csosb,ckemb,ckchgb,cv25b]]))[0][1];mc.append(("Full",pa/(pa+pb),pb/(pa+pb)))
            fig=make_subplots(1,3,subplot_titles=[n for n,_,_ in mc])
            for i,(n,ra,rb) in enumerate(mc,1):
                fig.add_trace(go.Bar(y=[f'A (#{csa})'],x=[ra*100],orientation='h',marker_color=GREEN,
                    showlegend=False,text=[f"{ra:.0%}"],textposition='outside'),row=1,col=i)
                fig.add_trace(go.Bar(y=[f'B (#{csb})'],x=[rb*100],orientation='h',marker_color=RED,
                    showlegend=False,text=[f"{rb:.0%}"],textposition='outside'),row=1,col=i)
                fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=i)
            fig.update_layout(**plotly_layout(height=280));fig.update_xaxes(range=[0,105])
            st.plotly_chart(fig,use_container_width=True)
            best_ra=mc[-1][1]
            bottom_line(f"Our best model gives Team A a {best_ra:.0%} chance. "
                f"{'A solid pick.' if best_ra>=0.6 else 'Close game — could go either way.' if best_ra>=0.45 else 'Team B is the favorite here.'}")

    with tab_h2h:
        st.subheader("Head-to-Head Simulator")
        guide_box('Set custom <strong>seeds and distances</strong> for two hypothetical teams. '
            'The right chart shows how Team A\'s chances change if they had to travel farther or closer.')
        ca,cb=st.columns(2)
        with ca: st.markdown("**Team A**");sa=st.slider("Seed A",1,16,3,key="ha");da=st.slider("Dist A (mi)",0,2500,200,25,key="hda")
        with cb: st.markdown("**Team B**");sb=st.slider("Seed B",1,16,11,key="hb");db=st.slider("Dist B (mi)",0,2500,1200,25,key="hdb")
        pa=lr2.predict_proba(sc2.transform([[da,sa]]))[0][1];pb=lr2.predict_proba(sc2.transform([[db,sb]]))[0][1]
        ra,rb=pa/(pa+pb),pb/(pa+pb)
        m1,m2,m3=st.columns(3);m1.metric("Team A",f"{ra:.0%}");m2.metric("Team B",f"{rb:.0%}")
        m3.metric("Pick",f"{'A' if ra>rb else 'B'} by {abs(ra-rb):.0%}")
        fig=make_subplots(1,2,subplot_titles=["Head-to-Head","What if Team A Traveled More/Less?"])
        fig.add_trace(go.Bar(y=[f'A (#{sa})'],x=[ra*100],orientation='h',marker_color=GREEN,showlegend=False,text=[f"{ra:.0%}"],textposition='outside'),row=1,col=1)
        fig.add_trace(go.Bar(y=[f'B (#{sb})'],x=[rb*100],orientation='h',marker_color=RED,showlegend=False,text=[f"{rb:.0%}"],textposition='outside'),row=1,col=1)
        fig.add_vline(x=50,line_dash="dash",line_color=MUTED,row=1,col=1)
        dr=np.linspace(0,2500,100);pv=[lr2.predict_proba(sc2.transform([[d,sa]]))[0][1] for d in dr]
        rv2=[p/(p+pb)*100 for p in pv]
        fig.add_trace(go.Scatter(x=dr,y=rv2,mode='lines',line=dict(color=GREEN,width=3),showlegend=False),row=1,col=2)
        fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=2)
        fig.add_vline(x=da,line_color=CYAN,line_width=2,row=1,col=2)
        fig.update_layout(**plotly_layout(height=350));fig.update_xaxes(range=[0,105],row=1,col=1)
        fig.update_xaxes(title_text="Team A Distance (mi)",row=1,col=2)
        fig.update_yaxes(title_text="Team A Win Chance (%)",row=1,col=2)
        st.plotly_chart(fig,use_container_width=True)
        bottom_line(f"A #{sa}-seed traveling {da} miles beats a #{sb}-seed traveling {db} miles about {ra:.0%} of the time. "
            f"{'Distance is a factor here.' if abs(da-db)>500 else 'Distance is not a big factor in this matchup.'}")

    with tab_team:
        st.subheader("Team Lookup")
        guide_box('Search for any team to see their <strong>tournament travel history</strong>. '
            'Green dots = wins, red X\'s = losses. See if they perform better close to home.')
        dt=g_fav if g_fav!="—" else ('Duke' if 'Duke' in teams else teams[0])
        tn=st.selectbox("Team",teams,index=teams.index(dt) if dt in teams else 0,key="tl")
        df=td[td['TEAM']==tn]
        if not len(df): st.warning(f"No data for {tn}.")
        else:
            med=df['DISTANCE (MI)'].median();cl=df[df['DISTANCE (MI)']<=med];fa=df[df['DISTANCE (MI)']>med]
            yrs=sorted(df['YEAR'].unique())
            m1,m2,m3,m4=st.columns(4);m1.metric("Appearances",f"{len(yrs)}");m2.metric("Games",f"{len(df)}")
            m3.metric("Win Rate",f"{df['WON'].mean():.0%}");m4.metric("Avg Distance",f"{df['DISTANCE (MI)'].mean():.0f} mi")
            fig=make_subplots(1,3,subplot_titles=[f"{tn} Games","Close vs Far","Travel Distribution"])
            w=df[df['WON']==1];l=df[df['WON']==0]
            fig.add_trace(go.Scatter(x=w['DISTANCE (MI)'],y=w['YEAR'],mode='markers',
                marker=dict(size=10,color=GREEN),name=f'Wins ({len(w)})'),row=1,col=1)
            fig.add_trace(go.Scatter(x=l['DISTANCE (MI)'],y=l['YEAR'],mode='markers',
                marker=dict(size=10,color=RED,symbol='x'),name=f'Losses ({len(l)})'),row=1,col=1)
            cwr=cl['WON'].mean()*100 if len(cl) else 0;fwr=fa['WON'].mean()*100 if len(fa) else 0
            fig.add_trace(go.Bar(x=[f'Close (<{med:.0f}mi)',f'Far (>{med:.0f}mi)'],y=[cwr,fwr],
                marker_color=[GREEN,RED],text=[f"{cwr:.0f}%",f"{fwr:.0f}%"],textposition='outside',showlegend=False),row=1,col=2)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=2)
            fig.add_trace(go.Histogram(x=df['DISTANCE (MI)'],marker_color=BLUE,opacity=0.7,showlegend=False),row=1,col=3)
            fig.add_vline(x=df['DISTANCE (MI)'].mean(),line_dash="dash",line_color=ORANGE,line_width=2,row=1,col=3)
            fig.update_layout(**plotly_layout(height=400,showlegend=True))
            st.plotly_chart(fig,use_container_width=True)
            diff=cwr-fwr
            if abs(diff)<3: bottom_line(f"{tn} performs about the same regardless of travel distance.")
            elif diff>0: bottom_line(f"{tn} wins {diff:.0f}% more often when playing close to home — distance matters for them.")
            else: bottom_line(f"{tn} actually wins more on the road — they don't seem bothered by travel.")

# ═══════════════════════════════════════════════════════════════════════════
#  TRAVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with m_travel:
    landing("📊","Travel & Distance Analysis",
        "Does playing closer to home give teams an edge? Explore the data by distance, "
        "round, and seed. Use the sidebar filters to narrow your focus.")
    tab_exp,tab_rnd,tab_seed=st.tabs(["📊 Distance Explorer","🔄 By Round","🌱 By Seed"])

    with tab_exp:
        st.subheader("Distance vs Win Rate")
        guide_box('Teams are grouped by <strong>how far they traveled</strong>. '
            'If closer teams win more often, the bars on the left will be taller (greener) than those on the right.')
        rf=st.selectbox("Round",['All Rounds']+list(rmap.values()),key="er")
        rv={v:k for k,v in rmap.items()}.get(rf)
        df=td.copy()
        if rv: df=df[df['CURRENT ROUND']==rv]
        df=df[(df['SEED'].between(*g_seeds))&(df['YEAR'].between(*g_years))]
        if len(df)<20: st.warning("Not enough data — try widening the sidebar filters.")
        else:
            corr,pv=stats.pointbiserialr(df['WON'],df['DISTANCE (MI)'])
            wm=df[df['WON']==1]['DISTANCE (MI)'].mean();lm=df[df['WON']==0]['DISTANCE (MI)'].mean()
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Games",f"{len(df):,}");m2.metric("Relationship Strength",f"{'Weak' if abs(corr)<0.05 else 'Moderate' if abs(corr)<0.15 else 'Strong'}")
            sig="Yes ✓" if pv<0.05 else "No ✗"
            m3.metric("Statistically Significant?",sig);m4.metric("Losers Travel Extra",f"{lm-wm:+.0f} mi")
            bk=[0,100,250,500,750,1000,1500,3500];lb=['0–100','100–250','250–500','500–750','750–1K','1K–1.5K','1.5K+']
            dp=df.copy();dp['BK']=pd.cut(dp['DISTANCE (MI)'],bins=bk,labels=lb)
            bs=dp.groupby('BK',observed=True).agg(wr=('WON','mean'),n=('WON','count')).reset_index()
            bs=bs[bs['n']>=5]
            fig=make_subplots(1,2,subplot_titles=["Win Rate by Distance","Trend Line"])
            cs2=[GREEN if w>=0.5 else RED for w in bs['wr']]
            fig.add_trace(go.Bar(x=bs['BK'],y=bs['wr']*100,marker_color=cs2,
                text=[f"{w:.0%}\nn={n}" for w,n in zip(bs['wr'],bs['n'])],textposition='outside',showlegend=False),row=1,col=1)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
            if df['WON'].nunique()>1:
                s2=StandardScaler();xt=s2.fit_transform(df[['DISTANCE (MI)']].values)
                lt=LogisticRegression(random_state=42).fit(xt,df['WON'].values)
                xr=np.linspace(df['DISTANCE (MI)'].min(),df['DISTANCE (MI)'].max(),200)
                pr2=lt.predict_proba(s2.transform(xr.reshape(-1,1)))[:,1]
                fig.add_trace(go.Scatter(x=xr,y=pr2,mode='lines',line=dict(color=ORANGE,width=3),name='Trend'),row=1,col=2)
            jy=df['WON']+np.random.normal(0,0.03,len(df))
            fig.add_trace(go.Scatter(x=df['DISTANCE (MI)'],y=jy,mode='markers',
                marker=dict(size=4,color=[GREEN if w else RED for w in df['WON']],opacity=0.2),
                showlegend=False,hoverinfo='skip'),row=1,col=2)
            fig.update_layout(**plotly_layout(height=420,showlegend=True))
            fig.update_xaxes(title_text="Distance (mi)",row=1,col=1);fig.update_yaxes(title_text="Win Rate (%)",row=1,col=1)
            st.plotly_chart(fig,use_container_width=True)
            if abs(corr)<0.05: bottom_line("Distance has almost no effect on outcomes in this filter. Other factors matter more.")
            elif pv<0.05: bottom_line(f"There's a real (statistically proven) effect: losers traveled {lm-wm:.0f} miles farther on average.")
            else: bottom_line(f"There's a slight trend, but it's not strong enough to be statistically reliable with this sample size.")

    with tab_rnd:
        st.subheader("Round-by-Round Analysis")
        guide_box('The distance advantage may <strong>change by round</strong>. Early rounds are at regional sites '
            '(closer teams have an edge), while the Final Four is neutral. This shows where distance matters most.')
        df=td[(td['YEAR'].between(*g_years))&(td['SEED'].between(*g_seeds))]
        rnds=[64,32,16,8,4,2];rlb=['R64','R32','Sweet 16','Elite 8','Final 4','Championship'];res=[]
        for r in rnds:
            rd=df[df['CURRENT ROUND']==r]
            if len(rd)>10:
                cr,p=stats.pointbiserialr(rd['WON'],rd['DISTANCE (MI)']);md=rd['DISTANCE (MI)'].median()
                cw=rd[rd['DISTANCE (MI)']<=md]['WON'].mean();fw=rd[rd['DISTANCE (MI)']>md]['WON'].mean()
                res.append(dict(rnd=r,label=rlb[rnds.index(r)],r=cr,p=p,cw=cw,fw=fw,n=len(rd),sig=p<0.05))
        if not res: st.warning("Not enough data.")
        else:
            rdf=pd.DataFrame(res)
            fig=make_subplots(1,3,subplot_titles=["Close vs Far Win Rate","Effect Strength by Round","Sample Size"])
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['cw']*100,name='Closer Teams',marker_color=GREEN),row=1,col=1)
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['fw']*100,name='Farther Teams',marker_color=RED),row=1,col=1)
            fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
            bc2=[GREEN if r<0 else RED for r in rdf['r']];bo=[0.9 if s else 0.35 for s in rdf['sig']]
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['r'],showlegend=False,
                marker=dict(color=bc2,opacity=bo),text=[f"{v:+.3f}" for v in rdf['r']],textposition='outside'),row=1,col=2)
            fig.add_hline(y=0,line_color=MUTED,row=1,col=2)
            fig.add_trace(go.Bar(x=rdf['label'],y=rdf['n'],showlegend=False,marker_color=BLUE,
                text=[str(n) for n in rdf['n']],textposition='outside'),row=1,col=3)
            fig.update_layout(**plotly_layout(height=420,barmode='group'))
            st.plotly_chart(fig,use_container_width=True)
            sig_rnds=[r['label'] for _,r in rdf.iterrows() if r['sig']]
            if sig_rnds:
                bottom_line(f"Distance matters most in the <strong>{', '.join(sig_rnds)}</strong>. "
                    "In other rounds, the effect isn't strong enough to rely on for your picks.")
            else:
                bottom_line("Distance doesn't have a statistically reliable effect in any individual round — "
                    "it's a small factor that adds up across many games.")

    with tab_seed:
        st.subheader("Seed Deep Dive")
        guide_box('Do some seeds benefit more from playing close to home? A positive bar means '
            '<strong>closer teams win more often</strong> for that seed. Useful for deciding toss-up games.')
        ss_sel=st.multiselect("Seeds",[f'Seed {i}' for i in range(1,17)],
            default=[f'Seed {i}' for i in range(g_seeds[0],min(g_seeds[1]+1,9))],key="sss")
        if not ss_sel: st.info("Select at least one seed.")
        else:
            sn=[int(s.replace('Seed ','')) for s in ss_sel]
            df=td[(td['YEAR'].between(*g_years))&(td['SEED'].isin(sn))]
            if len(df)<20: st.warning("Not enough data.")
            else:
                res=[]
                for s in sorted(sn):
                    sd=df[df['SEED']==s]
                    if len(sd)>=10:
                        md=sd['DISTANCE (MI)'].median()
                        cw=sd[sd['DISTANCE (MI)']<=md]['WON'].mean();fw=sd[sd['DISTANCE (MI)']>md]['WON'].mean()
                        res.append(dict(seed=s,cw=cw,fw=fw,diff=cw-fw,n=len(sd)))
                if not res: st.warning("Min 10 games per seed needed.")
                else:
                    sdf=pd.DataFrame(res)
                    fig=make_subplots(1,2,subplot_titles=["Win Rate: Close vs Far","Advantage (pp)"])
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['cw']*100,name='Close',marker_color=GREEN),row=1,col=1)
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['fw']*100,name='Far',marker_color=RED),row=1,col=1)
                    fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=1)
                    dc=[GREEN if d>=0 else RED for d in sdf['diff']]
                    fig.add_trace(go.Bar(x=[f'#{s}' for s in sdf['seed']],y=sdf['diff']*100,marker_color=dc,
                        showlegend=False,text=[f"{d*100:+.1f}" for d in sdf['diff']],textposition='outside'),row=1,col=2)
                    fig.add_hline(y=0,line_color=MUTED,row=1,col=2)
                    fig.update_layout(**plotly_layout(height=420,barmode='group'))
                    st.plotly_chart(fig,use_container_width=True)
                    pos=(sdf['diff']>0).sum()
                    biggest=sdf.loc[sdf['diff'].abs().idxmax()]
                    bottom_line(f"{pos} of {len(sdf)} seeds benefit from playing close to home. "
                        f"<strong>Seed #{int(biggest['seed'])}</strong> shows the biggest effect "
                        f"({biggest['diff']*100:+.1f} percentage points).")

# ═══════════════════════════════════════════════════════════════════════════
#  SHOT PROFILES
# ═══════════════════════════════════════════════════════════════════════════
with m_shot:
    landing("🏀","Shot Profile DNA",
        "Every team has a unique shooting fingerprint. See how teams distribute their shots, "
        "how accurately they score from each zone, and how they compare to past champions.")
    tab_dna,tab_champ=st.tabs(["🔬 Team Shot DNA","🏆 Champion Blueprint"])

    _ss_mg=td.merge(ss[['YEAR','TEAM NO','TEAM','CONF','DUNKS FG%','DUNKS SHARE','DUNKS FG%D','DUNKS D SHARE',
        'CLOSE TWOS FG%','CLOSE TWOS SHARE','CLOSE TWOS FG%D','CLOSE TWOS D SHARE',
        'FARTHER TWOS FG%','FARTHER TWOS SHARE','FARTHER TWOS FG%D','FARTHER TWOS D SHARE',
        'THREES FG%','THREES SHARE','THREES FG%D','THREES D SHARE']],
        on=['YEAR','TEAM NO'],how='inner',suffixes=('','_ss'))
    _tr=_ss_mg.groupby(['YEAR','TEAM NO']).agg(deepest=('ROUND','min'),gw=('WON','sum'),
        seed=('SEED','first'),team=('TEAM','first')).reset_index()
    _tr=_tr.merge(ss[['YEAR','TEAM NO','CONF','DUNKS FG%','DUNKS SHARE','DUNKS FG%D','DUNKS D SHARE',
        'CLOSE TWOS FG%','CLOSE TWOS SHARE','CLOSE TWOS FG%D','CLOSE TWOS D SHARE',
        'FARTHER TWOS FG%','FARTHER TWOS SHARE','FARTHER TWOS FG%D','FARTHER TWOS D SHARE',
        'THREES FG%','THREES SHARE','THREES FG%D','THREES D SHARE']],on=['YEAR','TEAM NO'],how='inner')
    _tr['run']=_tr['deepest'].map({1:'Champion',2:'Runner-Up',4:'Final Four',8:'Elite 8',
        16:'Sweet 16',32:'Round of 32',64:'Round of 64'})
    _zones=['Dunks','Close Twos','Mid-Range','Threes']
    _sh=['DUNKS SHARE','CLOSE TWOS SHARE','FARTHER TWOS SHARE','THREES SHARE']
    _fg=['DUNKS FG%','CLOSE TWOS FG%','FARTHER TWOS FG%','THREES FG%']
    _fgd=['DUNKS FG%D','CLOSE TWOS FG%D','FARTHER TWOS FG%D','THREES FG%D']
    _zcol=[GREEN,CYAN,ORANGE,PURPLE]

    def pctile(val,series,inv=False):
        p=stats.percentileofscore(series.dropna(),val,kind='rank')
        return 100-p if inv else p

    with tab_dna:
        st.subheader("Team Shot DNA")
        guide_box('Compare any two teams\' <strong>shooting fingerprints</strong>. '
            'See where their shots come from, how accurately they score each type, '
            'and how they stack up against the field.')
        c1,c2,c3=st.columns([1,2,2])
        with c1: dy=st.selectbox("Season",sorted(ss['YEAR'].unique(),reverse=True),key="dy")
        yt=sorted(ss[ss['YEAR']==dy]['TEAM'].unique())
        da=g_fav if g_fav!="—" and g_fav in yt else yt[0]
        with c2: dta=st.selectbox("Team A",yt,index=yt.index(da) if da in yt else 0,key="dta")
        with c3: dtb=st.selectbox("Compare To",["— None —"]+yt,key="dtb")
        ra=ss[(ss['YEAR']==dy)&(ss['TEAM']==dta)]
        if not len(ra): st.warning(f"No data for {dta}.")
        else:
            ra=ra.iloc[0]; comp=dtb!="— None —"
            rb=None
            if comp:
                _rb=ss[(ss['YEAR']==dy)&(ss['TEAM']==dtb)]
                if len(_rb): rb=_rb.iloc[0]
                else: comp=False
            st.markdown("#### Where Do They Shoot From?")
            sha=[ra[c] for c in _sh]
            if comp:
                shb=[rb[c] for c in _sh]
                fig=make_subplots(1,2,specs=[[{"type":"pie"},{"type":"pie"}]],subplot_titles=[dta,dtb])
                fig.add_trace(go.Pie(labels=_zones,values=sha,marker=dict(colors=_zcol),
                    textinfo='label+percent',hole=0.35),row=1,col=1)
                fig.add_trace(go.Pie(labels=_zones,values=shb,marker=dict(colors=_zcol),
                    textinfo='label+percent',hole=0.35),row=1,col=2)
            else:
                fig=go.Figure(go.Pie(labels=_zones,values=sha,marker=dict(colors=_zcol),
                    textinfo='label+percent+value',hole=0.4))
            fig.update_layout(**plotly_layout(height=350,showlegend=False))
            st.plotly_chart(fig,use_container_width=True)
            bottom_line(f"{dta} takes {ra['THREES SHARE']:.0f}% of shots from three and {ra['DUNKS SHARE']:.0f}% from dunks. "
                f"{'Heavy perimeter team.' if ra['THREES SHARE']>38 else 'Balanced attack.' if ra['THREES SHARE']>32 else 'Inside-focused team.'}")

            st.markdown("#### How Accurately Do They Score?")
            read_box('📊 <strong>Green</strong> = their shooting accuracy. <strong>Red</strong> = what they allow opponents. '
                'Bigger green-red gaps = better team. Focus on <strong>close twos</strong> and <strong>threes</strong> — those separate good teams from great ones.')
            fga=[ra[c] for c in _fg];fgda=[ra[c] for c in _fgd]
            if comp:
                fgb=[rb[c] for c in _fg];fgdb=[rb[c] for c in _fgd]
                fig=make_subplots(1,2,subplot_titles=[dta,dtb])
                for ci,(fg,fgd) in enumerate([(fga,fgda),(fgb,fgdb)],1):
                    fig.add_trace(go.Bar(x=_zones,y=fg,name='Offense' if ci==1 else None,marker_color=GREEN,
                        showlegend=(ci==1),text=[f"{v:.0f}%" for v in fg],textposition='outside'),row=1,col=ci)
                    fig.add_trace(go.Bar(x=_zones,y=fgd,name='Defense' if ci==1 else None,marker_color=RED,
                        showlegend=(ci==1),text=[f"{v:.0f}%" for v in fgd],textposition='outside'),row=1,col=ci)
            else:
                fig=go.Figure()
                fig.add_trace(go.Bar(x=_zones,y=fga,name='Their Shooting',marker_color=GREEN,
                    text=[f"{v:.0f}%" for v in fga],textposition='outside'))
                fig.add_trace(go.Bar(x=_zones,y=fgda,name='What They Allow',marker_color=RED,
                    text=[f"{v:.0f}%" for v in fgda],textposition='outside'))
            fig.update_layout(**plotly_layout(height=400,barmode='group'));fig.update_yaxes(title_text="FG %")
            st.plotly_chart(fig,use_container_width=True)

            st.markdown("#### Strengths & Weaknesses Radar")
            read_box('📊 Each spoke is ranked as a <strong>percentile</strong> (0–100) against all tournament teams that year. '
                'Bigger = better. Hover for exact values.')
            yr_d=ss[ss['YEAR']==dy]
            rcols=['DUNKS SHARE','CLOSE TWOS FG%','FARTHER TWOS FG%','THREES FG%','THREES SHARE','DUNKS FG%D','THREES FG%D']
            rnames=['Rim Pressure','Close 2s Accuracy','Mid-Range','3PT Shooting','3PT Volume','Rim Protection↓','3PT Defense↓']
            fig=go.Figure()
            va=[];ha=[]
            for c,l in zip(rcols,rnames):
                inv='↓' in l;p=pctile(ra[c],yr_d[c],inv=inv);va.append(p)
                ha.append(f"{l}: {ra[c]:.1f} ({p:.0f}th percentile)")
            va.append(va[0]);ha.append(ha[0]);lc=rnames+[rnames[0]]
            fig.add_trace(go.Scatterpolar(r=va,theta=lc,fill='toself',name=dta,
                line=dict(color=GREEN,width=2),fillcolor='rgba(34,197,94,0.15)',hovertext=ha,hoverinfo='text'))
            if comp:
                vb=[];hb=[]
                for c,l in zip(rcols,rnames):
                    inv='↓' in l;p=pctile(rb[c],yr_d[c],inv=inv);vb.append(p)
                    hb.append(f"{l}: {rb[c]:.1f} ({p:.0f}th percentile)")
                vb.append(vb[0]);hb.append(hb[0])
                fig.add_trace(go.Scatterpolar(r=vb,theta=lc,fill='toself',name=dtb,
                    line=dict(color=ORANGE,width=2),fillcolor='rgba(249,115,22,0.15)',hovertext=hb,hoverinfo='text'))
            fig.update_layout(**plotly_layout(height=480,showlegend=True),
                polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,100],
                    gridcolor='#1e2530',tickfont=dict(size=9,color=MUTED)),
                    angularaxis=dict(gridcolor='#2a2f3a',tickfont=dict(size=10,color=LIGHT))))
            st.plotly_chart(fig,use_container_width=True)

    with tab_champ:
        st.subheader("What Does a Champion Look Like?")
        guide_box('Champions share a consistent shooting DNA. This tab shows the <strong>champion blueprint</strong> '
            'and lets you check how any team measures up. Gold = champion average, Green = your team.')
        st.markdown("#### Shooting Profile by Tournament Run")
        read_box('📊 Each bar shows the average for teams whose <strong>deepest run</strong> was that round. '
            'A clear upward trend means that metric helps teams advance. Gold = champions.')
        ro=['Round of 64','Round of 32','Sweet 16','Elite 8','Final Four','Runner-Up','Champion']
        dm=['DUNKS SHARE','CLOSE TWOS FG%','THREES FG%','THREES FG%D']
        dl=['Dunk Share %','Close 2s Accuracy','3PT Accuracy','Opponent 3PT% (lower=better)']
        fig=make_subplots(1,4,subplot_titles=dl)
        for i,(c,l) in enumerate(zip(dm,dl),1):
            vs=[];ls2=[];cs2=[]
            for r in ro:
                rd=_tr[_tr['run']==r]
                if len(rd)>=3:
                    vs.append(rd[c].mean());ls2.append(r.replace('Round of ','R'));cs2.append(GOLD if r=='Champion' else BLUE)
            fig.add_trace(go.Bar(x=ls2,y=vs,marker_color=cs2,showlegend=False,
                text=[f"{v:.1f}" for v in vs],textposition='outside'),row=1,col=i)
        fig.update_layout(**plotly_layout(height=400))
        st.plotly_chart(fig,use_container_width=True)
        bottom_line("Champions get to the rim more (10% dunk share vs 6% for early exits) and "
            "are elite at close-range scoring (65.6% vs 61.3%). When in doubt, pick the team that dominates inside.")

        st.markdown("#### Your Team vs the Champion Blueprint")
        cc1,cc2=st.columns([1,2])
        with cc1: chy=st.selectbox("Season",sorted(ss['YEAR'].unique(),reverse=True),key="chy")
        cht=sorted(ss[ss['YEAR']==chy]['TEAM'].unique())
        chd=g_fav if g_fav!="—" and g_fav in cht else cht[0]
        with cc2: chtn=st.selectbox("Team",cht,index=cht.index(chd) if chd in cht else 0,key="cht")
        champ=_tr[_tr['deepest']==1];trow=ss[(ss['YEAR']==chy)&(ss['TEAM']==chtn)]
        if not len(trow): st.warning(f"No data for {chtn}.")
        elif not len(champ): st.warning("No champion data available.")
        else:
            trow=trow.iloc[0]
            rm2=['DUNKS SHARE','CLOSE TWOS FG%','FARTHER TWOS FG%','THREES FG%','THREES SHARE','THREES FG%D','CLOSE TWOS FG%D']
            rn2=['Dunk Share','Close 2s FG%','Mid-Range FG%','3PT FG%','3PT Volume','3PT Defense↓','Close 2s Defense↓']
            fig=go.Figure()
            cv2=[];ch2=[]
            for c,l in zip(rm2,rn2):
                inv='↓' in l;raw=champ[c].mean();p=pctile(raw,ss[c],inv=inv)
                cv2.append(p);ch2.append(f"{l}: {raw:.1f} ({p:.0f}th %ile)")
            cv2.append(cv2[0]);ch2.append(ch2[0]);lc=rn2+[rn2[0]]
            fig.add_trace(go.Scatterpolar(r=cv2,theta=lc,fill='toself',name='Champion Avg',
                line=dict(color=GOLD,width=3),fillcolor='rgba(251,191,36,0.12)',hovertext=ch2,hoverinfo='text'))
            tv=[];th=[]
            for c,l in zip(rm2,rn2):
                inv='↓' in l;raw=trow[c];p=pctile(raw,ss[c],inv=inv)
                tv.append(p);th.append(f"{l}: {raw:.1f} ({p:.0f}th %ile)")
            tv.append(tv[0]);th.append(th[0])
            fig.add_trace(go.Scatterpolar(r=tv,theta=lc,fill='toself',name=f'{chtn} ({chy})',
                line=dict(color=GREEN,width=2),fillcolor='rgba(34,197,94,0.12)',hovertext=th,hoverinfo='text'))
            fig.update_layout(**plotly_layout(height=500,showlegend=True),
                polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,100],
                    gridcolor='#1e2530',tickfont=dict(size=9,color=MUTED)),
                    angularaxis=dict(gridcolor='#2a2f3a',tickfont=dict(size=10,color=LIGHT))))
            st.plotly_chart(fig,use_container_width=True)
            gaps=[]
            for c,l in zip(rm2,rn2):
                inv='↓' in l;ca2=champ[c].mean();tv2=trow[c]
                d2=(ca2-tv2) if inv else(tv2-ca2)
                gaps.append(dict(Metric=l.replace('↓',''),Team=f"{tv2:.1f}",Champion=f"{ca2:.1f}",
                    Gap=f"{d2:+.1f}",Status='✅ Above' if d2>=0 else '⚠️ Below'))
            st.dataframe(pd.DataFrame(gaps),use_container_width=True,hide_index=True)
            above=sum(1 for g in gaps if '✅' in g['Status'])
            if above==len(gaps): bottom_line(f"<strong>{chtn}</strong> matches the champion blueprint in every category — elite profile!")
            elif above>=len(gaps)*0.7: bottom_line(f"<strong>{chtn}</strong> hits {above}/{len(gaps)} champion marks. Strong contender, but watch the gaps.")
            else: bottom_line(f"<strong>{chtn}</strong> only matches {above}/{len(gaps)} champion marks — they'd need to overperform to win it all.")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFERENCES
# ═══════════════════════════════════════════════════════════════════════════
with m_conf:
    landing("🏛️","Conference Intelligence","Compare conferences across efficiency, talent, and the four factors. "
        "When two teams from different conferences meet, this helps you judge which conference prepares teams better.")
    st.subheader("Conference Comparison")
    c1,c2=st.columns([1,2])
    with c1: cy=st.selectbox("Season",sorted(cs['YEAR'].unique(),reverse=True),key="cy")
    with c2:
        ac=sorted(cs[cs['YEAR']==cy]['CONF'].unique())
        md=[c for c in['SEC','B10','B12','ACC','BE','MWC'] if c in ac]
        sc2=st.multiselect("Conferences",ac,default=md or ac[:4],key="cs2")
    if not sc2: st.info("Select at least one conference.")
    else:
        csy=cs[(cs['YEAR']==cy)&(cs['CONF'].isin(sc2))].sort_values('BADJ EM',ascending=False)
        st.markdown("#### Overview")
        read_box('📊 <strong>Adj EM</strong> = how many points per 100 possessions the conference is above average (higher = stronger). '
            '<strong>WAB</strong> = wins above bubble (positive = conference produces tournament-quality teams). '
            '<strong>Talent</strong> = recruiting strength.')
        ov=csy[['CONF','BADJ EM','BADJ O','BADJ D','BARTHAG','TALENT','ELITE SOS','WAB']].copy()
        ov.columns=['Conference','Strength','Offense','Defense','Win Prob vs Avg','Talent','Elite SOS','Wins Above Bubble']
        st.dataframe(ov.style.format({'Strength':'{:.1f}','Offense':'{:.1f}','Defense':'{:.1f}',
            'Win Prob vs Avg':'{:.3f}','Talent':'{:.1f}','Elite SOS':'{:.1f}','Wins Above Bubble':'{:.1f}'}
        ),use_container_width=True,hide_index=True)
        top_conf=csy.iloc[0]['CONF']
        bottom_line(f"The <strong>{top_conf}</strong> is the strongest conference by efficiency margin. "
            f"Teams from stronger conferences tend to be battle-tested for the tournament.")

        fig=make_subplots(1,3,subplot_titles=["Conference Strength","Offense vs Defense","Talent vs WAB"])
        ec=[GREEN if v>=10 else(ORANGE if v>=5 else(BLUE if v>=0 else RED)) for v in csy['BADJ EM']]
        fig.add_trace(go.Bar(x=csy['CONF'],y=csy['BADJ EM'],marker_color=ec,
            text=[f"{v:.1f}" for v in csy['BADJ EM']],textposition='outside',showlegend=False),row=1,col=1)
        fig.add_hline(y=0,line_color=MUTED,row=1,col=1)
        for _,r in csy.iterrows():
            fig.add_trace(go.Scatter(x=[r['BADJ O']],y=[r['BADJ D']],mode='markers+text',
                marker=dict(size=14,color=ORANGE),text=[r['CONF']],textposition='top center',
                textfont=dict(size=11,color=LIGHT),showlegend=False),row=1,col=2)
        fig.update_yaxes(autorange='reversed',title_text="Defense (↓ better)",row=1,col=2)
        for _,r in csy.iterrows():
            fig.add_trace(go.Scatter(x=[r['TALENT']],y=[r['WAB']],mode='markers+text',
                marker=dict(size=14,color=GREEN if r['WAB']>=0 else RED),text=[r['CONF']],
                textposition='top center',textfont=dict(size=11,color=LIGHT),showlegend=False),row=1,col=3)
        fig.add_hline(y=0,line_dash="dash",line_color=MUTED,row=1,col=3)
        fig.update_layout(**plotly_layout(height=420))
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("#### Four Factors of Basketball")
        read_box('📊 The four keys to winning: <strong>shooting efficiency</strong>, <strong>avoiding turnovers</strong>, '
            '<strong>rebounding</strong>, and <strong>getting to the free throw line</strong>. Green = offense, Red = defense.')
        fig=make_subplots(1,4,subplot_titles=["Shooting Eff.","Turnovers","Rebounding","Free Throws"])
        for i,(o,d) in enumerate([('EFG%','EFGD%'),('TOV%','TOV%D'),('OREB%','DREB%'),('FTR','FTRD')],1):
            fig.add_trace(go.Bar(x=csy['CONF'],y=csy[o],name='Off' if i==1 else None,
                marker_color=GREEN,showlegend=(i==1)),row=1,col=i)
            fig.add_trace(go.Bar(x=csy['CONF'],y=csy[d],name='Def' if i==1 else None,
                marker_color=RED,showlegend=(i==1)),row=1,col=i)
        fig.update_layout(**plotly_layout(height=350,barmode='group'))
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("#### Historical Trends")
        ch=cs[cs['CONF'].isin(sc2)];cc=[ORANGE,GREEN,BLUE,CYAN,PURPLE,RED,'#ec4899','#84cc16']
        fig=make_subplots(1,3,subplot_titles=["Strength Over Time","Talent","Wins Above Bubble"])
        for i,cf in enumerate(sc2):
            cd=ch[ch['CONF']==cf].sort_values('YEAR');cl=cc[i%len(cc)]
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['BADJ EM'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf),row=1,col=1)
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['TALENT'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf,showlegend=False),row=1,col=2)
            fig.add_trace(go.Scatter(x=cd['YEAR'],y=cd['WAB'],mode='lines+markers',
                line=dict(color=cl,width=2),marker=dict(size=5),name=cf,legendgroup=cf,showlegend=False),row=1,col=3)
        fig.add_hline(y=0,line_dash="dash",line_color=MUTED,row=1,col=3)
        fig.update_layout(**plotly_layout(height=400))
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  UNDER THE HOOD (MODELS)
# ═══════════════════════════════════════════════════════════════════════════
with m_models:
    landing("🤖","Under the Hood","For the data nerds — see how the prediction models work, "
        "compare their accuracy, and understand what features drive tournament outcomes.")
    st.subheader("Model Accuracy")
    guide_box('We trained <strong>5 prediction models</strong>, each using more data than the last. '
        'Accuracy is measured on a 50–100% scale where 50% = random guessing and 100% = perfect predictions.')
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("Distance Only",f"{acc1:.0f}%")
    m2.metric("+ Seed",f"{acc2:.0f}%")
    m3.metric("+ Time Zones",f"{acc3:.0f}%")
    m4.metric("+ Power Ratings",f"{acc4:.0f}%")
    m5.metric("+ KenPom (Best)",f"{acc5:.0f}%")
    read_box('📊 <strong>Left:</strong> ROC curves show each model\'s ability to separate winners from losers — '
        'further from the diagonal = better. <strong>Right:</strong> Accuracy scores with error bars.')
    fig=make_subplots(1,2,subplot_titles=["Model Performance Curves","Accuracy Comparison"])
    for Xs,md,yt,nm,cl,ds in[(X1s,lr1,ya,f'Distance ({acc1:.0f}%)',MUTED,'dot'),
        (X2s,lr2,ya,f'+Seed ({acc2:.0f}%)',ORANGE,'solid'),
        (X3s,lr3,ya,f'+TZ ({acc3:.0f}%)',CYAN,'dash'),
        (X4s,lr4,y4,f'+Power ({acc4:.0f}%)',GREEN,'solid'),
        (X5s,lr5,y5,f'+KenPom ({acc5:.0f}%)',PURPLE,'solid')]:
        yp=md.predict_proba(Xs)[:,1];fpr,tpr,_=roc_curve(yt,yp)
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',line=dict(color=cl,width=2.5,dash=ds),name=nm),row=1,col=1)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color=MUTED,dash='dot',width=1),name='Random'),row=1,col=1)
    nms=['Dist','D+S','D+S+TZ','Enh','Full'];aucs=[acc1,acc2,acc3,acc4,acc5]
    sds=[std1,std2,std3,std4,std5];bcs=[MUTED,ORANGE,CYAN,GREEN,PURPLE]
    fig.add_trace(go.Bar(x=nms,y=aucs,marker_color=bcs,
        error_y=dict(type='data',array=[float(s*100) for s in sds],visible=True,color=MUTED),
        text=[f"{a:.0f}%" for a in aucs],textposition='outside',showlegend=False),row=1,col=2)
    fig.add_hline(y=50,line_dash="dash",line_color=MUTED,row=1,col=2)
    fig.update_layout(**plotly_layout(height=420))
    fig.update_yaxes(range=[45,max(aucs)+8],row=1,col=2)
    st.plotly_chart(fig,use_container_width=True)
    bottom_line(f"Our best model is {max(aucs):.0f}% accurate — significantly better than the {acc1:.0f}% "
        f"you'd get from distance alone. Adding team quality data is what makes the difference.")

    st.markdown("#### What Matters Most?")
    guide_box('These charts show which factors have the <strong>biggest impact</strong> on predictions. '
        'Green bars increase a team\'s win chance. Red bars decrease it. Longer bars = bigger impact.')
    fig=make_subplots(1,2,subplot_titles=[f"Enhanced Model ({acc4:.0f}% acc.)",
        f"Full Model ({acc5:.0f}% acc.)"])
    ft4=['Distance','Seed','Power Rating','Schedule Strength'];c4=list(lr4.coef_[0]);si=np.argsort(np.abs(c4))
    fig.add_trace(go.Bar(y=[ft4[i] for i in si],x=[c4[i] for i in si],orientation='h',
        marker_color=[RED if c<0 else GREEN for c in[c4[i] for i in si]],
        text=[f"{c4[i]:+.3f}" for i in si],textposition='outside',showlegend=False),row=1,col=1)
    ft5=['Distance','Seed','Power Rating','Sched. Str.','KenPom Eff.','Improvement','Top-25 Wins']
    c5=list(lr5.coef_[0]);si=np.argsort(np.abs(c5))
    fig.add_trace(go.Bar(y=[ft5[i] for i in si],x=[c5[i] for i in si],orientation='h',
        marker_color=[RED if c<0 else GREEN for c in[c5[i] for i in si]],
        text=[f"{c5[i]:+.3f}" for i in si],textposition='outside',showlegend=False),row=1,col=2)
    fig.add_vline(x=0,line_color=MUTED,row=1,col=1);fig.add_vline(x=0,line_color=MUTED,row=1,col=2)
    fig.update_layout(**plotly_layout(height=380))
    st.plotly_chart(fig,use_container_width=True)
    bottom_line("KenPom efficiency (how well a team scores vs defends per possession) is the single strongest predictor. "
        "Distance matters, but team quality matters <strong>far</strong> more.")

    with st.expander("📊 Detailed Model Summary Table"):
        ms=pd.DataFrame([
            dict(Model='1. Distance',What_It_Uses='Distance only',Accuracy=f"{acc1:.1f}%",Games=f"{len(ya):,}",Years='2008–2025'),
            dict(Model='2. + Seed',What_It_Uses='Distance, Seed',Accuracy=f"{acc2:.1f}%",Games=f"{len(ya):,}",Years='2008–2025'),
            dict(Model='3. + Time Zones',What_It_Uses='Distance, Seed, Time Zones',Accuracy=f"{acc3:.1f}%",Games=f"{len(ya):,}",Years='2008–2025'),
            dict(Model='4. Enhanced',What_It_Uses='Distance, Seed, Power Rating, SOS',Accuracy=f"{acc4:.1f}%",Games=f"{len(y4):,}",Years='2008–2025'),
            dict(Model='5. Full (Best)',What_It_Uses='All above + KenPom, Improvement, Top-25 Wins',Accuracy=f"{acc5:.1f}%",Games=f"{len(y5):,}",Years='2012–2025')])
        st.dataframe(ms,use_container_width=True,hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("NCAA Tournament Analytics Hub · Created by Levi Schrandt · Purdue University · "
           "Data: 2008–2025 · Built with Streamlit + Plotly + scikit-learn")
