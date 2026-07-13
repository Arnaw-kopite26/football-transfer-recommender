import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="⚽ Transfer Scout AI",
    page_icon="⚽",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;600;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #060614 0%, #0d0d2b 50%, #0a0a1a 100%);
        color: white;
    }

    .main-title {
        text-align: center;
        font-family: 'Rajdhani', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff416c, #00d2ff);
        background-size: 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        letter-spacing: 4px;
        padding: 10px 0;
    }

    @keyframes gradientShift {
        0% { background-position: 0% }
        50% { background-position: 100% }
        100% { background-position: 0% }
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .stat-pill {
        background: rgba(123, 47, 247, 0.15);
        border: 1px solid rgba(123, 47, 247, 0.4);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.85rem;
        color: #aaa;
        display: inline-block;
        margin: 3px;
    }

    .player-card {
        background: linear-gradient(135deg, rgba(30,30,63,0.9), rgba(45,45,94,0.9));
        border: 1px solid rgba(123, 47, 247, 0.5);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(123, 47, 247, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .player-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #7b2ff7, #00d2ff, #ff416c);
    }

    .player-name {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1px;
    }

    .player-club {
        font-size: 1rem;
        color: #00d2ff;
        font-weight: 600;
        margin-top: 4px;
    }

    .player-league {
        font-size: 0.85rem;
        color: #666;
        margin-top: 2px;
    }

    .match-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 900;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.4);
    }

    .match-mid {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #000;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 900;
        font-size: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(247, 151, 30, 0.4);
    }

    .rank-circle {
        background: linear-gradient(135deg, #7b2ff7, #00d2ff);
        color: white;
        width: 55px;
        height: 55px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.5);
        flex-shrink: 0;
    }

    .stat-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
        transition: all 0.2s;
    }

    .stat-box:hover {
        background: rgba(123,47,247,0.15);
        border-color: rgba(123,47,247,0.4);
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #00d2ff;
    }

    .stat-label {
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 3px;
    }

    .section-header {
        background: linear-gradient(90deg, rgba(123,47,247,0.3), rgba(0,210,255,0.3));
        border: 1px solid rgba(123,47,247,0.4);
        padding: 15px 25px;
        border-radius: 12px;
        color: white;
        font-weight: 800;
        font-size: 1.3rem;
        margin: 25px 0 15px 0;
        text-align: center;
        letter-spacing: 1px;
    }

    .weakness-banner {
        background: linear-gradient(90deg, rgba(255,65,108,0.2), rgba(255,75,43,0.2));
        border: 1px solid rgba(255,65,108,0.5);
        border-radius: 12px;
        padding: 15px 25px;
        text-align: center;
        margin: 15px 0;
    }

    .weakness-text {
        color: #ff416c;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }

    .info-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }

    .info-number {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #7b2ff7, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .info-label {
        color: #666;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 5px;
    }

    .squad-player {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .pos-badge-fw { background: rgba(255,65,108,0.2); color: #ff416c; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .pos-badge-mf { background: rgba(123,47,247,0.2); color: #7b2ff7; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .pos-badge-df { background: rgba(0,210,255,0.2); color: #00d2ff; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
    .pos-badge-gk { background: rgba(150,201,61,0.2); color: #96c93d; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }

    .stButton > button {
        background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 18px 40px !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        width: 100% !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        box-shadow: 0 8px 25px rgba(123,47,247,0.4) !important;
        transition: all 0.3s !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: rgba(30,30,63,0.8) !important;
        border: 1px solid rgba(123,47,247,0.5) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .footer {
        text-align: center;
        color: #333;
        font-size: 0.8rem;
        letter-spacing: 2px;
        padding: 20px 0;
        text-transform: uppercase;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\arnaw\OneDrive\Desktop\transfer_recomender\data"

# ── Load Data ─────────────────────────────────────────
@st.cache_data
def load_data():
    cols = ['Player', 'Squad', 'Comp', 'Pos', 'Age', 'Min', 'cluster']
    fw      = pd.read_csv(os.path.join(DATA_DIR, "fw_clustered.csv"))
    mf      = pd.read_csv(os.path.join(DATA_DIR, "mf_clustered.csv"))
    df      = pd.read_csv(os.path.join(DATA_DIR, "df_clustered.csv"))
    players = pd.read_csv(os.path.join(DATA_DIR, "players_data-2024_2025.csv"),
                         usecols=['Player', 'Squad', 'Comp', 'Pos', 'Age', 'Nation'])
    return fw, mf, df, players

fw, mf, df, players = load_data()
team_list = sorted(players['Squad'].unique().tolist())

# ── Stat name mapping ─────────────────────────────────
STAT_NAMES = {
    'Gls': 'Goals', 'Ast': 'Assists', 'Sh': 'Shots',
    'SoT': 'On Target', 'KP': 'Key Passes',
    'PrgP': 'Prog Passes', 'PrgC': 'Prog Carries',
    'Tkl': 'Tackles', 'Int': 'Interceptions',
    'Clr': 'Clearances', 'Blocks': 'Blocks',
    'G/Sh': 'Goals/Shot', 'xG': 'xGoals',
    'xAG': 'xAssists', 'PPA': 'Passes to Box'
}

POS_STATS = {
    'FW': ['Gls', 'Ast', 'Sh', 'SoT', 'KP'],
    'MF': ['Ast', 'KP', 'PrgP', 'Tkl', 'Gls'],
    'DF': ['Tkl', 'Int', 'Clr', 'PrgP', 'Blocks']
}

POS_LABELS = {
    'FW': '⚡ Forward',
    'MF': '🎨 Midfielder',
    'DF': '🛡️ Defender'
}

# ── Core Functions ────────────────────────────────────
def get_pos_data(position):
    if position == 'FW':
        return fw, ['FW', 'FW,MF', 'MF,FW']
    elif position == 'MF':
        return mf, ['MF', 'MF,DF', 'DF,MF', 'MF,FW', 'FW,MF']
    else:
        return df, ['DF', 'DF,MF', 'MF,DF']

def detect_weakness(team_name, position):
    pos_data, pos_filter = get_pos_data(position)
    numeric_cols = [c for c in pos_data.select_dtypes(include=[np.number]).columns if c != 'cluster']
    league_avg = pos_data[numeric_cols].mean()
    team_players = players[(players['Squad'] == team_name) & (players['Pos'].isin(pos_filter))]
    team_pos = pos_data[pos_data['Player'].isin(team_players['Player'])]
    if team_pos.empty:
        return None
    return (league_avg - team_pos[numeric_cols].mean()).mean()

def find_weakest_position(team_name):
    scores = {pos: detect_weakness(team_name, pos) for pos in ['FW', 'MF', 'DF']}
    scores = {k: v for k, v in scores.items() if v is not None}
    return max(scores, key=scores.get) if scores else 'FW'

def get_recommendations(team_name, position, top_n=5):
    pos_data, pos_filter = get_pos_data(position)
    numeric_cols = [c for c in pos_data.select_dtypes(include=[np.number]).columns if c != 'cluster']
    key_stats = [s for s in POS_STATS[position] if s in pos_data.columns]

    team_players = players[(players['Squad'] == team_name) & (players['Pos'].isin(pos_filter))]
    team_pos = pos_data[pos_data['Player'].isin(team_players['Player'])]

    if team_pos.empty:
        return None, [], None

    team_avg = team_pos[numeric_cols].mean().fillna(0)
    other = pos_data[~pos_data['Player'].isin(team_players['Player'])].copy()

    scores = cosine_similarity(team_avg.values.reshape(1, -1), other[numeric_cols].fillna(0).values)[0]
    other['similarity'] = scores

    cols = ['Player', 'Squad', 'Comp', 'Age', 'similarity'] + key_stats
    available_cols = [c for c in cols if c in other.columns]

    return other.sort_values('similarity', ascending=False).head(top_n)[available_cols], key_stats, team_avg

def make_radar_chart(player_row, team_avg, key_stats):
    available = [s for s in key_stats if s in player_row.index and s in team_avg.index]
    if len(available) < 3:
        return None

    labels = [STAT_NAMES.get(s, s) for s in available]
    player_vals = [float(player_row[s]) if pd.notna(player_row[s]) else 0 for s in available]
    team_vals   = [float(team_avg[s])   if pd.notna(team_avg[s])   else 0 for s in available]

    max_vals = [max(p, t) + 0.1 for p, t in zip(player_vals, team_vals)]
    player_norm = [p/m if m > 0 else 0 for p, m in zip(player_vals, max_vals)]
    team_norm   = [t/m if m > 0 else 0 for t, m in zip(team_vals,   max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_norm + [player_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Player',
        line=dict(color='#00d2ff', width=2),
        fillcolor='rgba(0,210,255,0.15)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=team_norm + [team_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Team Ideal',
        line=dict(color='#7b2ff7', width=2),
        fillcolor='rgba(123,47,247,0.15)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(color='#666')
        ),
        showlegend=True,
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=280
    )
    return fig

# ── HEADER ────────────────────────────────────────────
st.markdown('<div class="main-title">⚽ TRANSFER SCOUT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning • Big 5 European Leagues • 2024/25 Season</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── STATS BAR ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="info-card"><div class="info-number">2,854</div><div class="info-label">Players Scouted</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-card"><div class="info-number">96</div><div class="info-label">Teams</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-card"><div class="info-number">5</div><div class="info-label">Leagues</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="info-card"><div class="info-number">267</div><div class="info-label">Data Points</div></div>', unsafe_allow_html=True)

st.divider()

# ── CONTROLS ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏟️ Your Team")
    team = st.selectbox("", team_list, label_visibility="collapsed")

with col2:
    st.markdown("### 📍 Position Need")
    position_mode = st.selectbox(
        "",
        ["🤖 Auto Detect Weakness", "⚡ Striker (FW)", "🎨 Midfielder (MF)", "🛡️ Defender (DF)"],
        label_visibility="collapsed"
    )

with col3:
    st.markdown("### 🎯 Results Count")
    top_n = st.slider("", 3, 10, 5, label_visibility="collapsed")

# ── SQUAD VIEWER ──────────────────────────────────────
with st.expander(f"👥 View {team}'s Current Squad"):
    squad = players[players['Squad'] == team][['Player', 'Pos', 'Age', 'Nation']].dropna()
    if not squad.empty:
        cols = st.columns(2)
        for idx, (_, p) in enumerate(squad.iterrows()):
            pos = str(p['Pos']).split(',')[0]
            if pos == 'FW':
                badge = f'<span class="pos-badge-fw">FW</span>'
            elif pos == 'MF':
                badge = f'<span class="pos-badge-mf">MF</span>'
            elif pos == 'DF':
                badge = f'<span class="pos-badge-df">DF</span>'
            else:
                badge = f'<span class="pos-badge-gk">GK</span>'

            with cols[idx % 2]:
                st.markdown(f"""
                <div class="squad-player">
                    {badge}
                    <span style="color:white; font-weight:600;">{p['Player']}</span>
                    <span style="color:#666; font-size:0.8rem; margin-left:auto;">Age {int(p['Age']) if pd.notna(p['Age']) else '?'}</span>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SEARCH BUTTON ─────────────────────────────────────
search = st.button("🔍 SCOUT TRANSFER TARGETS", use_container_width=True)

if search:
    with st.spinner("🧠 Analysing squad DNA and scanning 2,854 players..."):

        # resolve position
        if "Auto" in position_mode:
            position = find_weakest_position(team)
            st.markdown(f"""
            <div class="weakness-banner">
                <div class="weakness-text">⚠️ WEAKNESS DETECTED — {POS_LABELS[position]} Position</div>
                <div style="color:#666; font-size:0.85rem; margin-top:5px;">
                    Based on squad analysis vs Big 5 league averages
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif "FW" in position_mode:
            position = "FW"
        elif "MF" in position_mode:
            position = "MF"
        else:
            position = "DF"

        results, key_stats, team_avg = get_recommendations(team, position, top_n)

        if results is not None:
            st.markdown(f'<div class="section-header">🏆 TOP {top_n} {POS_LABELS[position].upper()} TARGETS FOR {team.upper()}</div>', unsafe_allow_html=True)

            for rank, (i, row) in enumerate(results.iterrows(), 1):
                pct = round(row['similarity'] * 100, 1)
                badge_class = "match-high" if pct >= 99 else "match-mid"
                age_str = f"Age {int(row['Age'])}" if 'Age' in row and pd.notna(row['Age']) else ""

                st.markdown(f"""
                <div class="player-card">
                    <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
                        <div class="rank-circle">#{rank}</div>
                        <div style="flex:1; min-width:200px;">
                            <div class="player-name">{row['Player']}</div>
                            <div class="player-club">🏟️ {row['Squad']} &nbsp;•&nbsp; {age_str}</div>
                            <div class="player-league">🌍 {row['Comp']}</div>
                        </div>
                        <div class="{badge_class}">{pct}%<br><span style="font-size:0.7rem; font-weight:600;">MATCH</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # stats + radar chart side by side
                stat_col, radar_col = st.columns([3, 2])

                with stat_col:
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                    available_stats = [s for s in key_stats if s in row.index]
                    cols = st.columns(len(available_stats)) if available_stats else []
                    for idx, stat in enumerate(available_stats):
                        val = round(float(row[stat]), 1) if pd.notna(row[stat]) else 'N/A'
                        label = STAT_NAMES.get(stat, stat)
                        with cols[idx]:
                            st.markdown(f"""
                            <div class="stat-box">
                                <div class="stat-value">{val}</div>
                                <div class="stat-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                with radar_col:
                    if team_avg is not None:
                        fig = make_radar_chart(row, team_avg, key_stats)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"radar_{rank}")

                st.markdown("<br>", unsafe_allow_html=True)

            # ── COMPARISON BAR CHART ──────────────────
            st.markdown('<div class="section-header">📊 Top Targets Comparison</div>', unsafe_allow_html=True)

            if key_stats and len(results) > 0:
                primary_stat = key_stats[0]
                if primary_stat in results.columns:
                    fig_bar = px.bar(
                        results,
                        x='Player',
                        y=primary_stat,
                        color='similarity',
                        color_continuous_scale=['#7b2ff7', '#00d2ff'],
                        title=f"{STAT_NAMES.get(primary_stat, primary_stat)} Comparison",
                        labels={primary_stat: STAT_NAMES.get(primary_stat, primary_stat)}
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font=dict(color='white', size=16),
                        xaxis=dict(color='#666', gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(color='#666', gridcolor='rgba(255,255,255,0.05)'),
                        coloraxis_showscale=False,
                        margin=dict(t=50, b=20)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("❌ No players found for this team/position combination.")

# ── FOOTER ────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="footer">
    Built with K-Means Clustering • Cosine Similarity • Streamlit & Plotly<br>
    2024/25 Big 5 European Leagues • Premier League • La Liga • Serie A • Bundesliga • Ligue 1
</div>
""", unsafe_allow_html=True)
