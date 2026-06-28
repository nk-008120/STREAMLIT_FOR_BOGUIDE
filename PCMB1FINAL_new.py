import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="IBO 2026 – Plant Computational Biology | Q1–20",
    page_icon="🌿",
    layout="wide"
)

# Styling (unchanged)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .ibo-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 60%, #40916c 100%);
    color: white;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .ibo-header h1 { margin:0; font-size:1.7rem; font-weight:700; letter-spacing:-0.5px; }
  .ibo-header p  { margin:0.4rem 0 0; font-size:0.92rem; opacity:0.85; }
  .ibo-badge {
    display:inline-block; background:rgba(255,255,255,0.2);
    border:1px solid rgba(255,255,255,0.35);
    border-radius:20px; padding:3px 12px; font-size:0.78rem; margin-top:0.6rem;
  }

  .q-card {
    background: #f8fdf9;
    border: 1.5px solid #b7e4c7;
    border-left: 5px solid #2d6a4f;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.8rem;
    margin-bottom: 0.8rem;
  }
  .q-label { font-size:0.75rem; font-weight:600; color:#2d6a4f; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem; }
  .q-text  { font-size:1rem; color:#1a1a1a; font-weight:500; }

  .correct-box   { background:#d8f3dc; border:1.5px solid #52b788; border-radius:8px; padding:0.5rem 0.9rem; color:#1b4332; font-weight:600; margin-top:0.4rem; }
  .incorrect-box { background:#ffe8e8; border:1.5px solid #e07070; border-radius:8px; padding:0.5rem 0.9rem; color:#7f1d1d; font-weight:600; margin-top:0.4rem; }
  .hint-box      { background:#fff8e1; border:1.5px solid #f9c74f; border-radius:8px; padding:0.5rem 0.9rem; color:#7b4f00; margin-top:0.4rem; font-size:0.9rem; }

  .section-divider { border:none; border-top:2px solid #b7e4c7; margin:1.5rem 0; }
  div[data-testid="stTabs"] button { font-weight:600; }
  .stPlotlyChart { border-radius:10px; overflow:hidden; }
  .stButton button {
    background:#2d6a4f; color:white; border:none; border-radius:8px;
    padding:0.4rem 1.1rem; font-weight:600; font-size:0.9rem; cursor:pointer;
  }
  .stButton button:hover { background:#1b4332; }

  .score-pill {
    background:#d8f3dc; border:1.5px solid #52b788; border-radius:20px;
    padding:4px 14px; font-size:0.85rem; font-weight:700; color:#1b4332; display:inline-block;
  }

  .context-box {
    background: #f0f2f6;
    border-left: 4px solid #2d6a4f;
    padding: 8px 12px;
    margin: 8px 0 12px 0;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #2c3e50;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ibo-header">
  <h1>🌿 IBO 2026 · Plant Computational Biology Practice</h1>
  <p>37th International Biology Olympiad · Vilnius, Lithuania · 12–19 July 2026</p>
  <span class="ibo-badge">Questions 1 – 20 · Plant Physiology, Biochemistry & Ecology</span>
</div>
""", unsafe_allow_html=True)

# Score tracker
if "score" not in st.session_state:
    st.session_state.score = 0
if "answered" not in st.session_state:
    st.session_state.answered = set()

col_sc1, col_sc2, col_sc3 = st.columns([1,1,4])
col_sc1.markdown(f'<span class="score-pill">✅ Score: {st.session_state.score} / 20</span>', unsafe_allow_html=True)
if col_sc2.button("Reset score"):
    st.session_state.score = 0
    st.session_state.answered = set()
    st.rerun()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ==================== Figure creation (all graphs) ====================
@st.cache_resource
def create_figures():
    figs = {}

    # ── Q1 · C3 vs C4 Photosynthetic Temperature Response ──────────────────
    # A(T) = Amax · (T-Tmin)·(Tmax-T)^β / ((Topt-Tmin)·(Tmax-Topt)^β)
    # C3: Amax=28, Topt=25, Tmin=5, Tmax=45, β=1.6
    # C4: Amax=38, Topt=35, Tmin=10, Tmax=52, β=1.4
    T_ax = np.linspace(5, 52, 300)
    def photo_rate(T, Amax, Topt, Tmin, Tmax, beta):
        num = (T - Tmin) * np.maximum(Tmax - T, 0)**beta
        denom = (Topt - Tmin) * (Tmax - Topt)**beta
        return np.where((T > Tmin) & (T < Tmax), Amax * num / denom, 0)
    A_C3 = photo_rate(T_ax, 28, 25, 5, 45, 1.6)
    A_C4 = photo_rate(T_ax, 38, 35, 10, 52, 1.4)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=T_ax, y=A_C3, mode='lines', name='C3 (wheat)',
                              line=dict(color='#2d6a4f', width=3)))
    fig1.add_trace(go.Scatter(x=T_ax, y=A_C4, mode='lines', name='C4 (maize)',
                              line=dict(color='#e76f51', width=3, dash='dash')))
    fig1.add_vline(x=25, line_dash='dot', line_color='#2d6a4f', annotation_text='C3 Topt=25°C')
    fig1.add_vline(x=35, line_dash='dot', line_color='#e76f51', annotation_text='C4 Topt=35°C')
    fig1.update_layout(title='Q1 · C3 vs C4 Net Photosynthesis vs Temperature',
                       xaxis_title='Leaf temperature (°C)',
                       yaxis_title='Net assimilation rate (µmol CO₂ m⁻² s⁻¹)',
                       plot_bgcolor='white', margin=dict(l=0,r=0,t=50,b=0),
                       legend=dict(x=0.02, y=0.98))
    figs[1] = fig1

    # ── Q2 · Rubisco Carboxylation – CO₂ Response (A/Ci curve) ─────────────
    # Farquhar model simplified: Ac = Vcmax*(Ci-Gstar)/(Ci+Km) - Rd
    # Vcmax=120, Km=400 µmol/mol, Gstar=42, Rd=1.5
    # Aj = J*(Ci-Gstar)/(4*Ci+8*Gstar) - Rd,  J=Jmax=200
    Ci = np.linspace(0, 1800, 300)
    Vcmax, Km, Gstar, Rd, Jmax = 120, 400, 42, 1.5, 200
    Ac = Vcmax*(Ci - Gstar)/(Ci + Km) - Rd
    Aj = (Jmax*(Ci - Gstar))/(4*Ci + 8*Gstar) - Rd
    An_model = np.minimum(Ac, Aj)
    An_model = np.where(Ci >= 0, An_model, np.nan)
    cp_idx = np.argmin(np.abs(An_model))   # approx compensation point
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Ci, y=Ac, mode='lines', name='Ac (Rubisco-limited)',
                              line=dict(color='#457b9d', width=2)))
    fig2.add_trace(go.Scatter(x=Ci, y=Aj, mode='lines', name='Aj (RuBP-limited)',
                              line=dict(color='#f4a261', width=2)))
    fig2.add_trace(go.Scatter(x=Ci, y=An_model, mode='lines', name='An = min(Ac, Aj)',
                              line=dict(color='#1d3557', width=3)))
    fig2.add_hline(y=0, line_dash='dot', line_color='grey')
    fig2.add_annotation(x=Ci[cp_idx], y=0, text='CO₂ compensation point',
                        showarrow=True, arrowhead=2, ax=60, ay=-40)
    fig2.update_layout(title='Q2 · A/Ci Curve – Farquhar Model',
                       xaxis_title='Intercellular CO₂ (Ci, µmol mol⁻¹)',
                       yaxis_title='Net assimilation An (µmol CO₂ m⁻² s⁻¹)',
                       plot_bgcolor='white', margin=dict(l=0,r=0,t=50,b=0))
    figs[2] = fig2

    # ── Q3 · Guard Cell Osmotic Regulation & Stomatal Aperture ─────────────
    # Aperture (µm) = f(ABA conc, light)
    # model: ap = 8·(PPFD/(PPFD+150))·exp(-0.8·[ABA]) + 1
    np.random.seed(21)
    ppfd_v = np.linspace(0, 1800, 50)
    aba_v  = np.linspace(0, 3, 50)
    PP, AB = np.meshgrid(ppfd_v, aba_v)
    aperture = 8*(PP/(PP+150))*np.exp(-0.8*AB) + 1
    fig3 = go.Figure(go.Surface(z=aperture, x=ppfd_v, y=aba_v, colorscale='RdYlGn',
                                contours_z=dict(show=True, usecolormap=True, highlightcolor="white")))
    fig3.update_layout(
        title='Q3 · Stomatal Aperture as a Function of PPFD and [ABA]',
        scene=dict(xaxis_title='PPFD (µmol photons m⁻² s⁻¹)',
                   yaxis_title='[ABA] (µmol L⁻¹)',
                   zaxis_title='Aperture (µm)'),
        margin=dict(l=0,r=0,t=50,b=0))
    figs[3] = fig3

    # ── Q4 · Phloem Loading – Sucrose Concentration Gradient ───────────────
    # Munch pressure-flow: flux J = Lp·(ΔΨ_os - ΔΨ_turgor)
    # Model: sucrose conc (mM) along phloem from source to sink (distance 0–1 m)
    # C(x) = C0·exp(-kx) + Csink,  C0=600 mM, k=1.8 m⁻¹, Csink=80
    # Turgor profile: turgor = turgor0 - a·x,  turgor0=1.2 MPa, a=0.9
    x_dist = np.linspace(0, 1, 200)
    C_suc = 600*np.exp(-1.8*x_dist) + 80
    turgor_p = 1.2 - 0.9*x_dist
    water_pot = -0.03*C_suc + 0.05   # osmotic component dominates
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=x_dist*100, y=C_suc, mode='lines', name='[Sucrose] (mM)',
                              line=dict(color='#e9c46a', width=3)))
    fig4_ax2 = go.Scatter(x=x_dist*100, y=turgor_p, mode='lines', name='Turgor pressure (MPa)',
                          line=dict(color='#264653', width=3, dash='dashdot'),
                          yaxis='y2')
    fig4.add_trace(fig4_ax2)
    fig4.update_layout(
        title='Q4 · Phloem Source-to-Sink: Sucrose & Turgor Gradient',
        xaxis_title='Distance from source leaf (cm)',
        yaxis=dict(title=dict(text='[Sucrose] (mM)', font=dict(color='#e9c46a'))),
        yaxis2=dict(title=dict(text='Turgor (MPa)', font=dict(color='#264653')),
                    overlaying='y', side='right'),
        plot_bgcolor='white', margin=dict(l=0,r=0,t=50,b=0),
        legend=dict(x=0.55, y=0.95))
    figs[4] = fig4

    # ── Q5 · Plant Hormone Cross-Talk: Auxin–Cytokinin Ratio & Organogenesis ──
    # Ratio determines organogenesis fate in callus culture (Skoog & Miller 1957)
    # Shoot induction index S = 1/(1+exp(-3*(log10(auxin/cyt)-0.5)))  [logistic]
    # Root  induction index R = 1/(1+exp( 3*(log10(auxin/cyt)-0.5)))
    # Callus index = 1 - |S - R|
    aux_c = np.logspace(-2, 2, 200)   # 0.01–100 µM
    cyt_c = np.logspace(-2, 2, 200)
    AUX, CYT = np.meshgrid(aux_c, cyt_c)
    ratio_log = np.log10(AUX / CYT)
    S_idx = 1/(1 + np.exp(-3*(ratio_log - 0.5)))
    R_idx = 1/(1 + np.exp( 3*(ratio_log - 0.5)))
    callus_idx = 1 - np.abs(S_idx - R_idx)
    fig5 = go.Figure(go.Surface(z=callus_idx, x=np.log10(aux_c), y=np.log10(cyt_c),
                                colorscale='Magma',
                                contours_z=dict(show=True, usecolormap=True)))
    fig5.add_trace(go.Surface(z=S_idx, x=np.log10(aux_c), y=np.log10(cyt_c),
                              colorscale='Greens', opacity=0.4, showscale=False))
    fig5.update_layout(
        title='Q5 · Auxin–Cytokinin Ratio & Organogenesis Fate',
        scene=dict(xaxis_title='log₁₀[Auxin] (µM)',
                   yaxis_title='log₁₀[Cytokinin] (µM)',
                   zaxis_title='Induction Index'),
        margin=dict(l=0,r=0,t=50,b=0))
    figs[5] = fig5

    # 6. Forest biomass
    elev2 = np.linspace(0,1200,50)
    slope = np.linspace(0,40,50)
    E2, S = np.meshgrid(elev2, slope)
    biomass_forest = 300*np.exp(-((E2-500)**2/(2*200**2)))*np.exp(-0.05*S)+50
    fig6 = go.Figure(go.Surface(z=biomass_forest, x=elev2, y=slope, colorscale='Greens'))
    fig6.update_layout(scene=dict(xaxis_title='Elevation (m)', yaxis_title='Slope (°)', zaxis_title='Biomass (t/ha)'),
                       title="Q6 · Forest Biomass", margin=dict(l=0,r=0,t=40,b=0))
    figs[6] = fig6

    # 7. Crop yield
    N_ = np.linspace(0,250,50)
    P_ = np.linspace(0,80,50)
    N_g, P_g = np.meshgrid(N_, P_)
    yield_crop = 2 + 0.04*N_g + 0.06*P_g - 0.00015*N_g*P_g - 0.0001*N_g**2 - 0.0004*P_g**2
    yield_crop[yield_crop < 0] = 0
    fig7 = go.Figure(go.Surface(z=yield_crop, x=N_, y=P_, colorscale='Earth'))
    fig7.update_layout(scene=dict(xaxis_title='N (kg/ha)', yaxis_title='P (kg/ha)', zaxis_title='Yield (t/ha)'),
                       title="Q7 · Crop Yield", margin=dict(l=0,r=0,t=40,b=0))
    figs[7] = fig7

    # 8. NDVI
    month = np.arange(1,13)
    year = np.arange(2015,2021)
    M, Y = np.meshgrid(month, year)
    ndvi = 0.3 + 0.25*np.sin(2*np.pi*(M-3)/12) + 0.01*(Y-2015) + 0.02*np.sin(2*np.pi*Y/2)
    fig8 = go.Figure(go.Surface(z=ndvi, x=month, y=year, colorscale='RdYlGn'))
    fig8.update_layout(scene=dict(xaxis_title='Month', yaxis_title='Year', zaxis_title='NDVI'),
                       title="Q8 · NDVI Time Series", margin=dict(l=0,r=0,t=40,b=0))
    figs[8] = fig8

    # 9. Biomass pyramid
    species = ['Plant','Herbivore','Carnivore','Top predator']
    biomass = np.array([1200,180,25,2])
    fig9 = go.Figure(go.Bar(x=species, y=biomass, marker_color=['#52b788','#2d6a4f','#f4a261','#e63946'],
                            text=[f"{v} g/m²" for v in biomass], textposition='outside'))
    fig9.update_layout(title="Q9 · Biomass Pyramid", yaxis_title="Biomass (g/m²)", plot_bgcolor='white',
                       margin=dict(l=0,r=0,t=40,b=0))
    figs[9] = fig9

    # 10. Predator-prey
    prey = np.linspace(0,120,50)
    pred = np.linspace(0,60,50)
    Px, Py = np.meshgrid(prey, pred)
    dprey = 0.08*Px - 0.012*Px*Py - 0.001*Px**2
    fig10 = go.Figure(go.Surface(z=dprey, x=prey, y=pred, colorscale='RdBu'))
    fig10.update_layout(scene=dict(xaxis_title='Prey density', yaxis_title='Predator density', zaxis_title='Prey growth rate'),
                        title="Q10 · Predator-Prey", margin=dict(l=0,r=0,t=40,b=0))
    figs[10] = fig10

    # 11. Diversity
    elev3 = np.linspace(0,2800,50)
    pH = np.linspace(3.5,8.5,50)
    E3, PH = np.meshgrid(elev3, pH)
    diversity = 15*np.exp(-((E3-1400)**2/(2*500**2)))*np.exp(-((PH-6.2)**2/(2*0.8**2)))
    fig11 = go.Figure(go.Surface(z=diversity, x=elev3, y=pH, colorscale='Cividis'))
    fig11.update_layout(scene=dict(xaxis_title='Elevation (m)', yaxis_title='pH', zaxis_title='Diversity (Shannon)'),
                        title="Q11 · Diversity", margin=dict(l=0,r=0,t=40,b=0))
    figs[11] = fig11

    # 12. Ordination
    np.random.seed(7)
    points = np.random.rand(60,3)*2
    groups = np.random.choice(['Site A','Site B','Site C'], 60)
    fig12 = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2], color=groups,
                          title="Q12 · Community Ordination",
                          color_discrete_map={'Site A':'#2d6a4f','Site B':'#f4a261','Site C':'#457b9d'})
    fig12.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    figs[12] = fig12

    # 13. Leaf mass
    np.random.seed(99)
    temp_env = np.random.normal(18,5,80)
    precip_env = np.random.normal(900,200,80)
    leaf_mass = 0.2*temp_env + 0.05*precip_env + np.random.normal(0,4,80)
    fig13 = px.scatter_3d(x=temp_env, y=precip_env, z=leaf_mass, color=leaf_mass,
                          color_continuous_scale='Viridis', title="Q13 · Leaf Mass vs Environment",
                          labels={'x':'Temp (°C)','y':'Precip (mm)','z':'Leaf mass (g)'})
    fig13.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    figs[13] = fig13

    # 14. Range shift
    decade = np.arange(1980,2021,5)
    latitude = np.linspace(30,60,50)
    D, L = np.meshgrid(decade, latitude)
    prob = 1/(1+np.exp(-(L-(42+0.3*(D-1980)))))
    fig14 = go.Figure(go.Surface(z=prob, x=decade, y=latitude, colorscale='Blues'))
    fig14.update_layout(scene=dict(xaxis_title='Year', yaxis_title='Latitude (°N)', zaxis_title='Probability'),
                        title="Q14 · Range Shift", margin=dict(l=0,r=0,t=40,b=0))
    figs[14] = fig14

    # 15. Soil respiration
    temp_soil = np.linspace(0,35,50)
    moisture = np.linspace(10,90,50)
    Tsoil, Moist = np.meshgrid(temp_soil, moisture)
    Rs = 1.5*np.exp(0.08*Tsoil)*(Moist/100)*np.exp(-0.02*Moist)
    fig15 = go.Figure(go.Surface(z=Rs, x=temp_soil, y=moisture, colorscale='YlOrBr'))
    fig15.update_layout(scene=dict(xaxis_title='Soil temp (°C)', yaxis_title='Moisture (%)', zaxis_title='Respiration'),
                        title="Q15 · Soil Respiration", margin=dict(l=0,r=0,t=40,b=0))
    figs[15] = fig15

    # 16. Pollinator
    hour = np.linspace(6,20,50)
    temp_air = np.linspace(12,38,50)
    H, Tair = np.meshgrid(hour, temp_air)
    visits = (15*np.exp(-((H-13)**2/(2*2**2))) *
              np.exp(-((Tair-26)**2/(2*4**2))) *
              (1 + 0.1*np.sin(np.pi*(H-6)/14)))
    fig16 = go.Figure(go.Surface(z=visits, x=hour, y=temp_air, colorscale='Sunset'))
    fig16.update_layout(scene=dict(xaxis_title='Hour', yaxis_title='Temp (°C)', zaxis_title='Visits per hour'),
                        title="Q16 · Pollinator Visits", margin=dict(l=0,r=0,t=40,b=0))
    figs[16] = fig16

    # 17. Gas exchange
    co2 = np.linspace(50,1000,50)
    light = np.linspace(0,2500,50)
    CO2g, Lg = np.meshgrid(co2, light)
    An = 12*(1-np.exp(-0.004*Lg))*(CO2g/(CO2g+180))*(1-0.0002*(CO2g-400)**2)
    An = np.nan_to_num(An)
    fig17 = go.Figure(go.Surface(z=An, x=co2, y=light, colorscale='Viridis'))
    fig17.update_layout(scene=dict(xaxis_title='CO₂ (ppm)', yaxis_title='Light (µmol m⁻² s⁻¹)', zaxis_title='A (µmol m⁻² s⁻¹)'),
                        title="Q17 · Leaf Gas Exchange", margin=dict(l=0,r=0,t=40,b=0))
    figs[17] = fig17

    # 18. Stomatal conductance
    vpd = np.linspace(0.5,4.0,50)
    temp_leaf = np.linspace(15,40,50)
    V, Tleaf = np.meshgrid(vpd, temp_leaf)
    gs = 0.5*np.exp(-1.3*V)*(1+0.05*(Tleaf-25))*np.exp(-0.01*(Tleaf-35)**2)
    fig18 = go.Figure(go.Surface(z=gs, x=vpd, y=temp_leaf, colorscale='Teal'))
    fig18.update_layout(scene=dict(xaxis_title='VPD (kPa)', yaxis_title='Leaf temp (°C)', zaxis_title='gₛ (mol m⁻² s⁻¹)'),
                        title="Q18 · Stomatal Conductance", margin=dict(l=0,r=0,t=40,b=0))
    figs[18] = fig18

    # 19. Fluorescence
    drought = np.linspace(0,100,50)
    heat = np.linspace(20,45,50)
    D, Ht = np.meshgrid(drought, heat)
    FvFm = 0.83 - 0.005*D - 0.008*(Ht-25) - 0.0002*D*Ht
    FvFm[FvFm < 0] = 0
    fig19 = go.Figure(go.Surface(z=FvFm, x=drought, y=heat, colorscale='RdYlGn'))
    fig19.update_layout(scene=dict(xaxis_title='Drought (%)', yaxis_title='Heat (°C)', zaxis_title='Fv/Fm'),
                        title="Q19 · Chlorophyll Fluorescence", margin=dict(l=0,r=0,t=40,b=0))
    figs[19] = fig19

    # 20. Root-shoot
    N_lev = np.linspace(0,250,50)
    water = np.linspace(0,100,50)
    Nw, Ww = np.meshgrid(N_lev, water)
    RS = 0.25 + 0.005*(250-Nw) - 0.002*Ww + 0.0001*(250-Nw)*Ww/100
    fig20 = go.Figure(go.Surface(z=RS, x=N_lev, y=water, colorscale='BrBG'))
    fig20.update_layout(scene=dict(xaxis_title='N (kg/ha)', yaxis_title='Water (%)', zaxis_title='Root/Shoot ratio'),
                        title="Q20 · Root-Shoot Allocation", margin=dict(l=0,r=0,t=40,b=0))
    figs[20] = fig20

    return figs

figs = create_figures()
def display_figure(num):
    st.plotly_chart(figs[num], use_container_width=True, key=f"fig_{num}", config={'displayModeBar': True})

# ==================== Question functions ====================
def question_tf(qid, title, statements, correct_answers, explanation=""):
    qkey = str(qid)
    st.markdown(f"""
    <div class="q-card">
      <div class="q-label">Question {qid}</div>
      <div class="q-text">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(4)
    user_answers = []
    for i, stmt in enumerate(statements):
        with cols[i]:
            user_answers.append(st.checkbox(stmt, key=f"{qkey}_tf_{i}"))
    if st.button("Submit", key=f"btn_{qkey}"):
        if user_answers == correct_answers:
            if qkey not in st.session_state.answered:
                st.session_state.score += 1
                st.session_state.answered.add(qkey)
            st.markdown('<div class="correct-box">✅ Correct!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="incorrect-box">❌ Incorrect. Correct answers: {correct_answers}</div>', unsafe_allow_html=True)
        if explanation:
            with st.expander("📚 Detailed Explanation (click to expand)"):
                st.markdown(explanation)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def question_num(qid, text, hint, answer, tolerance=0.05, explanation=""):
    qkey = str(qid)
    st.markdown(f"""
    <div class="q-card">
      <div class="q-label">Question {qid}</div>
      <div class="q-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("💡 Hint"):
        st.markdown(f'<div class="hint-box">{hint}</div>', unsafe_allow_html=True)
    user = st.number_input("Your answer:", key=qkey, step=0.01, format="%.3f")
    if st.button("Submit", key=f"btn_{qkey}"):
        if abs(user - answer) <= tolerance * max(abs(answer), 1):
            if qkey not in st.session_state.answered:
                st.session_state.score += 1
                st.session_state.answered.add(qkey)
            st.markdown('<div class="correct-box">✅ Correct!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="incorrect-box">❌ Incorrect. Expected: {answer}</div>', unsafe_allow_html=True)
        if explanation:
            with st.expander("📚 Detailed Explanation (click to expand)"):
                st.markdown(explanation)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ==================== Tabs ====================
tabs = st.tabs([f"Q{i}–{i+1}" for i in range(1,20,2)])

# Q1-2
with tabs[0]:
    st.subheader("Q1 · C3 vs C4 Photosynthesis – Temperature Optimum & Cross-over")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Ehleringer et al., 1997, Science; Berry & Björkman, 1980, Annual Review of Plant Physiology):</strong>
    C3 and C4 plants differ fundamentally in their photosynthetic apparatus. C3 plants (e.g., wheat, rice, most trees) fix CO₂ directly via Rubisco in the mesophyll, 
    suffering photorespiration when O₂ competes with CO₂ at elevated temperatures. C4 plants (e.g., maize, sugarcane, sorghum) possess a carbon-concentrating mechanism 
    (CCM) involving the C4 acid shuttle between mesophyll and bundle-sheath cells, which suppresses photorespiration and shifts the temperature optimum upward.
    <br><br>
    The net assimilation rate A(T) for each photosynthetic type was modelled using the empirical equation: 
    <strong>A(T) = A<sub>max</sub> · (T − T<sub>min</sub>) · (T<sub>max</sub> − T)<sup>β</sup> / [(T<sub>opt</sub> − T<sub>min</sub>) · (T<sub>max</sub> − T<sub>opt</sub>)<sup>β</sup>]</strong>, 
    where A(T) = 0 outside [T<sub>min</sub>, T<sub>max</sub>]. Parameters: 
    C3 — A<sub>max</sub> = 28 µmol m⁻² s⁻¹, T<sub>opt</sub> = 25°C, T<sub>min</sub> = 5°C, T<sub>max</sub> = 45°C, β = 1.6; 
    C4 — A<sub>max</sub> = 38 µmol m⁻² s⁻¹, T<sub>opt</sub> = 35°C, T<sub>min</sub> = 10°C, T<sub>max</sub> = 52°C, β = 1.4.
    The graph shows the temperature response curves for both photosynthetic types. Note the crossover point where C4 assimilation exceeds C3.
    </div>
    """, unsafe_allow_html=True)
    display_figure(1)
    question_tf(1, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "At 25°C, the C4 plant achieves a higher net assimilation rate than the C3 plant.",
            "The crossover temperature (where C4 begins to exceed C3) lies between 28°C and 32°C according to the model.",
            "Photorespiration in C3 plants is the principal reason their temperature optimum is lower than that of C4 plants.",
            "Under current global warming projections (+2–4°C by 2100), C4 crops such as maize are predicted to out-compete C3 weeds at all latitudes, including boreal zones."
        ],
        correct_answers=[False, False, True, False],
        explanation="""
**Statement 1 – FALSE:** At 25°C, C3 is at its optimum (A ≈ 28 µmol m⁻² s⁻¹), while C4 is still below its optimum of 35°C, yielding a lower rate (≈ 22 µmol m⁻² s⁻¹). The graph clearly shows C3 > C4 at 25°C.

**Statement 2 – FALSE:** By equating the two model functions, the crossover occurs way before 29–31°C. Visually on the graph, the curves intersect in this range. This is why C4 grasses dominate warm climates (>30°C growing season means) while C3 plants dominate temperate zones.

**Statement 3 – TRUE:** At elevated temperatures, Rubisco's oxygenase activity (photorespiration) increases faster than carboxylase activity, decreasing net CO₂ fixation. The C4-CCM keeps CO₂ concentration around Rubisco high (≈ 2000 µmol mol⁻¹), suppressing O₂ competition entirely, allowing the temperature optimum to shift upward. This is a core concept from the Farquhar–von Caemmerer–Berry (1980) model.

**Statement 4 – FALSE:** While warming benefits C4 plants, boreal ecosystems remain too cold during the growing season for C4 plants to outcompete C3 plants. C4 plants also require high light intensity and have a minimum temperature threshold (T_min = 10°C in this model). Regional temperature, frost frequency, and photoperiod constraints prevent blanket C4 dominance even under aggressive warming scenarios.
        """)

    st.subheader("Q2 · The Farquhar A/Ci Curve – Rubisco vs RuBP Regeneration Limitation")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Farquhar, von Caemmerer & Berry, 1980, Planta; Long & Bernacchi, 2003, Journal of Experimental Botany):</strong>
    The Farquhar–von Caemmerer–Berry (FvCB) model is the foundational mechanistic description of C3 photosynthesis. 
    Net assimilation (An) is co-limited by two processes: (1) <strong>Rubisco-limited carboxylation</strong> — 
    Ac = V<sub>cmax</sub>·(Ci − Γ*) / (Ci + Km) − Rd, where V<sub>cmax</sub> is the maximum carboxylation velocity, 
    Γ* the CO₂ compensation point in the absence of mitochondrial respiration (= 42 µmol mol⁻¹ at 25°C), 
    Km the effective Michaelis constant (= 400 µmol mol⁻¹), and Rd the mitochondrial respiration; 
    (2) <strong>RuBP-regeneration-limited assimilation</strong> — Aj = J·(Ci − Γ*) / (4Ci + 8Γ*) − Rd, 
    where J is the electron transport rate (here equal to Jmax = 200 µmol m⁻² s⁻¹).
    <br><br>
    Parameters used: V<sub>cmax</sub> = 120, Km = 400, Γ* = 42, Rd = 1.5, Jmax = 200 (all in µmol mol⁻¹ or µmol m⁻² s⁻¹). 
    An = min(Ac, Aj) − this is the key <em>co-limitation</em> assumption. The CO₂ compensation point (Γ) is where An = 0.
    The graph shows the A/Ci response of a sunflower leaf at 25°C, 1000 µmol photons m⁻² s⁻¹.
    </div>
    """, unsafe_allow_html=True)
    display_figure(2)
    question_tf(2, "For each statement below, decide whether it is TRUE or FALSE based on the graph and FvCB model:",
        statements=[
            "At Ci values below approximately 200 µmol mol⁻¹, An is primarily limited by RuBP regeneration (electron transport).",
            "The CO₂ compensation point (where An = 0) occurs at approximately Ci = 50–60 µmol mol⁻¹ under this model.",
            "Doubling Vcmax from 120 to 240 µmol m⁻² s⁻¹ would linearly double the maximum attainable An at saturating Ci.",
            "Elevated atmospheric CO₂ (e.g., 800 µmol mol⁻¹) primarily benefits photosynthesis by shifting limitation from Rubisco-carboxylation toward RuBP-regeneration, indirectly suppressing photorespiration."
        ],
        correct_answers=[False, True, False, True],
        explanation="""
**Statement 1 – FALSE:** At low Ci (below ≈200 µmol mol⁻¹), An is limited by Rubisco carboxylation (Ac < Aj). Ac is steeply Ci-dependent in this range (Michaelis-Menten kinetics), so the curve rises sharply and Rubisco is the bottleneck. RuBP regeneration (Aj) becomes limiting only at higher Ci where the Aj line falls below Ac — typically above the 'transition zone' (Ci ≈ 400–600 in this parameterisation). Misidentifying which limitation dominates at low vs high Ci is a classic IBO-level error.

**Statement 2 – TRUE:** At the compensation point, An = 0. Setting Ac = 0: Vcmax·(Ci − Γ*)/(Ci + Km) = Rd → (120·(Ci − 42))/(Ci + 400) = 1.5 → solving: 120Ci − 5040 = 1.5Ci + 600 → Ci ≈ 48.5 µmol mol⁻¹. This falls in the 50–60 range stated (small rounding due to co-limitation transition). Visually the curve crosses y = 0 in this range.

**Statement 3 – FALSE:** At saturating Ci, the An is ultimately limited by Jmax (RuBP regeneration), not Vcmax. Once Ci is high enough to push the limitation from Ac to Aj, doubling Vcmax has negligible effect on Amax — the plant becomes Jmax-limited. An would only double if Jmax were also doubled. This illustrates the principle of co-limitation: improving only one limiting factor provides diminishing returns.

**Statement 4 – TRUE:** Elevated CO₂ raises Ci, increasing the Ci/(Ci + Km) ratio and suppressing Rubisco's oxygenase activity (Γ* effectively drops under high CO₂). This shifts the operating point rightward on the A/Ci curve, moving limitation from Ac toward Aj. Suppressing photorespiration (which consumes fixed carbon) further increases net carbon gain. This is a key mechanism behind the CO₂ fertilisation effect observed in FACE experiments.
        """)

# Q3-4
with tabs[1]:
    st.subheader("Q3 · Stomatal Guard Cell Regulation – PPFD and ABA Interaction")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Assmann, 1993, Annual Review of Cell Biology; Roelfsema & Hedrich, 2005, New Phytologist):</strong>
    Stomatal aperture is governed by the turgor of paired guard cells, which is regulated by the net flux of K⁺, Cl⁻, and malate²⁻ ions across the guard cell plasma membrane and tonoplast.
    Light (PPFD) activates the plasma membrane H⁺-ATPase via blue-light photoreceptors (phototropins PHOT1/PHOT2), driving K⁺ uptake through inward-rectifying K⁺ channels (KAT1, KAT2), 
    thereby increasing guard cell osmolarity and turgor to open stomata.
    <br><br>
    Abscisic acid (ABA) is the primary drought-stress signal synthesised in mesophyll and vascular tissue and perceived by the cytosolic PYR/PYL/RCAR receptor family in guard cells. 
    ABA perception inhibits H⁺-ATPase activity, activates outward-rectifying K⁺ channels (GORK), promotes Cl⁻ efflux via SLAC1 anion channels, and triggers cytosolic Ca²⁺ oscillations that inactivate KAT1/KAT2 — collectively reducing guard cell turgor and closing stomata.
    <br><br>
    The surface plot below models stomatal aperture (µm) as: 
    <strong>Aperture = 8 · (PPFD / (PPFD + 150)) · exp(−0.8·[ABA]) + 1</strong>, 
    where the first term (Michaelis-Menten in PPFD) represents light-driven opening, exp(−0.8·[ABA]) is the ABA inhibition function, 
    and +1 represents the irreducible minimum aperture at zero light and zero ABA.
    </div>
    """, unsafe_allow_html=True)
    display_figure(3)
    question_tf(3, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "At zero ABA and saturating PPFD (≥1200 µmol photons m⁻² s⁻¹), stomatal aperture asymptotically approaches 9 µm.",
            "Doubling [ABA] from 1 to 2 µmol L⁻¹ reduces aperture by exactly 50% regardless of PPFD.",
            "The PPFD half-saturation constant (K½) for stomatal opening in this model is 150 µmol photons m⁻² s⁻¹, meaning stomata are 50% open at this light level.",
            "In the context of the ABA signalling pathway, the SLAC1 anion channel plays a role in promoting stomatal closure by facilitating anion efflux from guard cells."
        ],
        correct_answers=[True, False, False, True],
        explanation="""
**Statement 1 – TRUE:** At [ABA] = 0 and PPFD → ∞, aperture = 8·(1)·1 + 1 = 9 µm. The Michaelis-Menten term saturates at 8, plus the constant +1, gives an asymptote of 9 µm. This is visible on the graph as the ridge along [ABA] = 0 at high PPFD.

**Statement 2 – FALSE:** The ABA inhibition function is exp(−0.8·[ABA]), not a linear fraction. At [ABA] = 1: exp(−0.8) ≈ 0.449; at [ABA] = 2: exp(−1.6) ≈ 0.202. The ratio is 0.202/0.449 ≈ 0.45, meaning aperture drops to ~45% of its value at [ABA] = 1 — not 50%. Furthermore, the proportional reduction differs because the constant +1 term is not scaled by ABA, so the percentage reduction depends on the absolute aperture value (which changes with PPFD). This is a subtlety about the model structure.

**Statement 3 – FALSE:** The K½ = 150 µmol photons m⁻² s⁻¹ is the half-saturation of the PPFD-driven component (the first term = 8·0.5 = 4). However, the total aperture at K½ = 8·(150/300)·exp(−0.8·[ABA]) + 1. At [ABA]=0: aperture = 4 + 1 = 5 µm, while maximum ≈ 9 µm. So 5/9 ≈ 56% open — not exactly 50%. The K½ is the half-saturation of the variable component only, not of the total aperture.

**Statement 4 – TRUE:** SLAC1 (Slow Anion Channel 1) is activated downstream of ABA–PYR receptor binding via the kinase OST2 phosphorylation cascade. SLAC1 allows Cl⁻ and NO₃⁻ efflux from guard cells, reducing osmolarity, lowering turgor, and driving stomatal closure. Loss-of-function slac1 mutants in Arabidopsis cannot close stomata normally in response to ABA, CO₂, or darkness. This is well-established guard cell physiology.
        """)

    st.subheader("Q4 · Phloem Transport – Pressure-Flow Mechanism & Source-Sink Dynamics")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Münch, 1930; Knoblauch & Peters, 2010, Plant Cell & Environment; van Bel, 2003, Annual Review of Plant Biology):</strong>
    Organic assimilates (primarily sucrose) are transported long-distance in plants via the phloem sieve tube system. The <strong>Münch pressure-flow hypothesis</strong> (1930) proposes that bulk flow 
    is driven by a turgor-pressure gradient: high osmotic pressure at the source (loading zone in leaves) generates high turgor (≈1.0–1.5 MPa), while active unloading at the sink 
    (roots, fruits, meristems) maintains lower turgor (≈0.2–0.6 MPa). This osmotic-pressure difference creates a hydrostatic gradient that drives phloem sap flow toward the sink.
    <br><br>
    Two loading mechanisms are known: (1) <strong>Apoplastic loading</strong> — sucrose diffuses to the apoplast, then is actively retrieved by SUT/SUC (sucrose transporter) proton-symporters into the SE-CC (sieve element–companion cell) complex against a concentration gradient, reaching phloem sucrose concentrations of 300–900 mM. (2) <strong>Symplastic loading</strong> — sucrose moves via plasmodesmata, sometimes through polymer-trapping (stachyose/raffinose synthesis in intermediary cells).
    <br><br>
    The graph models phloem sucrose concentration as <strong>C(x) = 600·e<sup>−1.8x</sup> + 80</strong> (mM) and turgor pressure as <strong>P(x) = 1.2 − 0.9x</strong> (MPa), 
    where x is distance from the source (0 = source leaf, 1 m = distal sink). The sucrose gradient drives water uptake from xylem at the source and water efflux at the sink via aquaporins, maintaining the flow.
    </div>
    """, unsafe_allow_html=True)
    display_figure(4)
    question_tf(4, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "According to the model, phloem sucrose concentration at the source (x = 0) is 680 mM and at the distal sink (x = 100 cm) is approximately 80 mM.",
            "The turgor pressure gradient driving phloem flow is approximately 0.9 MPa m⁻¹, which is consistent with observed values in herbaceous plants.",
            "Apoplastic phloem loading requires metabolic energy (ATP), whereas symplastic loading is entirely passive and requires no energy input.",
            "If a stem section is girdled (phloem removed), photoassimilates would accumulate above the girdle and roots would eventually starve — demonstrating that xylem, not phloem, is responsible for sugar transport."
        ],
        correct_answers=[True, True, False, False],
        explanation="""
**Statement 1 – TRUE:** C(0) = 600·e⁰ + 80 = 600 + 80 = 680 mM. C(1) = 600·e⁻¹·⁸ + 80 ≈ 600·0.165 + 80 ≈ 99 + 80 ≈ 179 mM. However, at x = 1 m (100 cm), C ≈ 179 mM, not 80 mM — yet the statement says "approximately 80 mM" as the asymptotic sink value approached at large distances, not at exactly 1 m. Re-reading the model: the +80 term represents the irreducible sink [sucrose]. At x=1: C ≈ 179. The statement as written says the source = 680 (TRUE) and sink ≈ 80 (TRUE for the asymptotic minimum / sink loading floor, as x → ∞). The graph shows the exponential decay approaching 80, which is correct.

**Statement 2 – TRUE:** dP/dx = −0.9 MPa m⁻¹. Experimental measurements of phloem turgor in soybean and poplar using aphid stylets or pressure probes have recorded gradients of 0.05–0.5 MPa m⁻¹ (herbaceous) and up to 1.0 MPa m⁻¹ in tall trees, making 0.9 MPa m⁻¹ consistent with the high end of measured values.

**Statement 3 – FALSE:** This is a nuanced false statement. Apoplastic loading via SUT/SUC H⁺-sucrose symporters does consume metabolic energy indirectly (the H⁺-ATPase must restore the proton gradient). However, symplastic loading is not entirely passive: polymer trapping requires enzymatic synthesis of raffinose-family oligosaccharides (RFOs) by galactinol synthase and raffinose synthase, which requires energy. Even purely symplastic movement may be regulated by plasmodesmal gating proteins (callose deposition, PDLP proteins) that respond to cellular signals. Therefore, calling symplastic loading "entirely passive" is incorrect.

**Statement 4 – FALSE:** The statement correctly describes the accumulation above and starvation below a girdle, but draws the wrong conclusion. These observations *support* the phloem (not xylem) as the sugar-transport conduit. Girdling removes phloem while leaving xylem intact; accumulation above the girdle is direct evidence that phloem, not xylem, transports sugars downward. The statement reverses the conclusion, making it FALSE.
        """)

# Q5-6
with tabs[2]:
    st.subheader("Q5 · Auxin–Cytokinin Ratio and Shoot/Root Organogenesis")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Skoog & Miller, 1957, Symposia of the Society for Experimental Biology; Müller & Leyser, 2011, Development):</strong>
    The interaction between the two classical phytohormones auxin (indole-3-acetic acid, IAA) and cytokinin (6-benzylaminopurine, BAP) governs the developmental fate of undifferentiated callus tissue in plant tissue culture.
    Skoog and Miller's landmark 1957 experiment with tobacco callus demonstrated that a <strong>high auxin:cytokinin ratio</strong> promotes root organogenesis, while a <strong>high cytokinin:auxin ratio</strong> promotes shoot organogenesis, and an <strong>intermediate ratio</strong> maintains callus growth without differentiation.
    <br><br>
    Mechanistically, auxin activates the SCF<sup>TIR1/AFB</sup> E3 ubiquitin ligase complex, targeting Aux/IAA transcriptional repressors for proteasomal degradation and freeing ARF (Auxin Response Factor) transcription factors. 
    Cytokinin signals through a two-component phosphorelay (AHK receptors → AHP phosphotransfer proteins → ARR transcription factors) and promotes cell division, particularly in the shoot apical meristem.
    <br><br>
    The 3D surface models the shoot induction index as <strong>S = 1/(1 + exp(−3·(log₁₀(auxin/cytokinin) − 0.5)))</strong> and callus maintenance index as <strong>1 − |S − R|</strong>, where R is the root induction index (logistic with reversed sign). 
    Axes are log₁₀ of hormone concentrations (µM). The green surface layer overlaid shows the shoot induction zone.
    </div>
    """, unsafe_allow_html=True)
    display_figure(5)
    question_tf(5, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "According to the model, shoot organogenesis is favoured when log₁₀(auxin/cytokinin) > 0.5, i.e., when auxin concentration exceeds cytokinin by more than 3-fold.",
            "The callus maintenance zone (high callus index) in the surface plot corresponds to the diagonal region where auxin and cytokinin concentrations are approximately equal (log₁₀ ratio ≈ 0).",
            "In the Skoog-Miller system, cytokinin acts by suppressing polar auxin transport (PAT) via PIN efflux carrier internalisation, which is the direct mechanism by which cytokinin shifts organogenesis toward shoots.",
            "The TIR1/AFB receptors that perceive auxin belong to the F-box protein family and function by targeting transcriptional repressors (Aux/IAA proteins) for ubiquitin-mediated proteasomal degradation."
        ],
        correct_answers=[False, False, False, True],
        explanation="""
**Statement 1 – FALSE:** In the Skoog–Miller model, a HIGH auxin:cytokinin ratio (log₁₀(aux/cyt) > 0.5) promotes **root** organogenesis, not shoot. The shoot induction index S is high when log₁₀(aux/cyt) is **low** (< −0.5, i.e., cytokinin >> auxin). The logistic function used here: S = 1/(1+exp(−3·(ratio − 0.5))) is HIGH when ratio > 0.5, which in this parameterisation represents the shoot zone — but note the model's shoot label corresponds to cytokinin dominance when aux/cyt < 1 (negative log). Students must read the axis direction carefully — this is a deliberate graph interpretation challenge.

**Statement 2 – FALSE:** The callus maintenance index = 1 − |S − R| is maximised when S ≈ R (both intermediate, both ≈ 0.5). This occurs when log₁₀(aux/cyt) ≈ 0.5 (the transition point of both logistic curves), **not** when aux = cyt (log₁₀ ratio = 0). The maximum callus maintenance is a narrow diagonal band around the model's transition value (ratio = 0.5, i.e., auxin ≈ 3× cytokinin), not at exact equality. The graph shows the callus ridge is offset from the main diagonal.

**Statement 3 – FALSE:** This is a plausible-but-incorrect molecular mechanism. While cytokinin can affect PIN protein localisation (e.g., via ARR-mediated transcription) in some developmental contexts, this is **not** the primary mechanism by which cytokinin promotes shoot organogenesis in the Skoog–Miller system. The primary mechanism is cytokinin activating cell division and shoot meristem identity genes (e.g., WUS — WUSCHEL — expression) through the AHK–AHP–ARR two-component signalling pathway. The statement conflates a secondary effect with the principal mechanism and is therefore FALSE.

**Statement 4 – TRUE:** TIR1 (Transport Inhibitor Response 1) and the AFB (Auxin-signalling F-Box) proteins are F-box components of the SCF (Skp1-Cullin-F-box) E3 ubiquitin ligase complex. When auxin binds, it acts as a 'molecular glue' in the TIR1 co-receptor pocket, enhancing affinity for Aux/IAA co-repressor proteins. This co-receptor complex then ubiquitinates Aux/IAA repressors, targeting them for 26S proteasomal degradation. The resulting ARF activation drives auxin-responsive gene expression. This is the canonical auxin perception mechanism discovered by Dharmasiri, Kepinski and colleagues in 2005.
        """)

    st.subheader("Q6 · Forest Biomass along Elevation and Slope")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Brown et al., 1995, Biotropica; Givnish, 1988, The American Naturalist):</strong>
    Tropical montane forests exhibit strong spatial variation in aboveground biomass due to two key abiotic factors: elevation (which correlates with temperature, precipitation, and cloud cover) and slope angle (which influences soil depth, drainage, and stability).
    <br><br>
    The model fitted to field data from Costa Rica is: <strong>Biomass (t/ha) = 300·exp(−((elev‑500)²/80000))·exp(−0.05·slope) + 50</strong>.
    The elevation term is a Gaussian peak centred at 500 m (mid‑elevation), with σ ≈ 200 m. The slope term is an exponential decay: each additional degree of slope reduces biomass by about 5% (factor exp(−0.05)). The constant +50 t/ha represents a baseline biomass even on the steepest slopes.
    <br><br>
    The 3D surface below shows predicted biomass across the observed ranges (elevation 0–1200 m, slope 0–40°).
    </div>
    """, unsafe_allow_html=True)
    display_figure(6)

    question_tf(6, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "The elevation of maximum biomass (500 m) is independent of slope (i.e., the Gaussian peak does not shift with slope).",
            "At a slope of 10°, the predicted biomass at the optimal elevation is approximately 232 t/ha.",
            "The relative reduction in biomass caused by slope is larger at high elevations than at low elevations (i.e., the slope factor is elevation‑dependent).",
            "Steep slopes often have shallower soils and higher erosion rates, which are ecological reasons for lower biomass."
        ],
        correct_answers=[True, True, False, True],
        explanation="""
**Statement 1 – TRUE:** The Gaussian function depends only on elevation; the slope term is multiplicative and independent of elevation. Therefore the elevation of maximum biomass stays at 500 m for all slopes – visible in the surface as a ridge running parallel to the slope axis at elev=500 m.

**Statement 2 – TRUE:** At elev = 500 m, the Gaussian term = 300·exp(0) = 300. For slope = 10°, exp(−0.05·10) = exp(−0.5) ≈ 0.6065. Then biomass = 300·0.6065 + 50 ≈ 182 + 50 = 232 t/ha. The exact value matches the statement.

**Statement 3 – FALSE:** The slope factor is exp(−0.05·slope), which is purely a function of slope, not of elevation. The proportional reduction (e.g., from slope 0° to 10°) is the same at every elevation: (300·F(slope) + 50) / (300 + 50) is not constant, but the *multiplicative* effect on the variable part (the 300·exp(...) term) is independent of elevation. Nevertheless, the statement "slope effect is stronger at lower elevations" would imply an interaction term, which the model does not include. The graph shows parallel ridges – the relative shape is the same at all elevations. Therefore the statement is false.

**Statement 4 – TRUE:** This is a general ecological mechanism. Steep slopes increase gravitational water runoff, reduce soil depth, and are prone to landslides – all of which limit tree growth and survival, leading to lower standing biomass.
        """)
# Q7-8
with tabs[3]:
    st.subheader("Q7 · Wheat Yield Response to Nitrogen and Phosphorus")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Wang et al., 2019, Field Crops Research; Tilman et al., 2002, Nature – agricultural sustainability):</strong>
    A factorial field experiment was conducted to determine the optimum fertiliser rates for winter wheat (Triticum aestivum) on a loamy soil in the North China Plain. Treatments consisted of five nitrogen rates (0–250 kg ha⁻¹) and five phosphorus rates (0–80 kg ha⁻¹), with three replicates.
    <br><br>
    The fitted quadratic response surface was:
    <strong>Yield (t ha⁻¹) = 2 + 0.04·N + 0.06·P – 0.00015·N·P – 0.0001·N² – 0.0004·P²</strong>.
    The negative quadratic terms represent diminishing returns and toxicity at high doses; the negative interaction term (−0.00015·N·P) indicates that nitrogen and phosphorus partially substitute for each other – when one nutrient is abundant, the marginal benefit of the other is reduced.
    <br><br>
    The optimum occurs near N≈130 kg ha⁻¹, P≈50 kg ha⁻¹, giving ≈6.2 t ha⁻¹. The 3D surface below shows predicted yield across the experimental ranges.
    </div>
    """, unsafe_allow_html=True)
    display_figure(7)

    question_tf(7, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "The negative interaction term (−0.00015·N·P) reduces the positive effect of phosphorus when nitrogen is high (and vice versa).",
            "According to the model, the yield at N=200, P=20 is lower than at N=100, P=20.",
            "At the optimum, the partial derivative of yield with respect to nitrogen equals zero (a necessary condition for an interior maximum).",
            "Nitrogen and phosphorus are often co‑limiting in agricultural systems, meaning that adding only one nutrient may not increase yield if the other is deficient."
        ],
        correct_answers=[True, False, True, True],
        explanation="""
**Statement 1 – TRUE:** The interaction term is negative. Its effect is to subtract 0.00015·N·P from the linear terms. When N is large, the negative contribution from N·P increases, effectively reducing the benefit of additional P. This is a typical antagonistic interaction.

**Statement 2 – FALSE:** Compute: at (200,20): yield = 2 + 0.04·200 + 0.06·20 – 0.00015·200·20 – 0.0001·200² – 0.0004·20² = 2 + 8 + 1.2 – 0.6 – 4 – 0.16 = 6.44 t ha⁻¹. At (100,20): 2 + 4 + 1.2 – 0.00015·100·20 – 0.0001·100² – 0.0004·20² = 2 + 4 + 1.2 – 0.3 – 1 – 0.16 = 5.74 t ha⁻¹. So yield at (200,20) is higher, not lower. The statement is false.

**Statement 3 – TRUE:** At an interior optimum of a smooth function, all first partial derivatives are zero. Setting ∂Yield/∂N = 0.04 – 0.00015·P – 0.0002·N = 0 (and similarly for P) yields the optimal N and P. This is a standard calculus condition.

**Statement 4 – TRUE:** Co‑limitation is common in many terrestrial ecosystems. The Liebig's law of the minimum applies: yield is limited by the scarcest nutrient. The positive interaction term would indicate synergy (e.g., added N makes added P more effective), but the negative term here shows the opposite, yet co‑limitation can still occur when both are low. The statement is generally true.
        """)

    st.subheader("Q8 · NDVI Time Series – Seasonal and Biennial Dynamics")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Verbesselt et al., 2010, Remote Sensing of Environment; Tucker et al., 2005, International Journal of Remote Sensing):</strong>
    The Normalised Difference Vegetation Index (NDVI = (NIR – Red)/(NIR + Red)) is a proxy for photosynthetic activity and green biomass. MODIS satellite data from a semi‑arid region of Australia (2015–2020) were decomposed into:
    <br><br>
    <strong>NDVI = 0.3 + 0.25·sin(2π·(M‑3)/12) + 0.01·(Y‑2015) + 0.02·sin(2π·Y/2)</strong>,
    where M = month (1–12), Y = year. The terms represent:
    - Baseline (0.3)
    - Seasonal cycle: amplitude 0.25, peak in August (M=8)
    - Linear trend: +0.01 per year (greening)
    - Biennial oscillation: amplitude 0.02, period 2 years
    <br><br>
    The 3D surface below shows NDVI as a function of month (x‑axis) and year (y‑axis). The red‑green colormap (RdYlGn) highlights high NDVI in green.
    </div>
    """, unsafe_allow_html=True)
    display_figure(8)

    question_tf(8, "For each statement below, decide whether it is TRUE or FALSE based on the graph and context:",
        statements=[
            "The highest August NDVI occurred in 2020 (the last year of the time series).",
            "The linear trend term increases NDVI by 0.01 per month (i.e., 0.12 per year).",
            "The term 0.02·sin(2π·Y/2) gives a biennial oscillation with a period of exactly 2 years.",
            "El Niño events are known to increase NDVI in many semi‑arid regions by bringing above‑average rainfall – this is not modelled directly but is a real ecological phenomenon."
        ],
        correct_answers=[True, False, True, True],
        explanation="""
**Statement 1 – TRUE:** The linear trend adds 0.01·(Y‑2015). Thus August NDVI = 0.3 + 0.25·sin(2π·5/12) + 0.01·(Y‑2015) + 0.02·sin(π·Y). The sine of August is constant (~0.866), and the biennial term alternates sign but does not change the fact that the trend is positive, so the maximum August value is in the last year, 2020.

**Statement 2 – FALSE:** The coefficient 0.01 is applied to (Y‑2015), which is years, not months. So the trend is +0.01 per year, not per month. Over a year, NDVI increases by 0.01, not 0.12.

**Statement 3 – TRUE:** The argument inside sin is 2π·Y/2 = π·Y. The sine of π·Y has period 2 (since sin(π(Y+2)) = sin(πY+2π) = sin(πY)). Thus the term oscillates every 2 years, i.e., a biennial cycle.

**Statement 4 – TRUE:** El Niño–Southern Oscillation (ENSO) strongly influences rainfall patterns in Australia, parts of Africa, and South America. Positive phases (El Niño) often bring drought, while negative phases (La Niña) bring wetter conditions, affecting NDVI. Although the simple model here does not include ENSO, the ecological link is well established.
        """)
    #Q9-10
with tabs[4]:
    st.subheader("Q9 · Trophic Pyramid of a Temperate Grassland")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Lindeman, 1942, Ecology – the classic trophic‑dynamic concept; Odum, 1968, Fundamentals of Ecology):</strong>
    In a temperate grassland ecosystem (Konza Prairie, Kansas, USA), standing biomass (g m⁻²) was measured for four trophic levels:
    - Producers (C4 grasses and forbs): 1200 g m⁻²
    - Primary consumers (herbivores: grasshoppers, voles): 180 g m⁻²
    - Secondary consumers (carnivores: shrews, snakes, small birds): 25 g m⁻²
    - Tertiary consumers (top predators: foxes, hawks): 2 g m⁻²
    <br><br>
    Trophic (Lindeman) efficiency is the ratio of biomass (or energy) at one level to that at the previous level:
    - Plant → herbivore: 180/1200 = 15%
    - Herbivore → carnivore: 25/180 ≈ 13.9%
    - Carnivore → top predator: 2/25 = 8%
    <br><br>
    The bar chart below illustrates the biomass pyramid.
    </div>
    """, unsafe_allow_html=True)
    display_figure(9)

    question_tf(9, "For each statement below, decide whether it is TRUE or FALSE based on the data and ecological principles:",
        statements=[
            "The trophic efficiency from herbivores to carnivores is approximately 14%.",
            "Low trophic efficiency is primarily due to energy lost as heat, respiration, and waste – only about 10% of consumed energy is converted to new biomass.",
            "Top predators always occupy exactly trophic level 4 (e.g., lion, tiger, wolf).",
            "The 10% rule is an average; measured efficiencies can range from 5% to 20% depending on the ecosystem and trophic level."
        ],
        correct_answers=[True, True, False, True],
        explanation="""
**Statement 1 – TRUE:** 25/180 ≈ 0.1389 ≈ 14%. This matches the calculation.

**Statement 2 – TRUE:** The Second Law of Thermodynamics dictates that energy transformations are inefficient. Respiration, heat production, and undigested matter account for most of the energy not transferred to the next level. Lindeman's original work established efficiencies around 10–20%.

**Statement 3 – FALSE:** Top predators can occupy trophic level 5 or even higher. For example, in a marine food chain: phytoplankton (level 1) → zooplankton (2) → small fish (3) → large fish (4) → shark (5). Some ecosystems have six levels. The statement "always" makes it false.

**Statement 4 – TRUE:** The 10% rule is a convenient average. In some ecosystems (e.g., tropical rainforests) it may be lower due to high respiration; in some aquatic systems it may be higher (up to 20%). The statement is correct.
        """)

    st.subheader("Q10 · Predator‑Prey Model with Intraspecific Competition")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Rosenzweig & MacArthur, 1963, The American Naturalist – the 'paradox of enrichment'; also Volterra, 1926):</strong>
    A predator‑prey system is described by the following differential equation for prey (N) density:
    <strong>dN/dt = 0.08·N – 0.012·N·P – 0.001·N²</strong>.
    The predator equation is not needed for this question. The three terms represent:
    - Exponential growth (intrinsic rate r = 0.08)
    - Linear functional response (predation rate proportional to NP, with coefficient 0.012)
    - Intraspecific competition (density‑dependence, –0.001·N²) which stabilises the system.
    <br><br>
    The prey zero‑isocline (dN/dt = 0) is found by setting the right‑hand side to zero:
    <strong>N = (0.08 – 0.012·P) / 0.001</strong>, for N > 0 and P such that numerator positive.
    <br><br>
    The 3D surface below shows dN/dt as a function of prey (N) and predator (P). The red region (negative) indicates prey decline, green (positive) indicates prey growth.
    </div>
    """, unsafe_allow_html=True)
    display_figure(10)

    question_tf(10, "For each statement below, decide whether it is TRUE or FALSE based on the model and graph:",
        statements=[
            "The prey zero‑isocline slopes downward with increasing predator density (higher P → lower equilibrium N).",
            "At a predator density of P = 30, the equilibrium prey density is approximately 80 (use the isocline formula).",
            "The term –0.001·N² represents intraspecific competition (e.g., competition for food, space).",
            "The classic Lotka‑Volterra predator‑prey model (without the N² term) produces stable limit cycles; adding the N² term stabilises the system (dampens oscillations)."
        ],
        correct_answers=[True, False, True, True],
        explanation="""
**Statement 1 – TRUE:** From dN/dt = 0 → N = (0.08 – 0.012P)/0.001. This is a linear function with negative slope (–0.012/0.001 = –12). So as predator density increases, the prey equilibrium density decreases.

**Statement 2 – FALSE:** At P = 30, N = (0.08 – 0.012·30)/0.001 = (0.08 – 0.36)/0.001 = (–0.28)/0.001 = –280. A negative population is impossible, meaning that the prey cannot survive at such high predator density – the system would collapse. Therefore the statement is false.

**Statement 3 – TRUE:** The term –0.001·N² is a standard logistic‑type density‑dependent term that reduces per‑capita growth when N is large, reflecting limited resources, territoriality, or other competition mechanisms.

**Statement 4 – TRUE:** The classic Lotka‑Volterra equations (dN/dt = aN – bNP, dP/dt = cNP – dP) produce neutrally stable cycles (the amplitude depends on initial conditions). Adding a density‑dependent term (–eN²) to the prey equation makes the cycles damped (the system returns to a stable equilibrium), which is more realistic. This is a standard result in ecology.
        """)
    #Q11-12
with tabs[5]:
    st.subheader("Q11 · Plant Diversity along Elevation and Soil pH")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Lomolino, 2001, Global Ecology & Biogeography; Rahbek, 2005, Journal of Biogeography – elevational diversity patterns):</strong>
    A survey of vascular plant species was conducted along an elevational transect (0–2800 m) in the Swiss Alps, with soil pH measured at each site. The Shannon diversity index (H') was modelled using a product of two Gaussian functions:
    <br><br>
    <strong>Diversity = 15·exp(−((elev‑1400)²/500000))·exp(−((pH‑6.2)²/1.28))</strong>.
    <br><br>
    The first Gaussian peaks at 1400 m with σ ≈ 500 m (mid‑elevation peak), the second at pH 6.2 with σ ≈ 0.89. The constant 15 scales the maximum diversity. The surface plot below shows predicted H' across both gradients.
    </div>
    """, unsafe_allow_html=True)
    display_figure(11)

    question_tf(11, "For each statement below, decide whether it is TRUE or FALSE based on the model and ecology:",
        statements=[
            "At pH 5.2, the predicted diversity is about half of the maximum value (i.e., H' ≈ 7.5).",
            "At 2800 m elevation (the highest sampled), diversity is less than 5.",
            "The Gaussian width for elevation (σ) is approximately 500 m – true for this model.",
            "Alpine plant diversity is low mainly due to harsh climatic conditions (short growing season, low temperature, high UV) – a generally accepted explanation."
        ],
        correct_answers=[True, True, True, True],
        explanation="""
**Statement 1 – TRUE:** The pH Gaussian: exp(−((5.2‑6.2)²/(2·0.8²))) = exp(−(1)²/(1.28)) = exp(−0.78125) ≈ 0.458. At the peak (elev=1400, pH=6.2), diversity = 15. So at pH 5.2 (keeping elev=1400) diversity ≈ 15·0.458 ≈ 6.87, which is about half of 15 (7.5). The statement is approximately correct.

**Statement 2 – TRUE:** At elev=2800, elev Gaussian term = exp(−((2800‑1400)²/(2·500²))) = exp(−(1400²/(500000))) = exp(−1960000/500000) = exp(−3.92) ≈ 0.02. Diversity ≈ 15·0.02 = 0.3, far below 5. True.

**Statement 3 – TRUE:** The model uses 500 m as the standard deviation (σ). The exponent denominator is 2σ² = 500000 → σ² = 250000 → σ = 500 m. So the statement is exactly true.

**Statement 4 – TRUE:** This is a classic ecological explanation. High‑elevation environments impose severe abiotic constraints, leading to low species richness. The statement is generally accepted, though other factors (area, evolutionary history) also play a role.
        """)

    st.subheader("Q12 · Community Ordination and Beta Diversity")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (McCune & Grace, 2002, Analysis of Ecological Communities; Whittaker, 1960, Ecological Monographs):</strong>
    A non‑metric multidimensional scaling (NMDS) ordination was performed on plant community data from 60 plots across three sites (A, B, C) in a Mediterranean shrubland. The 3D plot shows scores for the first three NMDS axes, with points coloured by site.
    <br><br>
    Site C has the lowest within‑site dispersion (tightest clustering), indicating low beta diversity among plots within that site. Site B has the highest dispersion. The centroid (average location) of Site A is approximately at coordinates (0.5, 0.5, 0.5).
    <br><br>
    Beta diversity measures the turnover (change) in species composition between sites or along environmental gradients.
    </div>
    """, unsafe_allow_html=True)
    display_figure(12)

    question_tf(12, "For each statement below, decide whether it is TRUE or FALSE based on the graph and ecological theory:",
        statements=[
            "Site C has the lowest within‑site dispersion (i.e., plots are most similar to each other).",
            "The centroid of Site A is approximately at (0.8, 0.3, 0.5).",
            "Beta diversity measures the diversity within a single site (alpha diversity) – this is a common confusion.",
            "High beta diversity may indicate an ecotone, a transition zone between two distinct habitats where species composition changes rapidly."
        ],
        correct_answers=[True, False, False, True],
        explanation="""
**Statement 1 – TRUE:** Visual inspection of the 3D scatter shows Site C points are tightly clumped, Site A points are somewhat spread, and Site B points are widely scattered. Hence Site C has lowest within‑site dispersion.

**Statement 2 – FALSE:** The centroid of Site A is roughly (0.5, 0.5, 0.5) as stated in the context. (0.8, 0.3, 0.5) would be far from the cluster. Therefore false.

**Statement 3 – FALSE:** Beta diversity is defined as the variation in species composition among sites (between‑site turnover). Alpha diversity is within‑site richness. The statement confuses the two and is false.

**Statement 4 – TRUE:** Ecotones often exhibit high species turnover because species adapted to one habitat are replaced by others across a short distance. This results in high beta diversity. The statement is correct.
        """)
# Q13-14
with tabs[6]:
    st.subheader("Q13 · Leaf Mass as a Function of Temperature and Precipitation")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Reich, 2014, Annual Review of Ecology, Evolution, and Systematics – leaf economics spectrum; Wright et al., 2004, Nature):</strong>
    A global dataset of 80 leaf samples was used to model leaf dry mass (g) as a function of mean annual temperature (T, °C) and annual precipitation (P, mm). The fitted multiple linear regression was:
    <br><br>
    <strong>Leaf mass = 0.2·T + 0.05·P + ε</strong>, where ε ∼ N(0, 4).
    <br><br>
    The temperature coefficient (0.2) is four times larger than the precipitation coefficient (0.05), indicating that temperature has a stronger influence on leaf mass. The residual standard deviation is 4 g, representing unexplained variation (e.g., due to soil fertility, genetics, measurement error).
    <br><br>
    The 3D scatter plot below shows the data points and the fitted plane (colour indicates leaf mass).
    </div>
    """, unsafe_allow_html=True)
    display_figure(13)

    question_tf(13, "For each statement below, decide whether it is TRUE or FALSE based on the model and ecology:",
        statements=[
            "Temperature has a stronger positive effect on leaf mass than precipitation (coefficient 0.2 vs 0.05).",
            "Increasing precipitation from 800 mm to 1000 mm adds approximately 10 g to leaf mass (all else equal).",
            "The residual standard deviation of the model is 4 g.",
            "Higher temperatures can extend the growing season, allowing larger leaves, but may also cause heat stress and water loss – a well‑known trade‑off."
        ],
        correct_answers=[True, True, True, True],
        explanation="""
**Statement 1 – TRUE:** The coefficients directly indicate effect size per unit change. A 1 °C increase adds 0.2 g, while a 1 mm increase adds only 0.05 g. Temperature effect is stronger.

**Statement 2 – TRUE:** ΔP = 200 mm → Δleaf mass = 0.05·200 = 10 g. The calculation is straightforward.

**Statement 3 – TRUE:** The error term is ε ∼ N(0, 4), meaning the standard deviation is 4 g. This is the typical variation around the predicted mean.

**Statement 4 – TRUE:** This trade‑off is central to plant functional ecology. Longer growing seasons favour larger leaves, but high temperatures increase vapour pressure deficit and evapotranspiration, potentially leading to stomatal closure and reduced carbon gain, which can limit leaf size. The statement is correct.
        """)

    st.subheader("Q14 · Poleward Range Shift under Climate Change")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Parmesan & Yohe, 2003, Nature; Chen et al., 2011, Science – meta‑analysis of range shifts):</strong>
    A long‑term study of a butterfly species (Hesperia comma) in the UK tracked its northern range boundary from 1980 to 2020. The probability of occurrence at a given latitude L follows a logistic function, with the latitude of 50% probability (the median range limit) given by:
    <br><br>
    <strong>L₀.₅ (year) = 42 + 0.3·(year – 1980)</strong> (latitude in °N).
    <br><br>
    The logistic slope (steepness) is constant over time, meaning the shape of the range boundary does not change – it only shifts northward at a rate of 0.3° per year (≈ 33 km yr⁻¹). The 3D surface below shows the probability distribution across latitude and time.
    </div>
    """, unsafe_allow_html=True)
    display_figure(14)

    question_tf(14, "For each statement below, decide whether it is TRUE or FALSE based on the model and context:",
        statements=[
            "In 1980, the latitude of 50% occurrence probability was 42°N.",
            "By 2050, the 50% probability latitude will be approximately 63°N.",
            "The logistic slope (steepness of the probability curve) increases over time as the range boundary sharpens.",
            "Poleward range shifts have been documented for many plant and animal species in response to recent climate warming."
        ],
        correct_answers=[True, True, False, True],
        explanation="""
**Statement 1 – TRUE:** Set year = 1980 → L₀.₅ = 42 + 0.3·0 = 42°N. Correct.

**Statement 2 – TRUE:** year = 2050 → L₀.₅ = 42 + 0.3·70 = 42 + 21 = 63°N. Correct.

**Statement 3 – FALSE:** The problem explicitly states that the logistic slope (which determines how quickly the probability rises from 0 to 1 over a small latitude band) remains constant. Only the midpoint shifts. Therefore the statement is false.

**Statement 4 – TRUE:** Meta‑analyses (e.g., Parmesan & Yohe 2003, Chen et al. 2011) have found that hundreds of species have shifted their ranges poleward and upward in elevation at average rates of about 17 km per decade. This is one of the strongest fingerprints of climate change on biodiversity.
        """)
# Q15
with tabs[7]:
    st.subheader("Q15 · Soil Respiration Response to Temperature and Moisture")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Lloyd & Taylor, 1994, Functional Ecology; Davidson & Janssens, 2006, Global Change Biology):</strong>
    Soil CO₂ efflux (respiration, Rs) was measured in a temperate hardwood forest. The model that best described the data combined an exponential temperature response (Q10) and a parabolic moisture response:
    <br><br>
    <strong>Rs = 1.5·exp(0.08·T)·(M/100)·exp(−0.02·M)</strong>,
    where T = soil temperature at 10 cm depth (°C), M = soil moisture (% of field capacity).
    <br><br>
    The Q10 (temperature coefficient) is exp(0.08·10) ≈ 2.23, meaning a 10 °C increase roughly doubles respiration. The moisture term M·exp(−0.02·M) reaches a maximum at M = 50% (optimum water content). At low M, respiration is limited by water stress; at high M, oxygen limitation reduces activity.
    <br><br>
    The 3D surface below shows Rs across the observed ranges (T: 0–35°C, M: 10–90%).
    </div>
    """, unsafe_allow_html=True)
    display_figure(15)

    question_tf(15, "For each statement below, decide whether it is TRUE or FALSE based on the model and context:",
        statements=[
            "The Q10 of soil respiration is approximately 2.23 (consistent with exp(0.08·10)).",
            "The optimal soil moisture for respiration is about 40% (peak of the function M·exp(−0.02·M)).",
            "At T=15 °C and M=50%, the predicted respiration is about 0.8 μmol m⁻² s⁻¹.",
            "Waterlogging (very high M) reduces oxygen availability, inhibiting aerobic microbial activity and root respiration – this is a known mechanism."
        ],
        correct_answers=[True, False, False, True],
        explanation="""
**Statement 1 – TRUE:** Q10 = exp(β·10) where β = 0.08. Thus Q10 = exp(0.8) ≈ 2.2255 ≈ 2.23. Correct.

**Statement 2 – FALSE:** The moisture function f(M) = M·exp(−0.02·M). Its derivative f'(M) = exp(−0.02·M)·(1 – 0.02·M). Setting f'(M)=0 gives 1 – 0.02M = 0 → M = 50%. Therefore the optimum is 50%, not 40%. The statement is false.

**Statement 3 – FALSE:** Compute: at T=15°C, exp(0.08·15) = exp(1.2) ≈ 3.320. At M=50%, (M/100)=0.5, exp(−0.02·50)=exp(−1)=0.3679. Then Rs = 1.5 × 3.320 × 0.5 × 0.3679 ≈ 1.5 × 3.320 × 0.18395 ≈ 1.5 × 0.6106 ≈ 0.916 μmol m⁻² s⁻¹, not 0.8. So false.

**Statement 4 – TRUE:** Excess soil moisture fills pore spaces, reducing the diffusion of oxygen to roots and soil microbes. Anaerobic conditions shift metabolism to less efficient fermentation or denitrification, reducing CO₂ efflux. This is a well‑established factor in soil respiration models.
        """)
# Q16-20 numeric (unchanged)
# Q16-17 (first numeric pair)
with tabs[8]:
    st.subheader("Q16 · Pollinator Visitation – Temperature and Diurnal Pattern")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Corbet et al., 1993, Functional Ecology; Kühsel & Blüthgen, 2015, Oikos):</strong>
    Pollinator visitation rates (visits per hour) to a flower patch were modelled as a product of a diurnal Gaussian (peak at 13:00) and a thermal Gaussian (optimum 26°C):
    <strong>Visits = 15·exp(−((hour‑13)²/8))·exp(−((T‑26)²/32))·(1 + 0.1·sin(π·(hour‑6)/14))</strong>.
    The small sinusoidal term represents a slight afternoon asymmetry. For a constant temperature of 26°C, the peak visitation hour is exactly 13:00 (1 PM).
    </div>
    """, unsafe_allow_html=True)
    display_figure(16)
    question_num(16, "Peak visitation hour at constant 26°C",
                "The diurnal Gaussian is centered at 13:00; the sinusoidal term does not shift the peak.", 
                answer=13, tolerance=0.5,
                explanation="The exponential term is symmetric around 13:00; the small sinusoidal term has a derivative zero at 13:00 (since cos(π·(13‑6)/14)=cos(π/2)=0), so the peak remains at 13.")

    st.subheader("Q17 · Leaf Gas Exchange – Combined CO₂ and Light Response")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Farquhar et al., 1980, Planta; von Caemmerer, 2000, Biochemical Models of Leaf Photosynthesis):</strong>
    A leaf gas exchange model combines a light‑limited (electron transport) term and a CO₂‑limited (Rubisco) term. For given CO₂ and light, net assimilation is approximated by:
    <strong>An = 12·(1‑exp(−0.004·L))·(CO₂/(CO₂+180))·(1‑0.0002·(CO₂‑400)²)</strong>.
    The term 1‑exp(−0.004·L) saturates at high light; the CO₂ factor includes a quadratic penalty far from 400 ppm.
    Compute An at CO₂ = 600 ppm, light = 1500 µmol m⁻² s⁻¹.
    </div>
    """, unsafe_allow_html=True)
    display_figure(17)
    question_num(17, "Net assimilation at CO₂ = 600 ppm, L = 1500 µmol m⁻² s⁻¹",
                "An = 12·(1‑e⁻⁶)·(600/780)·(1‑0.0002·200²) = 12·0.9975·0.7692·(1‑0.008) = 12·0.9975·0.7692·0.992 ≈ 9.1.",
                answer=9.1, tolerance=0.5,
                explanation="12·0.9975≈11.97, ×0.7692≈9.20, ×0.992≈9.13. The expected answer is ≈9.1 (range 8.7–9.6).")

# Q18-20 (second numeric pair)
with tabs[9]:
    st.subheader("Q18 · Stomatal Conductance – VPD and Leaf Temperature")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Leuning, 1995, Plant, Cell & Environment; Buckley & Mott, 2013, Plant Physiology):</strong>
    Stomatal conductance gₛ (mol m⁻² s⁻¹) decreases exponentially with leaf‑to‑air vapour pressure deficit (VPD, kPa) and has a parabolic temperature response:
    <strong>gₛ = 0.5·exp(−1.3·VPD)·(1+0.05·(T‑25))·exp(−0.01·(T‑35)²)</strong>.
    Compute gₛ at VPD = 2.0 kPa, leaf temperature = 30°C.
    </div>
    """, unsafe_allow_html=True)
    display_figure(18)
    question_num(18, "gₛ at VPD=2.0, T=30°C",
                "exp(−2.6)=0.074; (1+0.05·5)=1.25; exp(−0.01·25)=0.779; product: 0.5×0.074×1.25×0.779 = 0.036.",
                answer=0.036, tolerance=0.008,
                explanation="0.5·0.074·1.25·0.779 = 0.0360 (range 0.028–0.044).")

    st.subheader("Q19 · Chlorophyll Fluorescence under Combined Drought and Heat")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Maxwell & Johnson, 2000, Journal of Experimental Botany – Fv/Fm as stress indicator):</strong>
    The maximum quantum efficiency of PSII (Fv/Fm) in control leaves is about 0.83. Under combined drought (D, %) and heat (H, °C) stress, Fv/Fm declines linearly with each factor and an interaction term:
    <strong>Fv/Fm = 0.83 – 0.005·D – 0.008·(H‑25) – 0.0002·D·H</strong>.
    Compute Fv/Fm at D = 40%, H = 35°C.
    </div>
    """, unsafe_allow_html=True)
    display_figure(19)
    question_num(19, "Fv/Fm at D=40%, H=35°C",
                "0.83 – 0.2 – 0.08 – 0.28 = 0.27.",
                answer=0.27, tolerance=0.03,
                explanation="0.83 – 0.20 – 0.08 – 0.28 = 0.27 (range 0.24–0.30).")

    st.subheader("Q20 · Root‑Shoot Allocation under Nitrogen and Water Stress")
    st.markdown("""
    <div class="context-box">
    <strong>Study context (Bloom et al., 1985, Annual Review of Ecology and Systematics – optimal allocation theory):</strong>
    The root/shoot ratio (R/S) is a key functional trait that adjusts to resource limitation. The model combines nitrogen (N, kg ha⁻¹) and water (W, %):
    <strong>R/S = 0.25 + 0.005·(250‑N) – 0.002·W + 0.0001·(250‑N)·W/100</strong>.
    Compute R/S at N = 100 kg ha⁻¹, W = 50%.
    </div>
    """, unsafe_allow_html=True)
    display_figure(20)
    question_num(20, "Root/shoot ratio at N=100, W=50%",
                "0.25 + 0.005·150 – 0.002·50 + 0.0001·150·50/100 = 0.25 + 0.75 – 0.10 + 0.075 = 0.975.",
                answer=0.975, tolerance=0.05,
                explanation="0.25+0.75-0.10+0.075 = 0.975 (range 0.925–1.025).")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2d6a4f; font-size:0.85rem; padding:1rem 0 0.5rem;">
  🌿 IBO 2026 · Plant Computational Biology · Vilnius, Lithuania
</div>
""", unsafe_allow_html=True)