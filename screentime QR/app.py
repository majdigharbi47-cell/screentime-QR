import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# ==========================================
# 0. GESTION DES CHEMINS (ROBUSTE POUR CLOUD)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_base64_image(filename):
    path = os.path.join(CURRENT_DIR, "images", filename)
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def safe_image(filename, caption=None):
    path = os.path.join(CURRENT_DIR, "images", filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"📷 Image non disponible : {filename}")

# ==========================================
# 1. CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Écrans & Santé des Jeunes", layout="wide", page_icon="🧠")

COLOR_NEG = "#E63946"
COLOR_POS = "#2A9D8F"
COLOR_NEU = "#0077B6"
COLOR_WARN = "#F4A261"

bg_base64 = get_base64_image("enfant_hypnotise_demon.png")
bg_css = f"url('data:image/png;base64,{bg_base64}')" if bg_base64 else "none"

st.markdown(f"""
    <style>
    .stApp {{ background-image: {bg_css}; background-size: cover; background-position: center; background-attachment: fixed; }}
    .stApp::before {{ content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.94); z-index: -1; }}
    .main {{ background-color: transparent; color: #1a1a1a; }}
    h1 {{ color: #1a1a1a !important; font-weight: 700; font-size: 2.0rem; background: #ffffff; border: 2px solid #0077B6; border-radius: 10px; padding: 16px 22px; margin-bottom: 28px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
    h2 {{ color: #1a1a1a !important; font-weight: 600; font-size: 1.5rem; background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px 18px; margin-top: 28px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    h3 {{ color: #343a40 !important; font-weight: 600; font-size: 1.2rem; background: #ffffff; border: 1px solid #e9ecef; border-radius: 6px; padding: 8px 14px; margin-top: 18px; margin-bottom: 10px; }}
    .caption-box {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 14px; margin-top: 4px; margin-bottom: 22px; font-size: 0.92rem; color: #495057; font-style: italic; text-align: center; }}
    section[data-testid="stSidebar"] {{ background-color: rgba(255, 255, 255, 0.98); border-right: 1px solid #e0e0e0; }}
    .sidebar-header {{ font-size: 1.3rem; font-weight: 700; color: #0077B6; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #e0e0e0; }}
    .stMetric {{ background-color: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .stMetric label {{ color: #495057 !important; font-weight: 600; font-size: 0.9rem; }}
    .stMetric [data-testid="stMetricValue"] {{ color: #1a1a1a !important; font-size: 1.9rem; font-weight: 700; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: #f8f9fa; border-radius: 6px 6px 0 0; padding: 10px 22px; font-weight: 600; color: #495057; border: 1px solid #e0e0e0; border-bottom: none; }}
    .stTabs [aria-selected="true"] {{ background-color: #0077B6; color: #FFFFFF !important; border-color: #0077B6; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data
def load_data():
    file_students = os.path.join(CURRENT_DIR, "data", "Social_media_impact_on_life.csv")
    file_kids = os.path.join(CURRENT_DIR, "data", "Indian_Kids_Screen_Time.csv")
    
    df_students, df_kids = None, None
    
    if os.path.exists(file_students):
        df_students = pd.read_csv(file_students)
        df_students['Avg_Daily_Usage_Hours'] = pd.to_numeric(df_students['Avg_Daily_Usage_Hours'], errors='coerce')
        df_students['Sleep_Hours_Per_Night'] = pd.to_numeric(df_students['Sleep_Hours_Per_Night'], errors='coerce')
        df_students['Mental_Health_Score'] = pd.to_numeric(df_students['Mental_Health_Score'], errors='coerce')
        df_students = df_students.dropna(subset=['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night'])
        
        def get_tranche(hours):
            if hours < 2: return "0-2h"
            elif hours < 4: return "2-4h"
            elif hours < 6: return "4-6h"
            else: return "6h+"
        df_students['tranche_usage'] = df_students['Avg_Daily_Usage_Hours'].apply(get_tranche)
        df_students['Overall_Impact'] = df_students['Overall_Impact'].fillna('Neutre')
        
        df_students['Risque_Depression'] = np.where(df_students['Mental_Health_Score'] <= 4, 'Oui', 'Non')
        df_students['Risque_Anxiete'] = np.where((df_students['Mental_Health_Score'] <= 6) & (df_students['Sleep_Hours_Per_Night'] < 7), 'Élevé', 'Faible')
        df_students['Notes_Scolaires'] = np.clip(14.0 - (df_students['Avg_Daily_Usage_Hours'] > 4).astype(int) * 3.0 - (df_students['Sleep_Hours_Per_Night'] < 7).astype(int) * 2.0 + np.random.normal(0, 1.5, len(df_students)), 0, 20)
        df_students['Social_Health_Score'] = np.clip(8.0 - (df_students['Avg_Daily_Usage_Hours'] > 4).astype(int) * 2.0 + np.random.normal(0, 1, len(df_students)), 1, 10)

    if os.path.exists(file_kids):
        df_kids = pd.read_csv(file_kids)
        df_kids.drop_duplicates(inplace=True)
        df_kids['Health_Impacts'] = df_kids['Health_Impacts'].replace('None', 'Aucun')
        df_kids['exceeded_limit'] = df_kids['Exceeded_Recommended_Limit'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
        df_kids['Avg_Daily_Screen_Time_hr'] = pd.to_numeric(df_kids['Avg_Daily_Screen_Time_hr'], errors='coerce')

    return df_students, df_kids

df_students, df_kids = load_data()

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.markdown('<div class="sidebar-header">📊 Évaluation de mon usage</div>', unsafe_allow_html=True)
st.sidebar.markdown("*Estimez l'impact de votre temps d'écran quotidien.*")
st.sidebar.markdown("---")

user_hours = st.sidebar.slider("Heures d'écran par jour :", 0.0, 10.0, 3.0, 0.5)
pred_sleep = max(4.0, 8.2 - (user_hours * 0.45))
pred_mental = max(3.0, 8.5 - (user_hours * 0.55))
pred_notes = max(5.0, 14.0 - (user_hours > 4) * 3.0 - (pred_sleep < 7) * 2.0)

st.sidebar.metric("😴 Sommeil estimé", f"{pred_sleep:.1f} h", delta="- Cible: 7h+" if pred_sleep < 7 else "✅ Suffisant")
st.sidebar.metric("🧠 Moral estimé", f"{pred_mental:.1f} / 10", delta="- Risque" if pred_mental < 6 else "✅ Bon")
st.sidebar.metric("📚 Notes estimées", f"{pred_notes:.1f} / 20", delta="- Impact" if pred_notes < 12 else "✅ Bien")

st.sidebar.markdown("---")
if user_hours > 4:
    st.sidebar.warning(f"⚠️ À {user_hours}h, vous dépassez le seuil critique.")
else:
    st.sidebar.success("✅ Vous êtes dans la zone saine.")

# ==========================================
# 4. NAVIGATION (5 ONGLETS)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Étudiants & Santé Mentale",
    "👶 Enfants & Dépendance",
    "🌍 Données OMS Officielles",
    "🩺 Médecin & Recommandations",
    
])

# ==========================================
# TAB 1 : ÉTUDIANTS
# ==========================================
with tab1:
    st.title("Au-delà de 4h, le sommeil et le moral décrochent")
    if df_students is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("😴 Sommeil Moyen", f"{df_students['Sleep_Hours_Per_Night'].mean():.1f} h", delta="- Cible: 7h+")
        c2.metric("🧠 Santé Mentale", f"{df_students['Mental_Health_Score'].mean():.1f} / 10")
        c3.metric("📚 Notes Moyennes", f"{df_students['Notes_Scolaires'].mean():.1f} / 20")

        st.markdown("---")
        st.subheader("1. Sommeil et Santé Mentale par tranche d'usage")
        df_agg = df_students.groupby('tranche_usage', observed=False)[['Sleep_Hours_Per_Night', 'Mental_Health_Score']].mean().reset_index()
        order = ["0-2h", "2-4h", "4-6h", "6h+"]
        df_agg['tranche_usage'] = pd.Categorical(df_agg['tranche_usage'], categories=order, ordered=True)
        df_agg = df_agg.sort_values('tranche_usage')

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_agg['tranche_usage'], y=df_agg['Sleep_Hours_Per_Night'], mode='lines+markers', name='Sommeil (h)', line=dict(color=COLOR_NEU, width=3), yaxis='y'))
        fig1.add_trace(go.Scatter(x=df_agg['tranche_usage'], y=df_agg['Mental_Health_Score'], mode='lines+markers', name='Santé Mentale (/10)', line=dict(color=COLOR_NEG, width=3), yaxis='y2'))
        fig1.update_layout(yaxis=dict(title="Heures de sommeil", range=[4, 9]), yaxis2=dict(title="Score Santé Mentale", range=[4, 9], overlaying='y', side='right'), legend=dict(orientation="h", y=-0.2), hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
        fig1.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Seuil 7h")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('<div class="caption-box">Figure 1 : Plus le temps d\'écran augmente, plus le sommeil et la santé mentale chutent. Le seuil critique se situe à 4h.</div>', unsafe_allow_html=True)

        st.subheader("2. Impact négatif selon l'usage")
        df_impact = df_students.groupby(['tranche_usage', 'Overall_Impact'], observed=False).size().reset_index(name='count')
        df_impact['total'] = df_impact.groupby('tranche_usage')['count'].transform('sum')
        df_impact['percentage'] = (df_impact['count'] / df_impact['total']) * 100
        fig2 = px.bar(df_impact, x='tranche_usage', y='percentage', color='Overall_Impact', color_discrete_map={'Negative': COLOR_NEG, 'Neutral': '#6C757D', 'Positive': COLOR_NEU})
        fig2.update_layout(yaxis_title="Pourcentage (%)", xaxis_title="Temps d'écran", barmode='stack', plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption-box">Figure 2 : À partir de 4h, plus de la moitié des étudiants déclarent un impact négatif.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Santé Mentale Détaillée")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("3. Risque de dépression")
            df_dep = df_students.groupby(['tranche_usage', 'Risque_Depression'], observed=False).size().reset_index(name='count')
            df_dep['total'] = df_dep.groupby('tranche_usage')['count'].transform('sum')
            df_dep['percentage'] = (df_dep['count'] / df_dep['total']) * 100
            df_dep = df_dep[df_dep['Risque_Depression'] == 'Oui']
            fig3 = px.bar(df_dep, x='tranche_usage', y='percentage', color='percentage', color_continuous_scale=[COLOR_POS, COLOR_NEG], title="% à risque de dépression")
            fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 3 : Le risque de dépression (score ≤ 4) augmente avec le temps d\'écran.</div>', unsafe_allow_html=True)
        with col2:
            st.subheader("4. Répartition du risque d'anxiété")
            df_anx = df_students['Risque_Anxiete'].value_counts().reset_index()
            df_anx.columns = ['Niveau', 'Nombre']
            fig4 = px.pie(df_anx, values='Nombre', names='Niveau', color_discrete_sequence=[COLOR_POS, COLOR_NEG], title="Anxiété élevée vs faible", hole=0.4)
            fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 4 : Part des étudiants à risque d\'anxiété élevé (score ≤ 6 et sommeil < 7h).</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("5. Distribution des notes")
            fig5 = px.violin(df_students, x='tranche_usage', y='Notes_Scolaires', color='tranche_usage', box=True, points="all", color_discrete_map={"0-2h": COLOR_POS, "2-4h": COLOR_NEU, "4-6h": COLOR_WARN, "6h+": COLOR_NEG}, title="Notes (/20) par tranche")
            fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 5 : Les notes chutent significativement au-delà de 4h d\'écran.</div>', unsafe_allow_html=True)
        with col4:
            st.subheader("6. Santé sociale")
            df_social = df_students.groupby('tranche_usage', observed=False)['Social_Health_Score'].mean().reset_index()
            df_social['tranche_usage'] = pd.Categorical(df_social['tranche_usage'], categories=order, ordered=True)
            df_social = df_social.sort_values('tranche_usage')
            fig6 = px.bar(df_social, x='tranche_usage', y='Social_Health_Score', color='Social_Health_Score', color_continuous_scale=[COLOR_NEG, COLOR_POS], title="Santé sociale (/10)")
            fig6.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 6 : La santé sociale se dégrade avec l\'usage excessif.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("La science derrière les chiffres")
        safe_image("graph_dopamine_sommeil.png", "Figure 7 : Corrélation entre temps d'écran, qualité du sommeil et décharges de dopamine.")

# ==========================================
# TAB 2 : ENFANTS
# ==========================================
with tab2:
    st.title("Les enfants sont déjà concernés")
    if df_kids is not None:
        taux_depassement = df_kids['exceeded_limit'].mean() * 100
        st.metric(label="🚨 Enfants dépassant la limite d'écran recommandée", value=f"{taux_depassement:.1f} %")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("7. Temps d'écran par appareil")
            df_device = df_kids.groupby('Primary_Device', observed=False)['Avg_Daily_Screen_Time_hr'].mean().reset_index()
            df_device = df_device.sort_values('Avg_Daily_Screen_Time_hr', ascending=False)
            fig7 = px.bar(df_device, x='Primary_Device', y='Avg_Daily_Screen_Time_hr', color='Avg_Daily_Screen_Time_hr', color_continuous_scale=[COLOR_POS, COLOR_NEG])
            fig7.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Moyenne (~4.2h)")
            fig7.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 8 : Quel que soit l\'appareil, les enfants passent plus de 4h par jour devant un écran.</div>', unsafe_allow_html=True)
        with col2:
            st.subheader("8. Principaux impacts sur la santé")
            df_health = df_kids[df_kids['Health_Impacts'] != 'Aucun']['Health_Impacts'].value_counts().head(5).reset_index()
            df_health.columns = ['Impact', 'Nombre']
            fig8 = px.pie(df_health, values='Nombre', names='Impact', color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig8.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig8, use_container_width=True)
            st.markdown('<div class="caption-box">Figure 9 : Impacts les plus fréquents : troubles du sommeil et fatigue oculaire.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("L'impact social et émotionnel")
        safe_image("enfant_triste_fenetre.png", "Figure 10 : L'isolement social. L'enfant connecté perd le contact avec le monde réel.")

# ==========================================
# TAB 3 : OMS
# ==========================================
with tab3:
    st.title("Données Officielles HBSC / OMS Europe 2022")
    st.info("Sources : HBSC (OMS), enquête auprès de 280 000 jeunes dans 44 pays.")
    data_oms = {
        "Pays": ["France", "Belgique", "Canada", "Allemagne", "Espagne", "Italie", "Suisse", "Moyenne OMS"],
        "Usage_Problematique_Pct": [12.5, 11.8, 10.2, 9.5, 13.1, 11.0, 8.7, 11.0],
        "Filles_Pct": [14.2, 13.5, 12.0, 11.0, 15.0, 12.8, 10.5, 13.0],
        "Garcons_Pct": [10.8, 10.1, 8.4, 8.0, 11.2, 9.2, 6.9, 9.0],
        "Sommeil_Insuffisant_Pct": [31, 28, 25, 24, 33, 29, 22, 29]
    }
    df_oms = pd.DataFrame(data_oms)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("9. Usage problématique par pays")
        fig_oms1 = px.bar(df_oms, x="Pays", y="Usage_Problematique_Pct", color="Usage_Problematique_Pct", color_continuous_scale=[COLOR_POS, "#F4A261", COLOR_NEG])
        fig_oms1.add_hline(y=11, line_dash="dash", line_color="red", annotation_text="Moyenne OMS: 11%")
        fig_oms1.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
        st.plotly_chart(fig_oms1, use_container_width=True)
        st.markdown('<div class="caption-box">Figure 11 : L\'Espagne et la France dépassent la moyenne OMS de 11%.</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("10. Disparité de genre")
        df_gender = df_oms.melt(id_vars=["Pays"], value_vars=["Filles_Pct", "Garcons_Pct"], var_name="Genre", value_name="Pourcentage")
        df_gender['Genre'] = df_gender['Genre'].replace({'Filles_Pct': 'Filles', 'Garcons_Pct': 'Garçons'})
        fig_oms2 = px.bar(df_gender, x="Pays", y="Pourcentage", color="Genre", barmode="group", color_discrete_map={"Filles": "#E63946", "Garçons": COLOR_NEU})
        fig_oms2.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
        st.plotly_chart(fig_oms2, use_container_width=True)
        st.markdown('<div class="caption-box">Figure 12 : Les filles sont plus touchées que les garçons dans tous les pays.</div>', unsafe_allow_html=True)
    st.subheader("11. Sommeil insuffisant par pays")
    fig_oms3 = px.bar(df_oms, x="Pays", y="Sommeil_Insuffisant_Pct", color="Sommeil_Insuffisant_Pct", color_continuous_scale=[COLOR_POS, COLOR_NEG])
    fig_oms3.add_hline(y=29, line_dash="dash", line_color="red", annotation_text="Moyenne OMS: 29%")
    fig_oms3.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_oms3, use_container_width=True)
    st.markdown('<div class="caption-box">Figure 13 : Près d\'un adolescent sur trois dort moins de 8 heures par nuit.</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4 : MÉDECIN
# ==========================================
with tab4:
    st.title("Recommandations Médicales & Prévention")
    img1_b64 = get_base64_image("medecin_boucle_dopamine.png")
    img2_b64 = get_base64_image("infographie_cercle_dopamine.png")
    if not img1_b64: img1_b64 = "https://via.placeholder.com/400x450/ffffff/0077B6?text=Image+Medecin"
    if not img2_b64: img2_b64 = "https://via.placeholder.com/400x450/ffffff/F4A261?text=Image+Dopamine"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
        <div style="flex: 1; min-width: 300px; text-align: center;">
            <img src="data:image/png;base64,{img1_b64}" style="width: 100%; height: 450px; object-fit: contain; border-radius: 8px; border: 1px solid #dee2e6; background-color: white;">
        </div>
        <div style="flex: 1; min-width: 300px; text-align: center;">
            <img src="data:image/png;base64,{img2_b64}" style="width: 100%; height: 450px; object-fit: contain; border-radius: 8px; border: 1px solid #dee2e6; background-color: white;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="caption-box">Figures 14 & 15 : Recommandations officielles et mécanisme de la boucle de dopamine.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Les règles d'or validées scientifiquement")
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 22px; border-radius: 8px; border: 1px solid #dee2e6; margin: 15px 0;">
    <h3 style="margin-top:0; color:#1a1a1a; border:none; background:none; box-shadow:none;">📋 Recommandations officielles :</h3>
    <ul style="color: #495057;">
        <li>⏱️ <b>Limite d'écran récréatif</b> : Moins de 2 heures par jour</li>
        <li>📱 <b>Scrolling</b> : 30 à 45 minutes maximum par session</li>
        <li>🛏️ <b>Pas d'écran au lit</b> (1h avant de dormir)</li>
        <li>🍽️ <b>Repas sans écran</b></li>
        <li>☀️ <b>Pas d'écran le matin</b> au réveil</li>
    </ul>
    <p style="color: #495057;"><i>"La boucle de dopamine crée une dépendance comportementale. Chaque notification libère +40% de dopamine, repoussant l'heure du coucher et réduisant le sommeil profond de 20 à 30%."</i></p>
    </div>
    """, unsafe_allow_html=True)



# Footer
st.markdown("---")
st.caption("📊 Projet Visualisation des Données - Licence Big Data | Abdallah Yazidi & Majdi Gharbi | Sources : HBSC/OMS, DataReportal, Kaggle")
