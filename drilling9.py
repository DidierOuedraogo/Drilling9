import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io

# Configuration de la page
st.set_page_config(
    page_title="Analyse de forages miniers",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {text}</a>'
    return href

# Fonction pour créer des composites d'analyses
def create_composites(assays_df, hole_id_col, from_col, to_col, value_col, composite_length=1.0):
    if assays_df is None or assays_df.empty:
        return None
    
    # Créer une copie des données pour éviter de modifier l'original
    df = assays_df.copy()
    
    # Initialiser le DataFrame des composites
    composites = []
    
    # Pour chaque trou de forage
    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].sort_values(by=from_col)
        
        if hole_data.empty:
            continue
        
        # Pour chaque intervalle de composite
        composite_start = float(hole_data[from_col].min())
        while composite_start < float(hole_data[to_col].max()):
            composite_end = composite_start + composite_length
            
            # Trouver tous les intervalles qui chevauchent le composite actuel
            overlapping = hole_data[
                ((hole_data[from_col] >= composite_start) & (hole_data[from_col] < composite_end)) |
                ((hole_data[to_col] > composite_start) & (hole_data[to_col] <= composite_end)) |
                ((hole_data[from_col] <= composite_start) & (hole_data[to_col] >= composite_end))
            ]
            
            if not overlapping.empty:
                # Calculer le poids pondéré pour chaque intervalle chevauchant
                weighted_values = []
                total_length = 0
                
                for _, row in overlapping.iterrows():
                    overlap_start = max(composite_start, row[from_col])
                    overlap_end = min(composite_end, row[to_col])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0:
                        weighted_values.append(row[value_col] * overlap_length)
                        total_length += overlap_length
                
                # Calculer la valeur pondérée du composite
                if total_length > 0:
                    composite_value = sum(weighted_values) / total_length
                    
                    # Ajouter le composite au résultat
                    composites.append({
                        hole_id_col: hole_id,
                        'From': composite_start,
                        'To': composite_end,
                        'Length': total_length,
                        value_col: composite_value
                    })
            
            composite_start = composite_end
    
    # Créer un DataFrame à partir des composites
    if composites:
        return pd.DataFrame(composites)
    else:
        return pd.DataFrame()

# Fonction pour créer un strip log pour un forage spécifique
def create_strip_log(hole_id, collars_df, survey_df, lithology_df, assays_df, 
                    hole_id_col, depth_col, 
                    lith_from_col, lith_to_col, lith_col,
                    assay_from_col, assay_to_col, assay_value_col):
    
    # Vérifier si les données nécessaires sont disponibles
    if collars_df is None or survey_df is None:
        return None
    
    # Récupérer les informations du forage
    hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
    
    if hole_surveys.empty:
        return None
    
    # Profondeur maximale du forage
    max_depth = hole_surveys[depth_col].max()
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(12, max_depth/10 + 2), 
                            gridspec_kw={'width_ratios': [2, 1, 3]})
    
    # Titre du graphique
    fig.suptitle(f'Strip Log - Forage {hole_id}', fontsize=16)
    
    # 1. Colonne de lithologie
    if lithology_df is not None and lith_from_col and lith_to_col and lith_col:
        hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id].sort_values(by=lith_from_col)
        
        if not hole_litho.empty:
            # Définir une palette de couleurs pour les différentes lithologies
            unique_liths = hole_litho[lith_col].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_liths)))
            lith_color_map = {lith: color for lith, color in zip(unique_liths, colors)}
            
            # Dessiner des rectangles pour chaque intervalle de lithologie
            for _, row in hole_litho.iterrows():
                lith_from = row[lith_from_col]
                lith_to = row[lith_to_col]
                lith_type = row[lith_col]
                
                axes[0].add_patch(plt.Rectangle((0, lith_from), 1, lith_to - lith_from, 
                                                color=lith_color_map[lith_type]))
                
                # Ajouter le texte de la lithologie au milieu de l'intervalle
                axes[0].text(0.5, (lith_from + lith_to) / 2, lith_type,
                            ha='center', va='center', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Configurer l'axe de la lithologie
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(max_depth, 0)  # Inverser l'axe y pour que la profondeur augmente vers le bas
    axes[0].set_xlabel('Lithologie')
    axes[0].set_ylabel('Profondeur (m)')
    axes[0].set_xticks([])
    
    # 2. Colonne de profondeur
    depth_ticks = np.arange(0, max_depth + 10, 10)
    axes[1].set_yticks(depth_ticks)
    axes[1].set_ylim(max_depth, 0)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_xlabel('Profondeur')
    axes[1].grid(axis='y')
    
    # 3. Colonne d'analyses
    if assays_df is not None and assay_from_col and assay_to_col and assay_value_col:
        hole_assays = assays_df[assays_df[hole_id_col] == hole_id].sort_values(by=assay_from_col)
        
        if not hole_assays.empty:
            # Trouver la valeur maximale pour normaliser
            max_value = hole_assays[assay_value_col].max()
            
            # Dessiner des barres horizontales pour chaque intervalle d'analyse
            for _, row in hole_assays.iterrows():
                assay_from = row[assay_from_col]
                assay_to = row[assay_to_col]
                assay_value = row[assay_value_col]
                
                # Dessiner une barre horizontale pour la valeur
                bar_width = (assay_value / max_value) * 0.9  # Normaliser la largeur
                axes[2].add_patch(plt.Rectangle((0, assay_from), bar_width, assay_to - assay_from, 
                                                color='red', alpha=0.7))
                
                # Ajouter la valeur comme texte
                axes[2].text(bar_width + 0.05, (assay_from + assay_to) / 2, f"{assay_value:.2f}",
                            va='center', fontsize=8)
    
    # Configurer l'axe des analyses
    axes[2].set_xlim(0, 1.2)
    axes[2].set_ylim(max_depth, 0)
    axes[2].set_xlabel(f'Analyses ({assay_value_col})')
    axes[2].grid(axis='y')
    
    plt.tight_layout()
    
    # Convertir le graphique en image pour Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Fonction pour créer une représentation 3D des forages
def create_drillhole_3d_plot(collars_df, survey_df, lithology_df=None, assays_df=None, 
                            hole_id_col=None, x_col=None, y_col=None, z_col=None,
                            azimuth_col=None, dip_col=None, depth_col=None,
                            lith_from_col=None, lith_to_col=None, lith_col=None,
                            assay_from_col=None, assay_to_col=None, assay_value_col=None):
    
    if collars_df is None or survey_df is None:
        return None
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Pour chaque trou de forage
    for hole_id in collars_df[hole_id_col].unique():
        # Récupérer les données de collar pour ce trou
        collar = collars_df[collars_df[hole_id_col] == hole_id]
        if collar.empty:
            continue
            
        # Point de départ du trou (collar)
        x_start = collar[x_col].values[0]
        y_start = collar[y_col].values[0]
        z_start = collar[z_col].values[0]
        
        # Récupérer les données de survey pour ce trou
        hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        if hole_surveys.empty:
            continue
            
        # Calculer les points 3D pour le tracé du trou
        x_points = [x_start]
        y_points = [y_start]
        z_points = [z_start]
        
        current_x, current_y, current_z = x_start, y_start, z_start
        prev_depth = 0
        
        for _, survey in hole_surveys.iterrows():
            depth = survey[depth_col]
            azimuth = survey[azimuth_col]
            dip = survey[dip_col]
            
            segment_length = depth - prev_depth
            
            # Convertir l'azimuth et le dip en direction 3D
            azimuth_rad = np.radians(azimuth)
            dip_rad = np.radians(dip)
            
            # Calculer la nouvelle position
            dx = segment_length * np.sin(dip_rad) * np.sin(azimuth_rad)
            dy = segment_length * np.sin(dip_rad) * np.cos(azimuth_rad)
            dz = -segment_length * np.cos(dip_rad)  # Z est négatif pour la profondeur
            
            current_x += dx
            current_y += dy
            current_z += dz
            
            x_points.append(current_x)
            y_points.append(current_y)
            z_points.append(current_z)
            
            prev_depth = depth
        
        # Ajouter la trace du trou de forage
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='lines',
                name=f'Forage {hole_id}',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                hovertext=[f'ID: {hole_id}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
                          for x, y, z in zip(x_points, y_points, z_points)]
            )
        )
        
        # Ajouter les intersections lithologiques si disponibles
        if lithology_df is not None and lith_from_col and lith_to_col and lith_col:
            hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id]
            
            if not hole_litho.empty:
                for _, litho in hole_litho.iterrows():
                    from_depth = litho[lith_from_col]
                    to_depth = litho[lith_to_col]
                    lith_type = litho[lith_col]
                    
                    # Simplification: placer des marqueurs aux points médians des intervalles lithologiques
                    midpoint_depth = (from_depth + to_depth) / 2
                    
                    # Trouver les coordonnées 3D pour ce point
                    idx = np.interp(midpoint_depth, hole_surveys[depth_col], np.arange(len(hole_surveys)))
                    idx = int(min(idx, len(hole_surveys)-1))
                    
                    if idx < len(x_points) - 1:
                        # Calculer la fraction entre les deux points de survey
                        depths = hole_surveys[depth_col].values
                        if idx + 1 < len(depths):
                            fraction = (midpoint_depth - depths[idx]) / (depths[idx+1] - depths[idx]) if depths[idx+1] > depths[idx] else 0
                            
                            # Interpoler les coordonnées 3D
                            x_lith = x_points[idx] + fraction * (x_points[idx+1] - x_points[idx])
                            y_lith = y_points[idx] + fraction * (y_points[idx+1] - y_points[idx])
                            z_lith = z_points[idx] + fraction * (z_points[idx+1] - z_points[idx])
                            
                            # Ajouter un marqueur pour cette lithologie
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[x_lith],
                                    y=[y_lith],
                                    z=[z_lith],
                                    mode='markers',
                                    name=f'{hole_id}: {lith_type}',
                                    marker=dict(size=8, color=px.colors.qualitative.Plotly[hash(lith_type) % len(px.colors.qualitative.Plotly)]),
                                    hoverinfo='text',
                                    hovertext=f'ID: {hole_id}<br>Lithologie: {lith_type}<br>Profondeur: {from_depth:.2f}-{to_depth:.2f}m'
                                )
                            )
        
        # Ajouter les valeurs d'analyses si disponibles
        if assays_df is not None and assay_from_col and assay_to_col and assay_value_col:
            hole_assays = assays_df[assays_df[hole_id_col] == hole_id]
            
            if not hole_assays.empty:
                max_value = hole_assays[assay_value_col].max()
                
                for _, assay in hole_assays.iterrows():
                    from_depth = assay[assay_from_col]
                    to_depth = assay[assay_to_col]
                    value = assay[assay_value_col]
                    
                    # Simplification: placer des marqueurs aux points médians des intervalles d'analyse
                    midpoint_depth = (from_depth + to_depth) / 2
                    
                    # Trouver les coordonnées 3D pour ce point
                    idx = np.interp(midpoint_depth, hole_surveys[depth_col], np.arange(len(hole_surveys)))
                    idx = int(min(idx, len(hole_surveys)-1))
                    
                    if idx < len(x_points) - 1:
                        # Calculer la fraction entre les deux points de survey
                        depths = hole_surveys[depth_col].values
                        if idx + 1 < len(depths):
                            fraction = (midpoint_depth - depths[idx]) / (depths[idx+1] - depths[idx]) if depths[idx+1] > depths[idx] else 0
                            
                            # Interpoler les coordonnées 3D
                            x_assay = x_points[idx] + fraction * (x_points[idx+1] - x_points[idx])
                            y_assay = y_points[idx] + fraction * (y_points[idx+1] - y_points[idx])
                            z_assay = z_points[idx] + fraction * (z_points[idx+1] - z_points[idx])
                            
                            # Normaliser la valeur pour déterminer la taille du marqueur
                            marker_size = 5 + (value / max_value) * 15 if max_value > 0 else 5
                            
                            # Ajouter un marqueur pour cette analyse
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[x_assay],
                                    y=[y_assay],
                                    z=[z_assay],
                                    mode='markers',
                                    name=f'{hole_id}: {value:.2f}',
                                    marker=dict(
                                        size=marker_size, 
                                        color=value,
                                        colorscale='Reds',
                                        colorbar=dict(title=assay_value_col)
                                    ),
                                    hoverinfo='text',
                                    hovertext=f'ID: {hole_id}<br>Teneur: {value:.2f}<br>Profondeur: {from_depth:.2f}-{to_depth:.2f}m'
                                )
                            )
    
    # Ajuster la mise en page
    fig.update_layout(
        title="Visualisation 3D des forages",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Élévation)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Titre de l'application
st.title('Analyse de données de forages miniers')

# Barre latérale pour la navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Sélectionnez une page:', [
    'Chargement des données', 
    'Aperçu des données', 
    'Composites', 
    'Strip Logs',
    'Visualisation 3D'
])

# Initialisation des variables de session
if 'collars_df' not in st.session_state:
    st.session_state.collars_df = None
    
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
    
if 'lithology_df' not in st.session_state:
    st.session_state.lithology_df = None
    
if 'assays_df' not in st.session_state:
    st.session_state.assays_df = None
    
if 'composites_df' not in st.session_state:
    st.session_state.composites_df = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'hole_id': None,
        'x': None,
        'y': None,
        'z': None,
        'azimuth': None,
        'dip': None,
        'depth': None,
        'lith_from': None,
        'lith_to': None,
        'lithology': None,
        'assay_from': None,
        'assay_to': None,
        'assay_value': None
    }

# Page de chargement des données
if page == 'Chargement des données':
    st.header('Chargement des données')
    
    # Créer des onglets pour les différents types de données
    tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
    
    # Onglet Collars
    with tabs[0]:
        st.subheader('Chargement des données de collars')
        
        collars_file = st.file_uploader("Télécharger le fichier CSV des collars", type=['csv'])
        if collars_file is not None:
            st.session_state.collars_df = pd.read_csv(collars_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.collars_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.collars_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou", 
                                                                         [''] + cols, 
                                                                         index=0)
            with col2:
                st.session_state.column_mapping['x'] = st.selectbox("Colonne X", 
                                                                    [''] + cols,
                                                                    index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['y'] = st.selectbox("Colonne Y", 
                                                                    [''] + cols,
                                                                    index=0)
            with col2:
                st.session_state.column_mapping['z'] = st.selectbox("Colonne Z", 
                                                                    [''] + cols,
                                                                    index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données"):
                st.dataframe(st.session_state.collars_df.head())
    
    # Onglet Survey
    with tabs[1]:
        st.subheader('Chargement des données de survey')
        
        survey_file = st.file_uploader("Télécharger le fichier CSV des surveys", type=['csv'])
        if survey_file is not None:
            st.session_state.survey_df = pd.read_csv(survey_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.survey_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.survey_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Survey)", 
                                                                         [''] + cols, 
                                                                         index=0)
            with col2:
                st.session_state.column_mapping['depth'] = st.selectbox("Colonne profondeur", 
                                                                        [''] + cols,
                                                                        index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['azimuth'] = st.selectbox("Colonne azimut", 
                                                                          [''] + cols,
                                                                          index=0)
            with col2:
                st.session_state.column_mapping['dip'] = st.selectbox("Colonne pendage", 
                                                                      [''] + cols,
                                                                      index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Survey)"):
                st.dataframe(st.session_state.survey_df.head())
    
    # Onglet Lithologie
    with tabs[2]:
        st.subheader('Chargement des données de lithologie')
        
        lithology_file = st.file_uploader("Télécharger le fichier CSV des lithologies", type=['csv'])
        if lithology_file is not None:
            st.session_state.lithology_df = pd.read_csv(lithology_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.lithology_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.lithology_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Lithologie)", 
                                                                         [''] + cols, 
                                                                         index=0)
            with col2:
                st.session_state.column_mapping['lithology'] = st.selectbox("Colonne de lithologie", 
                                                                           [''] + cols,
                                                                           index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['lith_from'] = st.selectbox("Colonne de profondeur début", 
                                                                           [''] + cols,
                                                                           index=0)
            with col2:
                st.session_state.column_mapping['lith_to'] = st.selectbox("Colonne de profondeur fin", 
                                                                         [''] + cols,
                                                                         index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Lithologie)"):
                st.dataframe(st.session_state.lithology_df.head())
    
    # Onglet Analyses
    with tabs[3]:
        st.subheader('Chargement des données d\'analyses')
        
        assays_file = st.file_uploader("Télécharger le fichier CSV des analyses", type=['csv'])
        if assays_file is not None:
            st.session_state.assays_df = pd.read_csv(assays_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.assays_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.assays_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Analyses)", 
                                                                         [''] + cols, 
                                                                         index=0)
            with col2:
                st.session_state.column_mapping['assay_value'] = st.selectbox("Colonne de valeur (par ex. Au g/t)", 
                                                                             [''] + cols,
                                                                             index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['assay_from'] = st.selectbox("Colonne de profondeur début (Analyses)", 
                                                                            [''] + cols,
                                                                            index=0)
            with col2:
                st.session_state.column_mapping['assay_to'] = st.selectbox("Colonne de profondeur fin (Analyses)", 
                                                                          [''] + cols,
                                                                          index=0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Analyses)"):
                st.dataframe(st.session_state.assays_df.head())

# Page d'aperçu des données
elif page == 'Aperçu des données':
    st.header('Aperçu des données')
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None and st.session_state.lithology_df is None and st.session_state.assays_df is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger des données.")
    else:
        # Créer des onglets pour les différents types de données
        data_tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
        
        # Onglet Collars
        with data_tabs[0]:
            if st.session_state.collars_df is not None:
                st.subheader('Données de collars')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.collars_df)}")
                st.dataframe(st.session_state.collars_df)
                
                st.markdown(get_csv_download_link(st.session_state.collars_df, "collars_data.csv", "les données de collars"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de collars n'a été chargée.")
        
        # Onglet Survey
        with data_tabs[1]:
            if st.session_state.survey_df is not None:
                st.subheader('Données de survey')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.survey_df)}")
                st.dataframe(st.session_state.survey_df)
                
                st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data.csv", "les données de survey"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de survey n'a été chargée.")
        
        # Onglet Lithologie
        with data_tabs[2]:
            if st.session_state.lithology_df is not None:
                st.subheader('Données de lithologie')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.lithology_df)}")
                st.dataframe(st.session_state.lithology_df)
                
                st.markdown(get_csv_download_link(st.session_state.lithology_df, "lithology_data.csv", "les données de lithologie"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de lithologie n'a été chargée.")
        
        # Onglet Analyses
        with data_tabs[3]:
            if st.session_state.assays_df is not None:
                st.subheader('Données d\'analyses')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.assays_df)}")
                st.dataframe(st.session_state.assays_df)
                
                st.markdown(get_csv_download_link(st.session_state.assays_df, "assays_data.csv", "les données d'analyses"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée d'analyses n'a été chargée.")

# Page de calcul des composites
elif page == 'Composites':
    st.header('Calcul des composites')
    
    if st.session_state.assays_df is None:
        st.warning("Aucune donnée d'analyses n'a été chargée. Veuillez d'abord charger des données d'analyses.")
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        assay_from_col = st.session_state.column_mapping['assay_from']
        assay_to_col = st.session_state.column_mapping['assay_to']
        assay_value_col = st.session_state.column_mapping['assay_value']
        
        if not (hole_id_col and assay_from_col and assay_to_col and assay_value_col):
            st.warning("Veuillez d'abord spécifier toutes les colonnes nécessaires dans la page de chargement des données.")
        else:
            st.subheader("Options de composites")
            
            # Sélectionner la longueur des composites
            composite_length = st.slider("Longueur des composites (m)", 
                                        min_value=0.5, 
                                        max_value=5.0, 
                                        value=1.0, 
                                        step=0.5)
            
            # Calculer les composites
            if st.button("Calculer les composites"):
                with st.spinner("Calcul des composites en cours..."):
                    st.session_state.composites_df = create_composites(
                        st.session_state.assays_df,
                        hole_id_col,
                        assay_from_col,
                        assay_to_col,
                        assay_value_col,
                        composite_length
                    )
                
                if st.session_state.composites_df is not None and not st.session_state.composites_df.empty:
                    st.success(f"Composites calculés avec succès. {len(st.session_state.composites_df)} enregistrements générés.")
                    
                    # Afficher les composites
                    st.subheader("Résultats des composites")
                    st.dataframe(st.session_state.composites_df)
                    
                    # Lien de téléchargement
                    st.markdown(get_csv_download_link(st.session_state.composites_df, "composites.csv", "les composites"), unsafe_allow_html=True)
                    
                    # Résumé statistique
                    st.subheader("Résumé statistique")
                    st.write(st.session_state.composites_df[assay_value_col].describe())
                    
                    # Histogramme des valeurs de composites
                    fig = px.histogram(
                        st.session_state.composites_df, 
                        x=assay_value_col,
                        title=f"Distribution des valeurs de composites ({assay_value_col})",
                        labels={assay_value_col: f'Teneur'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparaison avec les données originales
                    st.subheader("Comparaison avec les données originales")
                    
                    comparison_data = pd.DataFrame({
                        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                        'Données originales': [
                            st.session_state.assays_df[assay_value_col].mean(),
                            st.session_state.assays_df[assay_value_col].median(),
                            st.session_state.assays_df[assay_value_col].std(),
                            st.session_state.assays_df[assay_value_col].min(),
                            st.session_state.assays_df[assay_value_col].max()
                        ],
                        'Composites': [
                            st.session_state.composites_df[assay_value_col].mean(),
                            st.session_state.composites_df[assay_value_col].median(),
                            st.session_state.composites_df[assay_value_col].std(),
                            st.session_state.composites_df[assay_value_col].min(),
                            st.session_state.composites_df[assay_value_col].max()
                        ]
                    })
                    
                    st.table(comparison_data.set_index('Statistique').round(3))
                else:
                    st.error("Impossible de calculer les composites avec les données fournies.")

# Page de Strip Logs
elif page == 'Strip Logs':
    st.header('Strip Logs des forages')
    
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.warning("Les données de collars et de survey sont nécessaires pour les strip logs. Veuillez les charger d'abord.")
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        depth_col = st.session_state.column_mapping['depth']
        
        if not (hole_id_col and depth_col):
            st.warning("Veuillez d'abord spécifier les colonnes d'ID de trou et de profondeur dans la page de chargement des données.")
        else:
            # Sélection du forage à afficher
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            selected_hole = st.selectbox("Sélectionner un forage", all_holes)
            
            if selected_hole:
                # Récupérer les informations du forage sélectionné
                selected_collar = st.session_state.collars_df[st.session_state.collars_df[hole_id_col] == selected_hole]
                selected_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col] == selected_hole]
                
                if not selected_survey.empty:
                    # Afficher les informations du forage
                    collar_info = selected_collar.iloc[0]
                    
                    st.subheader(f"Informations sur le forage {selected_hole}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"X: {collar_info[st.session_state.column_mapping['x']]}")
                        st.write(f"Y: {collar_info[st.session_state.column_mapping['y']]}")
                        st.write(f"Z: {collar_info[st.session_state.column_mapping['z']]}")
                    
                    with col2:
                        max_depth = selected_survey[depth_col].max()
                        st.write(f"Profondeur maximale: {max_depth:.2f} m")
                        
                        # Infos supplémentaires si lithologie disponible
                        if st.session_state.lithology_df is not None and st.session_state.column_mapping['lithology']:
                            lith_col = st.session_state.column_mapping['lithology']
                            selected_litho = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col] == selected_hole]
                            if not selected_litho.empty:
                                unique_liths = selected_litho[lith_col].nunique()
                                st.write(f"Nombre de lithologies: {unique_liths}")
                    
                    with col3:
                        # Infos supplémentaires si analyses disponibles
                        if st.session_state.assays_df is not None and st.session_state.column_mapping['assay_value']:
                            assay_value_col = st.session_state.column_mapping['assay_value']
                            selected_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col] == selected_hole]
                            if not selected_assays.empty:
                                avg_value = selected_assays[assay_value_col].mean()
                                max_value = selected_assays[assay_value_col].max()
                                st.write(f"Valeur moyenne: {avg_value:.2f}")
                                st.write(f"Valeur maximale: {max_value:.2f}")
                    
                    # Créer et afficher le strip log
                    strip_log_image = create_strip_log(
                        selected_hole,
                        st.session_state.collars_df,
                        st.session_state.survey_df,
                        st.session_state.lithology_df,
                        st.session_state.assays_df,
                        hole_id_col,
                        depth_col,
                        st.session_state.column_mapping['lith_from'],
                        st.session_state.column_mapping['lith_to'],
                        st.session_state.column_mapping['lithology'],
                        st.session_state.column_mapping['assay_from'],
                        st.session_state.column_mapping['assay_to'],
                        st.session_state.column_mapping['assay_value']
                    )
                    
                    if strip_log_image:
                        st.image(strip_log_image, caption=f"Strip Log du forage {selected_hole}", use_column_width=True)
                        
                        # Téléchargement de l'image
                        btn = st.download_button(
                            label="Télécharger le strip log",
                            data=strip_log_image,
                            file_name=f"strip_log_{selected_hole}.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Impossible de créer le strip log avec les données fournies.")
                else:
                    st.error(f"Aucune donnée de survey trouvée pour le forage {selected_hole}.")

# Page de visualisation 3D
elif page == 'Visualisation 3D':
    st.header('Visualisation 3D des forages')
    
    # Vérifier si les données nécessaires ont été chargées
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.warning("Les données de collars et de survey sont nécessaires pour la visualisation 3D. Veuillez les charger d'abord.")
    else:
        # Vérifier si les colonnes nécessaires ont été spécifiées
        required_cols = ['hole_id', 'x', 'y', 'z', 'azimuth', 'dip', 'depth']
        missing_cols = [col for col in required_cols if st.session_state.column_mapping[col] is None or st.session_state.column_mapping[col] == '']
        
        if missing_cols:
            st.warning(f"Certaines colonnes requises n'ont pas été spécifiées: {', '.join(missing_cols)}. Veuillez les définir dans l'onglet 'Chargement des données'.")
        else:
            # Options pour la visualisation
            st.subheader("Options de visualisation")
            
            # Sélection des forages à afficher
            hole_id_col = st.session_state.column_mapping['hole_id']
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            selected_holes = st.multiselect("Sélectionner les forages à afficher", all_holes, default=all_holes[:min(5, len(all_holes))])
            
            # Options additionnelles
            col1, col2 = st.columns(2)
            with col1:
                show_lithology = st.checkbox("Afficher la lithologie", value=True if st.session_state.lithology_df is not None else False)
            with col2:
                show_assays = st.checkbox("Afficher les teneurs", value=True if st.session_state.assays_df is not None else False)
            
            # Filtrer les données selon les forages sélectionnés
            if selected_holes:
                filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(selected_holes)]
                filtered_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(selected_holes)]
                
                # Filtrer lithology et assays si nécessaire
                filtered_lithology = None
                if show_lithology and st.session_state.lithology_df is not None:
                    filtered_lithology = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col].isin(selected_holes)]
                
                filtered_assays = None
                if show_assays and st.session_state.assays_df is not None:
                    filtered_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(selected_holes)]
                
                # Créer la visualisation 3D
                fig_3d = create_drillhole_3d_plot(
                    filtered_collars, filtered_survey, filtered_lithology, filtered_assays,
                    hole_id_col=hole_id_col,
                    x_col=st.session_state.column_mapping['x'],
                    y_col=st.session_state.column_mapping['y'],
                    z_col=st.session_state.column_mapping['z'],
                    azimuth_col=st.session_state.column_mapping['azimuth'],
                    dip_col=st.session_state.column_mapping['dip'],
                    depth_col=st.session_state.column_mapping['depth'],
                    lith_from_col=st.session_state.column_mapping['lith_from'],
                    lith_to_col=st.session_state.column_mapping['lith_to'],
                    lith_col=st.session_state.column_mapping['lithology'],
                    assay_from_col=st.session_state.column_mapping['assay_from'],
                    assay_to_col=st.session_state.column_mapping['assay_to'],
                    assay_value_col=st.session_state.column_mapping['assay_value']
                )
                
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.error("Impossible de créer la visualisation 3D avec les données fournies.")
            else:
                st.info("Veuillez sélectionner au moins un forage à afficher.")