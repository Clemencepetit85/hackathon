import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
from streamlit_card import card
import time
from streamlit_option_menu import option_menu
from sklearn.impute import SimpleImputer
import os
import psutil

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re
from streamlit_carousel import carousel

import streamlit as st




os.environ["LOKY_MAX_CPU_COUNT"] = str(psutil.cpu_count(logical=False))


# Appeler set_page_config() en premier

st.set_page_config(
    page_title="Veggie Delights",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# pour lancer le code streamlit : streamlit run recipe_reco_app_avec_rand.py

# ouverture des fichiers csv nécessaires à la création de l'application (j'effectue certaines modifications)

#import des datasets viande et vg

df_full = pd.read_csv("df_yann_10h15.csv", sep=";",encoding='utf-8')
df_full['aggregaterating_x'] =  (df_full['aggregaterating_x'] * 5).round(2)
df_full = df_full.drop(columns = ['ingredient_recherche','URL_x','cookingTime_x',
                                   'nutritionalFacts_x', 'name_y','URL_y', "Description_y","Cooking time",'ingredients_y'
                                   ,'cookingTime_y', 'likes_y','negativeFeedbacks_y',
                                     'nutritionalFacts_y', 'positiveFeedbacks_y','video_y','url_y',
                                     'vegetarien_y', 'vegan_y', 'aggregaterating_y', 'image_y','pricePerPortionTag_y', 'difficulty_y'])


#df_full.columns = df_full.columns.str.replace(' ', '_')


mots_non_vegetariens = [
    'viande', 'poisson', 'volaille', 'crustacé', 'fruit de mer',
    'porc', 'bœuf', 'poulet', 'canard', 'agneau', 'veau', 'saumon', 'truite', 'thon',
    'sardine', 'maquereau', 'anchois', 'hareng', 'dorade', 'morue', 'crabe', 'homard',
    'crevette', 'moule', 'huître', 'escargot', 'foie gras', 'jambon', 'saucisse', 'bacon',
    'steak', 'filet', 'rôti', 'escalope', 'coquille saint-jacques', 'terrine', 'bavette', 'lard',
    'saint-jacques', 'haddock', 'cabillaud', 'dinde'
]

df_full['vegetarien_x'] = df_full['words_ingredients'].apply(lambda x: 0 if any(word in x.lower() for word in mots_non_vegetariens) else 1)




# Sélection des lignes où 'vegetarien' est True
df_vg_test = df_full.loc[df_full['vegetarien_x'] == 1].reset_index(drop=True)

# Sélection des lignes où 'vegetarien' est False
df_viande_test = df_full.loc[df_full['vegetarien_x'] == 0].reset_index(drop=True)



# préparer les variables pour le système de recommandation par caractéristique

    
df_viande_test["vegan_x"] = df_viande_test["vegan_x"].astype(int)
df_vg_test["vegan_x"] = df_vg_test["vegan_x"].astype(int)
df_viande_test[ 'vegetarien_x'] = df_viande_test[ 'vegetarien_x'].astype(int)
df_vg_test[ 'vegetarien_x'] = df_vg_test[ 'vegetarien_x'].astype(int)
#df_vg_test.dropna(inplace=True)#suppression des lignes vides
#df_viande_test.dropna(inplace=True)#suppression des lignes vides



############################################################################################

# liste des recettes viande : pour selectbox

liste_viande  = df_viande_test.name_x

# liste des recettes végé : pour selectbox

#liste_vg  = df_vg_test.Nom_du_plat

# configuraton de streamlit

#st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

# permet de connecter l'application avec fichier css

#with open('style.css') as f:
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# pour avoir les animations

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# afficher les annimation et le text

# menu et lottie

# Appliquer un style CSS à la sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        font-family: serif; /* Utilisation de la typographie serif */
        /* Autres styles que vous souhaitez appliquer */
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Contenu de la sidebar avec le code pour l'animation et le menu
with st.sidebar:
    # Votre code pour l'animation
    url1 = "https://lottie.host/9f5df26b-1fa9-445c-aae0-432fca6ebc11/vYnbX7bQXg.json"
    lottie1 = load_lottieurl(url1)
    st_lottie(lottie1, key="jj", width=300, height=300)

    # Composant du menu
    selected2 = option_menu(
        "Menu", ["Page d'accueil", "Recommandation de recettes végétariennes par caractéristiques",
                  "Recommandation de recettes végétariennes par contenu", "Besoin d'inspiration ?", 
                  "Votre recette pas à pas",
                  "Pourquoi limiter la consommation de viande ?"], 
        icons=["caret-right-fill", "caret-right-fill", "caret-right-fill", "caret-right-fill", "caret-right-fill", "caret-right-fill"], 
        menu_icon="bi-gift-fill", default_index=0, orientation="vertical"
    )

############################################################################################


# configuration de la page d'accueil 

if selected2 == "Page d'accueil":
     
     
    st.image("images/logo.png", width=300)
    st.title("Bienvenue sur l'application Veggie Delights")
    #  on peut rajouter une balise html  avec markdown
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Les fêtes de fin d'année arrivent à grands pas ? Vous cherchez à limiter votre consommation de viande et de poisson ?
                Notre application transforme chaque repas en une célébration végétarienne inoubliable.
                Découvrez des alternatives créatives et délicieuses pour tous vos plats carnés de fin d'année.
                De délicats hors-d'œuvre aux plats principaux somptueux, laissez-nous réinventer vos festivités avec des saveurs végétales sensationnelles.
                Faites de chaque repas une fête pour les papilles, sans compromis sur le plaisir ni sur la qualité.
                """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<br>➜ Sur l'onglet "Recommandation de recettes végétariennes par caractéristiques", vous trouverez des alternatives à vos recettes carnées préférées basées sur les caractéristiques suivantes :
                durée de préparation, durée de cuisson, note sur Jow.fr, niveau de difficulté, niveau de prix et calories par portion.<br>
                ➜ Sur l'onglet "Recommandation de recettes végétariennes par contenu", vous trouverez des alternatives à vos recettes carnées préférées basées sur les caractéristiques suivantes :
                nom du plat, description et ingrédients.<br>
                ➜ Sur l'onglet "Besoin d'inspiration ?", obtenez en un clic une idée de recette végétarienne pour les fêtes !<br>
                ➜ Sur l'onglet "Pourquoi limiter la consommation de viande ?", retrouvez des graphiques et informations claires démontrant l'impact du végétarisme sur la planête et la santé.
                """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image("images/repas-de-noel-vegetarien-nos-idees-recettes-faciles.jpeg", width=300)


    # Utilisation de Markdown pour mise en forme du lien
    if st.markdown(
        """
        <style>
        /* Style pour le lien */
        .custom-link {
            color: #000000; /* Couleur du texte */
            text-decoration: none; /* Supprimer le soulignement */
        }
        </style>
        """
        , unsafe_allow_html=True
    ):
        pass  # Action à effectuer lors du clic, ici, ne rien faire

    # Affichage du lien avec la classe CSS pour le style personnalisé
    st.markdown(
        """
            👨‍🍳 Application réalisée en novembre 2023 par Clémence Petit, Yann Floquet et Sevan Doizon, étudiants en Data Analyse à la Wild Code School.
            <br><a href="https://jow.fr/" class="custom-link" target="_blank" rel="noopener noreferrer">Nos recettes sont toute issues du site internet de Jow.
        </a>
        """
        , unsafe_allow_html=True
    )


############################################################################################

# configuration de la page recommandation par genre

elif selected2 == "Recommandation de recettes végétariennes par caractéristiques":
    st.title('Recommandation de recettes végétariennes par caractéristiques')
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Si vous souhaitez remplacer un plat ou une entrée contenant de la viande par son alternative végétarienne, vous êtes au bon endroit !
                Ici, les recettes similaires recommandées se basent sur les caractéristiques suivantes : durée de préparation, durée de cuisson, note sur Jow.fr, niveau de difficulté, niveau de prix et calories par portion.
                """)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Configuration de la sélection d'une recette
 
    with st.container():
        st.markdown('''Choisir un plat contenant de la viande ou du poisson :''')
        option = st.selectbox(' ', liste_viande)

        with st.expander("Voir les détails du plat"):
            df_input = df_viande_test[df_viande_test.name_x == option]
          
            # Renommer les colonnes et sélectionner les colonnes à afficher
            colonnes_renommees = {
                "name_x": "Nom du plat ",
                "Description_x": "Description ",
                "Cooking_time" : "Durée de préparation en minutes ",
                "difficulty_x": "Niveau de difficulté (sur 3) ",
                "url_x": "Lien vers la recette "
            }

            colonnes_selectionnees = ["name_x", "Description_x", "Cooking_time", "difficulty_x", "url_x"]

            df_details = df_input[colonnes_selectionnees].rename(columns=colonnes_renommees)
            
            # Utilisation des colonnes pour afficher l'image et les détails côte à côte
            
            col1, col2 = st.columns([1, 2])

            with col1:
                image_url = df_input["image_x"].values[0]
                try:
                    st.image(image_url, width=180)
                except:
                    st.error("Pas de photo disponible")

            with col2:
                
                for col in df_details.columns:
                    
                    st.markdown(
                        f"-  <span style='color: #424b54; font-weight: bold;'>{col}</span> : **{df_details[col].values[0]}**",
                        unsafe_allow_html=True)


    

  





    #definition des données pour la reco

    #X_viande = df_viande_test.select_dtypes(include = "number")
    #X_vg = df_vg_test.select_dtypes(include = "number")

    #choix des colonnes pour la reco
    
    #st.write(df_vg_test)
    #st.write(df_details)
    #st.write(df_input)
    colonnes_a_garder = ['Preparation extra time per cover', 'difficulty_x', 'Cooking_time',
                         'likes_x', 'negativeFeedbacks_x', 'positiveFeedbacks_x', 'CHOAVL/Glucides',
                         'ENERC/Calories', 'FAT/Matiere_graces', 'FIBTG/Fibres', 'PRO/Proteines',"aggregaterating_x"]
    X_viande = df_input.loc[:, colonnes_a_garder]
    
    
    X_vg = df_vg_test.loc[:, colonnes_a_garder]

    #st.write(X_viande)
    #normaliser pr mettre à une meme echelle

    scaler = MinMaxScaler()
    # Appliquez le scaler aux données
    X_scale_viande = scaler.fit_transform(X_viande)
    #appliquer la meme chose pr les pok leg
    X_scale_vg = scaler.transform(X_vg)

    #recommandation

    #recherche dans les plats de viande
    distanceKNN = NearestNeighbors(n_neighbors=3).fit(X_scale_vg)

    #chercher les voisins vg
    result = distanceKNN.kneighbors(X_scale_viande)
    distances, ids =  distanceKNN.kneighbors(X_scale_viande)
    #print(ids)

    #ids_filtered = []

    #for i in range(len(ids)):
      #  valid_ids = [x for x in ids[i] if x < len(df_viande_test)]
      #  ids_filtered.append(valid_ids)
      #  print(ids_filtered)
    
    if st.button("Voir les recommandations"):
        with st.spinner("Recherche en cours..."):
            time.sleep(1)

# Utilisation des colonnes pour afficher l'image et les détails côte à côte
       
       
        flat = ids[0]
        reco = df_vg_test.iloc[flat,:]
     
        for i in range(3):
            
            with st.expander(f"Recommandation {i+1}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    try:
                        # Utilisation des indices individuels extraits des tuples dans ids
                        
                    
                        
                        

                        st.image(reco['image_x'].values[i], width=280)
                    except:
                        st.error("Pas de photo disponible")

        

                with col2:
                    
                    for col in reco[colonnes_selectionnees].columns:
                        
                    # Use the renaming dictionary to get the desired column name
                        renamed_col = colonnes_renommees.get(col, col)
                        st.markdown(f"- <span style='color: #424b54; font-weight: bold;'>{renamed_col}</span> : **{reco[col].values[i]}**",
                                unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                #st.download_button("Télécharger les infos de la recette", df_details_recommended[col]) #à tester

############################################################################################


# configuration de la page recommandation par thème

elif selected2 == "Recommandation de recettes végétariennes par contenu":
    
    st.title('Recommandation de recettes végétariennes par contenu')
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Si vous souhaitez remplacer un plat ou une entrée contenant de la viande par son alternative végétarienne, vous êtes au bon endroit !
                Ici, les recettes similaires recommandées se basent sur les caractéristiques suivantes : nom du plat, description et ingrédients.
                """)
    st.markdown("<hr>", unsafe_allow_html=True)

    # configuration de la sélection d'un plat

    with st.container() :
        
            st.markdown('''Choisir un plat contenant de la viande ou du poisson :''')
            option_txt = st.selectbox(' ',liste_viande)
           
            
            with st.expander("Voir les détails du plat :"):
                df_input_txt = df_viande_test[df_viande_test.name_x == option_txt]

               # Renommer les colonnes et sélectionner les colonnes à afficher

                colonnes_renommees = {
                "name_x": "Nom du plat ",
                "Description_x": "Description ",
                "Cooking_time" : "Durée de préparation en minutes ",
                "difficulty_x": "Niveau de difficulté (sur 3) ",
                "url_x": "Lien vers la recette "
                }

                colonnes_selectionnees = ["name_x", "Description_x", "Cooking_time", "difficulty_x", "url_x"]

                df_details_input_txt = df_input_txt[colonnes_selectionnees].rename(columns=colonnes_renommees)
                
                col1, col2 = st.columns([1, 2])

                with col1:

                    image_txt = df_input_txt["image_x"].values[0]

                    try :
                            
                        st.image(image_txt,width = 180)
                    except :
                        st.error("pas de photo disponible ")

                with col2:

                    for col in df_details_input_txt.columns:
                            st.markdown(f"- <span style='color: #424b54; font-weight: bold;'>{col}</span> : **{df_details_input_txt[col].values[0]}**",
                                        unsafe_allow_html=True)
                            
            

    

    # configuration de l'affichage et du système de recommandation     
    # préparation des variables pour la recommandation par contenu
    
   
    def clean_text(cell):
        # Convertir en minuscules
        cleaned_text = cell.lower()
        
        # Supprimer les caractères spéciaux et les virgules, conserver les accents
        cleaned_text = ''.join(c for c in unicodedata.normalize('NFD', cleaned_text) if unicodedata.category(c) != 'Mn')
        cleaned_text = re.sub(r'[^a-z, ]+', ' ', cleaned_text)
        
        return cleaned_text

    # Appliquer la fonction de nettoyage aux colonnes concernées et créer de nouvelles colonnes nettoyées
    columns_to_clean = ["name_x", "Description_x", "ingredients_x", "words_ingredients"]

    for col in columns_to_clean:
        new_col_name = col + "_cleaned"  # Nom des nouvelles colonnes nettoyées
        df_viande_test[new_col_name] = df_viande_test[col].apply(clean_text)
        df_vg_test[new_col_name] = df_vg_test[col].apply(clean_text)

    #création d'une liste de colonnes avec du contenu texte :

    liste_colonnes = ["name_x_cleaned", "Description_x_cleaned", "ingredients_x_cleaned", "words_ingredients_cleaned"] 

    #nous concaténons tout le texte dans une même colonne nommée "text_content"

    df_viande_test['text_content'] = df_viande_test[liste_colonnes].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    df_vg_test['text_content'] = df_vg_test[liste_colonnes].apply(lambda row: ' '.join(row.astype(str)), axis=1)


    # Définir la largeur maximale des colonnes de texte
    pd.set_option('display.max_colwidth', None)

    # Utiliser TfidfVectorizer pour convertir le texte en vecteurs TF-IDF en excluant les mots de liaison 

    # Liste de mots vides en français
    stop_words_french = [
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'en', 'au', 'aux', ","  # Exemple de mots vides en français
    ]

    # Utiliser TfidfVectorizer avec les mots vides en français
    vectorizer = TfidfVectorizer(stop_words=stop_words_french)
    text_matrix_viande = vectorizer.fit_transform(df_viande_test['text_content'])
    text_matrix_vg = vectorizer.transform(df_vg_test['text_content'])

    plat = option_txt

    condition = df_viande_test['name_x'].str.contains(plat) 

    df_input = df_viande_test[condition].reset_index(drop = True)

    
    # Trouver l'index de l'input dans df_viande_test
    input_index = df_viande_test.index[df_viande_test['name_x'] == plat][0]
    input_vector = text_matrix_viande[input_index]

    # Calculer la similarité cosinus entre l'input et les textes de df_vg_test
    similarities = cosine_similarity(input_vector, text_matrix_vg)

    # Obtenir les indices des lignes les plus similaires dans df_vg_test
    top_similar_indices = similarities.argsort(axis=1)[0][-3:]  # Obtenez les 3 lignes les plus similaires
    top_similar_rows = df_vg_test.iloc[top_similar_indices]
    #st.write(df_vg_test)
    if st.button("Voir les recommandations"):
        with st.spinner("Recherche en cours..."):
            time.sleep(1)
        
        for i, index in enumerate(top_similar_indices):
            with st.expander(f"Recommandation {i + 1}"):
                st.header(df_vg_test.iloc[index, 1])

                col1, col2 = st.columns([1, 2])

                with col1:

                    try:
                        st.image(df_vg_test.loc[index, "image_x"], width=200)
                    except:
                        st.error("Pas de photo disponible")

                with col2:

                    final = df_vg_test.loc[index, colonnes_selectionnees]
                    for col in colonnes_selectionnees:
                        renamed_col = colonnes_renommees[col]
                        value = final[col]
                        st.markdown(f"- <span style='color: #424b54; font-weight: bold;'>{renamed_col}</span> : **{value}**",
                                    unsafe_allow_html=True)



elif selected2 == "Besoin d'inspiration ?":
    # Lottie animation
    #url = "https://lottie.host/35c0a048-598b-4a3d-b088-dea216e64816/J1uxMIywwO.json"
    #st_lottie(url, speed=1, width=300, height=300)
    st.title("Besoin d'inspiration ?")
    #  on peut rajouter une balise html  avec markdown
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Cliquez sur le bouton ci-dessous pour obtenir une délicieuse recette végétarienne pour les fêtes :
                """)
    def generer_idee_recette():
        random_row = df_vg_test.sample(n=1)
        st.write(random_row)

        # Créer un bouton dans Streamlit
    if st.button('Générer une idée de recette'):
        random_row = df_vg_test.sample(n=1)
        col1, col2 = st.columns([1, 2])

        with col1:

            image_txt = random_row["image_x"].values[0]

            try :
                            
                st.image(image_txt,width = 180)
            except :
                st.error("pas de photo disponible ")

        with col2:
            colonnes_renommees = {
                "name_x": "Nom du plat ",
                "Description_x": "Description ",
                "Cooking_time" : "Durée de préparation en minutes ",
                "difficulty_x": "Niveau de difficulté (sur 3) ",
                "url_x": "Lien vers la recette "
            }

            colonnes_selectionnees = ["name_x", "Description_x", "Cooking_time", "difficulty_x", "url_x"]
            for col in  random_row[colonnes_selectionnees].columns:
                renamed_col = colonnes_renommees.get(col, col)
                st.markdown(f"- <span style='color: #424b54; font-weight: bold;'>{renamed_col}</span> : **{ random_row[col].values[0]}**",
                                        unsafe_allow_html=True)
            
elif selected2 =="Votre recette pas à pas":
    st.title("Votre recette pas à pas")
    #  on peut rajouter une balise html  avec markdown
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Sélectionnez la recette de votre choix et visionnez les étapes de la recette :
                """)
    get_name = df_full["name_x"]
    option = st.selectbox(' ', get_name)

    video = df_full.loc[df_full['name_x'] == option,'video_x'].values[0]
    st.video(video)
else :

    st.title("Pourquoi limiter la consommation de viande ?")
    #  on peut rajouter une balise html  avec markdown
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Au cours des dernières décennies, les émissions de gaz à effet de serre ont beaucoup augmenté comme vous pouvez le voir sur les graphiques suivants.
                Le végétarisme se présente comme une excellente manière d'aider à la diminution de ces émissions.
                """)
    st.markdown("<hr>", unsafe_allow_html=True)


    # Votre code pour tracer les courbes d'évolution des émissions GES
    data = {
        'Année': [1970, 1980, 1990, 2000, 2010, 2020],
        'Émissions de CO2 (millions de tonnes)': [4500, 6000, 8000, 9500, 11000, 12000],
        'Émissions de CH4 (milliers de tonnes)': [200, 220, 250, 300, 320, 350],
        'Émissions de N2O (milliers de tonnes)': [100, 120, 150, 180, 200, 220]
    }

    df_emissions = pd.DataFrame(data)

    # Tracer des courbes d'évolution des émissions de CO2, CH4 et N2O au fil des années

    import seaborn as sns
    import matplotlib.pyplot as plt


    # Tracer des courbes d'évolution des émissions de CO2, CH4 et N2O au fil des années
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8,4))  # Premier sous-plot pour les émissions de CO2
    sns.lineplot(data=df_emissions, x='Année', y='Émissions de CO2 (millions de tonnes)', color='#254E33', ax=ax)
    ax.set_title('Évolution des émissions de CO2 au fil des années')
    ax.set_xlabel('Année')
    ax.set_ylabel('Émissions de CO2 (millions de tonnes)')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8,4))  # Deuxième sous-plot pour les émissions de CH4
    sns.lineplot(data=df_emissions, x='Année', y='Émissions de CH4 (milliers de tonnes)', color='#254E33', ax=ax)
    ax.set_title('Évolution des émissions de CH4 au fil des années')
    ax.set_xlabel('Année')
    ax.set_ylabel('Émissions de CH4 (millions de tonnes)')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8,4))  # Troisième sous-plot pour les émissions de N2O
    sns.lineplot(data=df_emissions, x='Année', y='Émissions de N2O (milliers de tonnes)', color='#254E33', ax=ax)
    ax.set_title('Évolution des émissions de N2O au fil des années')
    ax.set_xlabel('Année')
    ax.set_ylabel('Émissions de N2O (millions de tonnes)')
    st.pyplot(fig)


    # Le contenu du carrousel

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""Découvrez ci-dessous les impacts du végétarisme sur notre environnement.
                """)
    st.markdown("<hr>", unsafe_allow_html=True)


    test_items = [
        {
            "title": "Moins d'émissions de gaz à effet de serre",
            "text": "Une nouvelle étude scientifique de chercheurs d'Oxford parue en juillet 2023 confirme qu'une alimentation carnée émet 75 % de gaz à effet de serre en plus qu'une diète à base de végétaux.",
            "img": "https://cdn.futura-sciences.com/cdn-cgi/image/width=1024,quality=50,format=auto/sources/images/qr/gaz-a-effet-de-serre-rechauffement-climatique-2.jpeg",
        },
        {
            "title": "Diminution de la déforestation",
            "text": "L’élevage du bétail est responsable d’environ 80% de la destruction de la forêt amazonienne et de 14% de la déforestation mondiale.",
            "img": "https://asialyst.com/fr/wp-content/uploads/2021/06/indonsie-deforestation.jpg",
        },
                {
            "title": "Une meilleure santé",
            "text": "Une nouvelle étude scientifique de chercheurs d'Oxford parue en juillet 2023 confirme qu'une alimentation carnée émet 75 % de gaz à effet de serre en plus qu'une diète à base de végétaux.",
            "img": "https://www.usine-digitale.fr/mediatheque/2/9/0/001212092_896x598_c.jpg",
        },
        {
            "title": "Amélioration du bien-être animal",
            "text": "Aujourd’hui les animaux sont élevés dans des usines, de manière peu convenable. La plupart des poules ne voient jamais la lumière du jour, les cochons ne vont jamais en extérieur (les truies reproductrices sont maintenues allongées au sol dans des cages de gestation), les vaches laitières sont fécondées 3 mois après chaque vêlage avant d’être abattue à 5 ans. Les animaux ainsi traités sont souvent plus vulnérables et donc bourrés d’antibiotiques.",
            "img": "https://www.eleusis-megara.fr/wp-content/uploads/2014/06/dfv.jpg",
        },
        # ... Autres éléments du carrousel
    ]

    for item in test_items:
        st.write(f"## {item['title']}")
        st.image(item['img'], use_column_width=True)
        st.write(item['text'])
        st.markdown("---")
