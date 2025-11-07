import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from src.utils import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
MATRIX_PATH = ARTIFACTS_DIR / "vagas_matrix.pkl"
DF_PATH = ARTIFACTS_DIR / "vagas_limpas.parquet"

@st.cache_resource  # (Clean Code: Cacheia os artefatos na memória)
def load_artifacts():
    """Carrega os artefatos (DF, Vetorizador, Matriz) uma vez."""
    print("Carregando artefatos...")
    try:
        df = pd.read_parquet(DF_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        matrix = joblib.load(MATRIX_PATH)
        print("Artefatos carregados com sucesso.")
        return df, vectorizer, matrix
    except FileNotFoundError:
        st.error(f"Erro: Arquivos de artefatos não encontrados. "
                 f"Execute 'scripts/preprocess.py' primeiro.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {e}")
        st.stop()

# Carrega os dados (só roda uma vez)
df_vagas, vetorizador, matriz_vagas_completa = load_artifacts()

# --- 2. Lógica Principal (Backend) ---

def find_top_matches(perfil_texto: str, filtros: dict, df_completo: pd.DataFrame, 
                     matriz_tfidf_completa, vectorizer_obj):
    """
    Função principal: Filtra o DF e encontra os top 3 matches por similaridade.
    (SRP: Esta função tem a única responsabilidade de fazer o matching)
    """
    
    # --- Passo 1: Filtragem Rígida (Pandas) ---
    df_filtrado = df_completo.copy()
    
    # Filtro de Nível
    if filtros['nivel'] != 'Todos':
        df_filtrado = df_filtrado[
            df_filtrado['formatted_experience_level'] == filtros['nivel']
        ]
    
    # Filtro de Salário
    if filtros['salario_min'] > 0:
        df_filtrado = df_filtrado[
            df_filtrado['normalized_salary'] >= filtros['salario_min']
        ]
        
    # Filtro Remoto
    if filtros['remoto']:
        df_filtrado = df_filtrado[df_filtrado['remote_allowed'] == True]
        
    if df_filtrado.empty:
        return pd.DataFrame() # Retorna DF vazio se nenhum filtro bateu

    # --- Passo 2: Filtragem da Matriz (Índices) ---
    indices_filtrados = df_filtrado.index
    matriz_vagas_filtradas = matriz_tfidf_completa[indices_filtrados]

    # --- Passo 3: Vetorizar o Perfil do Usuário ---
    perfil_limpo = clean_text(perfil_texto)
    
    if not perfil_limpo.strip(): # Se o perfil estiver vazio
        return pd.DataFrame() 
        
    vetor_perfil = vectorizer_obj.transform([perfil_limpo])

    # --- Passo 4: Calcular Similaridade de Cossenos ---
    scores = cosine_similarity(vetor_perfil, matriz_vagas_filtradas)
    
    # --- Passo 5: Rankear e Retornar ---
    # scores[0] porque cosine_similarity retorna um array 2D
    df_filtrado['similarity_score'] = scores[0]
    
    # Retorna o Top 3
    top_3_matches = df_filtrado.nlargest(3, 'similarity_score')
    
    return top_3_matches

# --- 3. Interface do Usuário (Streamlit) ---

st.set_page_config(page_title="Recomendador de Vagas", layout="wide")
st.title("🚀 Recomendador de Vagas da Hackathon")
st.markdown("Cole seu perfil ou currículo abaixo e encontre as vagas com maior *match*!")

# --- UI: Barra Lateral (Filtros) ---
st.sidebar.header("Seus Filtros 🔎")

# Opções de filtro (baseadas nos dados limpos)
niveis_disponiveis = ['Todos'] + list(df_vagas['formatted_experience_level'].unique())
filtro_nivel = st.sidebar.selectbox("Nível de Experiência:", options=niveis_disponiveis)

filtro_salario = st.sidebar.slider("Pretensão Salarial Mínima (R$):", 0, 20000, 0, step=1000)

filtro_remoto = st.sidebar.checkbox("Apenas vagas remotas", value=True)

# Agrupa os filtros em um dicionário (Clean Code)
filtros_usuario = {
    'nivel': filtro_nivel,
    'salario_min': filtro_salario,
    'remoto': filtro_remoto
}

# --- UI: Área Principal (Input e Output) ---

# Input do Perfil
texto_perfil_usuario = st.text_area(
    "Cole seu 'Sobre' do LinkedIn, seu currículo, ou apenas suas habilidades:",
    height=250,
    placeholder="Ex: Cientista de dados com 3 anos de experiência em Python, SQL, "
                "Scikit-learn e Power BI. Focado em modelos de classificação..."
)

# Botão de Ação
if st.button("Encontrar Vagas ✨"):
    if not texto_perfil_usuario.strip():
        st.warning("Por favor, insira o texto do seu perfil.")
    else:
        with st.spinner("Calculando o match... 🤖"):
            # Chama a função principal de backend
            top_vagas = find_top_matches(
                perfil_texto=texto_perfil_usuario,
                filtros=filtros_usuario,
                df_completo=df_vagas,
                matriz_tfidf_completa=matriz_vagas_completa,
                vectorizer_obj=vetorizador
            )
        
        # --- UI: Exibição dos Resultados ---
        if top_vagas.empty:
            st.error("Nenhuma vaga encontrada com seus critérios de filtro e perfil.")
        else:
            st.success(f"Encontramos {len(top_vagas)} vagas com alta compatibilidade!")
            
            # (Clean Code: Usando st.columns para layout)
            for index, row in top_vagas.iterrows():
                st.subheader(row['title'])
                st.markdown(f"**Empresa:** {row['company_name']} | "
                            f"**Localização:** {row['location']}")
                
                # Mostra o score de similaridade
                score_percent = row['similarity_score'] * 100
                st.progress(int(score_percent))
                st.markdown(f"**Compatibilidade:** {score_percent:.2f}%")
                
                # Link
                st.markdown(f"[Ver vaga completa]({row['job_posting_url']})", 
                            unsafe_allow_html=True)
                st.divider()