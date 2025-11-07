import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from typing import List, Dict, Any

# Tenta importar a função de limpeza da nossa pasta 'src'
try:
    from src.utils import clean_text
except ImportError:
    print("Erro: Não foi possível encontrar 'src/utils.py'.")
    print("Certifique-se de que a pasta 'src' e 'src/utils.py' existem.")
    print("E execute o app a partir da raiz do projeto com: python -m streamlit run app/app.py")
    # Definição fallback em caso de erro (não ideal, mas para o script funcionar)
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text.lower().strip()


# --- 1. Constantes e Configurações ---
# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Paths de Input/Output
INPUT_CSV_PATH = DATA_DIR / "postings.csv"

# --- CORREÇÃO AQUI ---
# "Mapa" (Dicionário: skill_abr -> skill_name)
SKILL_MAP_PATH = DATA_DIR / "mappings" / "skills.csv" 

# "Ponte" (Conexão: job_id -> skill_abr)
JOB_SKILL_PONTE_PATH = DATA_DIR / "jobs" / "job_skills.csv"
# --- FIM DA CORREÇÃO ---

OUTPUT_DF_PATH = ARTIFACTS_DIR / "vagas_limpas.parquet"
OUTPUT_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
OUTPUT_MATRIX_PATH = ARTIFACTS_DIR / "vagas_matrix.pkl"

# Parâmetros
TFIDF_MAX_FEATURES = 5000

# Colunas de Texto (agora inclui as skills estruturadas)
TEXT_COLS = ['title', 'description', 'skills_desc', 'skills_estruturadas']

FILTER_COLS = [
    'formatted_experience_level', 
    'normalized_salary', 
    'remote_allowed'
]
DISPLAY_COLS = [
    'job_id', 
    'title', 
    'company_name', 
    'job_posting_url', 
    'med_salary', 
    'location'
]


# --- 2. Funções de Responsabilidade Única (SRP) ---

def load_data(path: Path) -> pd.DataFrame:
    """Carrega os dados do CSV e limpa os nomes das colunas."""
    print(f"Carregando dados de {path}...")
    try:
        # Tenta detectar o BOM (Byte Order Mark)
        df = pd.read_csv(path, encoding='utf-8-sig')
        
        # --- INÍCIO DA CORREÇÃO ---
        # Limpa os nomes das colunas: remove espaços, \ufeff, e põe minúsculas
        df.columns = df.columns.str.strip().str.lower()
        
        # Garante que 'job_id' esteja limpo
        df = df.rename(columns={col: 'job_id' for col in df.columns if 'job_id' in col})
        # --- FIM DA CORREÇÃO ---
        
        print(f"Dados de {path.name} carregados com sucesso e colunas limpas.")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {path}")
        raise
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}")
        raise

def normalize_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza as colunas que serão usadas para filtragem."""
    print("Normalizando features de filtro...")
    
    # Nível de Experiência
    exp_map = {
        'Entry level': 'junior',
        'Associate': 'pleno',
        'Mid-Senior level': 'senior',
        'Director': 'senior',
        'Executive': 'senior'
    }
    # Usamos .get para mapear, mantendo o original se não estiver no mapa
    df['formatted_experience_level'] = df['formatted_experience_level'].apply(
        lambda x: exp_map.get(x, 'na')
    )
    
    # Salário
    df['normalized_salary'] = pd.to_numeric(df['normalized_salary'], errors='coerce').fillna(0)
    
    # Remoto
    df['remote_allowed'] = df['remote_allowed'].fillna(False).astype(bool)
    
    print("Features de filtro normalizadas.")
    return df


# --- NOVO ---
def enrich_with_structured_skills(df_vagas: pd.DataFrame, skill_map_path: Path, ponte_path: Path) -> pd.DataFrame:
    """Enriquece o DataFrame de vagas com skills estruturadas."""
    print("Enriquecendo vagas com skills estruturadas...")
    try:
        # 1. Carregar os arquivos auxiliares
        df_mapa_skills = load_data(skill_map_path) # (skill_abr, skill_name)
        df_ponte = load_data(ponte_path)           # (job_id, skill_abr)

        # 2. Fazer o primeiro Merge (Ponte + Mapa)
        # Isso cria a tabela: [job_id, skill_abr, skill_name]
        df_skills_com_nome = df_ponte.merge(df_mapa_skills, on='skill_abr', how='left')

        # 3. Agrupar por vaga (job_id)
        # Isso transforma várias linhas por vaga em uma só
        # Ex: "Design Advertising Marketing"
        df_skills_agrupadas = df_skills_com_nome.groupby('job_id')['skill_name'].apply(
            lambda x: ' '.join(x)
        ).reset_index()
        
        # Renomear para clareza
        df_skills_agrupadas = df_skills_agrupadas.rename(columns={'skill_name': 'skills_estruturadas'})

        # 4. Juntar com o DataFrame principal de vagas
        df_vagas = df_vagas.merge(df_skills_agrupadas, on='job_id', how='left')
        
        # Preenche vagas que não tinham skills com uma string vazia
        df_vagas['skills_estruturadas'] = df_vagas['skills_estruturadas'].fillna('')
        
        print("Enriquecimento de skills concluído.")
        return df_vagas

    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos de skill: {e}")
        print("Continuando o processamento sem as skills estruturadas.")
        df_vagas['skills_estruturadas'] = '' # Cria coluna vazia para o resto do script funcionar
        return df_vagas
    except Exception as e:
        print(f"Erro inesperado no enriquecimento de skills: {e}")
        df_vagas['skills_estruturadas'] = ''
        return df_vagas
# --- FIM DO NOVO ---


def create_combined_text_feature(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """Cria a 'super-coluna' de texto para o TF-IDF."""
    print(f"Combinando colunas de texto: {text_cols}...")
    
    # Garante que todas as colunas existem e preenche NaNs
    df_text = pd.DataFrame()
    for col in text_cols:
        if col in df.columns:
            df_text[col] = df[col].fillna("")
        else:
            print(f"Aviso: coluna '{col}' não encontrada, será ignorada.")
            
    # Aplica a limpeza
    for col in df_text.columns:
        df_text[col] = df_text[col].apply(clean_text)
        
    # Cria a "super-coluna"
    df['texto_vaga'] = df_text.apply(lambda row: ' '.join(row), axis=1)
    
    print("Coluna 'texto_vaga' criada.")
    return df


def train_vectorizer(text_series: pd.Series, max_features: int) -> TfidfVectorizer:
    """"Treina" (ajusta) o TfidfVectorizer."""
    print(f"Treinando TfidfVectorizer com max_features={max_features}...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True
    )
    vectorizer.fit(text_series)
    
    print("Vetorizador treinado.")
    return vectorizer


def save_artifacts(artifacts: Dict[str, Any]):
    """Salva todos os artefatos de saída (DF, Vetorizador, Matriz)."""
    print("Salvando artefatos...")
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Colunas de filtro + Colunas de display + ID
        all_cols_to_save = list(set(['job_id'] + FILTER_COLS + DISPLAY_COLS))
        cols_existentes = [col for col in all_cols_to_save if col in artifacts['df'].columns]
        
        df_final = artifacts['df'][cols_existentes]
        df_final.to_parquet(OUTPUT_DF_PATH, index=True) # <-- Mudei para index=True
        print(f"DataFrame limpo salvo em {OUTPUT_DF_PATH}")

        joblib.dump(artifacts['vectorizer'], OUTPUT_VECTORIZER_PATH)
        print(f"Vetorizador salvo em {OUTPUT_VECTORIZER_PATH}")
        
        joblib.dump(artifacts['matrix'], OUTPUT_MATRIX_PATH)
        print(f"Matriz TF-IDF salva em {OUTPUT_MATRIX_PATH}")

    except Exception as e:
        print(f"Erro ao salvar artefatos: {e}")
        raise

# --- 3. Orquestrador (main) ---

def main():
    """Orquestra o pipeline completo da Fase Offline."""
    
    # 1. Carregar
    df_vagas = load_data(INPUT_CSV_PATH)
    
    # 2. Limpar (Filtros)
    df_vagas = normalize_filter_features(df_vagas)
    
    # 3. Enriquecer (Skills) # <-- NOVO
    df_vagas = enrich_with_structured_skills(
        df_vagas, 
        SKILL_MAP_PATH, 
        JOB_SKILL_PONTE_PATH
    )
    
    # 4. Limpar (Texto)
    df_vagas = create_combined_text_feature(df_vagas, TEXT_COLS)
    
    # 5. Treinar
    vectorizer = train_vectorizer(df_vagas['texto_vaga'], TFIDF_MAX_FEATURES)
    
    # 6. Transformar
    vagas_matrix = vectorizer.transform(df_vagas['texto_vaga'])
    
    # 7. Salvar
    artifacts_to_save = {
        'df': df_vagas,
        'vectorizer': vectorizer,
        'matrix': vagas_matrix
    }
    save_artifacts(artifacts_to_save)
    
    print("\n--- Pré-processamento (Fase Offline) concluído! ---")
    print(f"Artefatos salvos em: {ARTIFACTS_DIR}")


# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()