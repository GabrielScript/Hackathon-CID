import os
import logging
import zipfile  # Usaremos para descompactar manualmente
import glob     # Usaremos para encontrar o arquivo .zip
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configurações ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_SLUG = 'arshkon/linkedin-job-postings'
DOWNLOAD_PATH = BASE_DIR / 'data'
# ---------------------

def setup_logging():
    """Configura o logging básico."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def extract_data_from_kaggle(dataset, destination):
    """
    Função 'Extract' com debug.
    
    Baixa o .zip e o extrai manualmente, logando cada etapa.
    """
    logging.info(f"Iniciando a etapa 'EXTRACT' para o dataset: {dataset}")
    abs_path = None # Definido fora do try para ser usado no 'finally'
    
    try:
        # 1. Autentica na API do Kaggle
        api = KaggleApi()
        api.authenticate()
        logging.info("Autenticação com a API do Kaggle bem-sucedida.")

        # 2. Resolve o caminho absoluto e garante que o diretório exista
        abs_path = os.path.abspath(destination)
        os.makedirs(abs_path, exist_ok=True)
        logging.info(f"Diretório de destino: {abs_path}")

        # 3. Baixa os arquivos (SEM extrair automaticamente)
        logging.info(f"Baixando arquivo .zip para {abs_path}...")
        api.dataset_download_files(
            dataset=dataset,
            path=abs_path,
            unzip=False,  # <--- MUDANÇA IMPORTANTE
            force=True    # Força o download mesmo que já exista (bom para debug)
        )
        
        logging.info("Comando de download enviado. Verificando o arquivo .zip...")

        # 4. Encontra o arquivo .zip que acabou de ser baixado
        # O Kaggle nomeia o zip com base no nome do dataset, ex: 'linkedin-job-postings.zip'
        # Usamos o glob para encontrar qualquer .zip na pasta
        zip_files = glob.glob(os.path.join(abs_path, "*.zip"))
        
        if not zip_files:
            logging.error("--- ERRO DE DOWNLOAD ---")
            logging.error("O arquivo .zip NÃO foi encontrado após o download.")
            logging.error("Verifique as permissões da pasta ou o espaço em disco.")
            return

        zip_file_path = zip_files[0]
        logging.info(f"Arquivo .zip encontrado: {zip_file_path}")

        # 5. Extrai o .zip manualmente
        logging.info(f"Iniciando extração de {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(abs_path)
        
        logging.info(f"Extração concluída para {abs_path}.")

        # 6. Limpa o arquivo .zip após a extração
        logging.info(f"Removendo arquivo .zip: {zip_file_path}")
        os.remove(zip_file_path)

        logging.info("="*30)
        logging.info(" 'EXTRACT' CONCLUÍDO COM SUCESSO!")
        logging.info("="*30)

    except Exception as e:
        logging.error(f"--- FALHA NA ETAPA 'EXTRACT' ---")
        logging.error(f"Erro: {e}", exc_info=True) # exc_info=True mostra mais detalhes
        
    finally:
        # 7. VERIFICAÇÃO FINAL (Roda mesmo se o 'try' falhar)
        logging.info(f"--- DEBUG FINAL: Verificando conteúdo de {destination} ---")
        if abs_path and os.path.exists(abs_path):
            try:
                files = os.listdir(abs_path)
                if not files:
                    logging.warning("O diretório de destino está VAZIO.")
                else:
                    logging.info(f"Arquivos encontrados ({len(files)}): {files}")
            except Exception as e:
                logging.error(f"Não foi possível ler o diretório {abs_path}: {e}")
        else:
            logging.error("O diretório de destino NÃO FOI CRIADO ou não foi encontrado.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    setup_logging()
    extract_data_from_kaggle(DATASET_SLUG, DOWNLOAD_PATH)