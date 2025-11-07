# Recomendador de Vagas de TI

## Visão Geral

Este projeto é um sistema de recomendação de vagas de emprego na área de tecnologia, construído como parte de um desafio de hackathon. A aplicação, desenvolvida com Streamlit, permite que os usuários colem o texto de seu perfil (como o "Sobre" do LinkedIn ou um currículo) e recebam uma lista das 3 vagas mais compatíveis com suas habilidades e experiência.

O sistema utiliza um modelo de **Machine Learning** baseado em **Processamento de Linguagem Natural (NLP)** para calcular a similaridade entre o perfil do usuário e as descrições das vagas.

---

## Hierarquia do Projeto

```
/
├── .gitignore
├── ETL.ipynb
├── readme.md
├── requirements.txt
├── transformed_postings.csv
├── app/
│   ├── __init__.py
│   └── app.py
├── etl/
│   ├── extract.py
│   ├── load.py
│   └── transform.py
├── scripts/
│   ├── __init__.py
│   └── preprocess.py
└── src/
    ├── __init__.py
    └── utils.py
```

---

## Tecnologias e Bibliotecas

Este projeto utiliza um conjunto de tecnologias focadas em análise de dados, machine learning e desenvolvimento web.

**Linguagem Principal:**
- **Python 3.11+**

**Bibliotecas Principais:**
- **Streamlit:** Para a construção da interface web interativa.
- **Pandas:** Para manipulação e análise de dados.
- **Scikit-learn:** Para a criação do modelo de NLP (TF-IDF) e cálculo de similaridade (cosseno).
- **Joblib:** Para serialização e carregamento dos artefatos do modelo.
- **Kaggle:** Para o download do dataset.
- **NLTK:** Para pré-processamento de texto (tokenização, remoção de stopwords).
- **PyArrow:** Para manipulação eficiente de dados em formato Parquet.
- **NumPy:** Para operações numéricas.

---

## Lógica do Modelo de Machine Learning

O coração do projeto é um sistema de recomendação baseado em conteúdo que segue os seguintes passos:

### 1. Fonte de Dados
- Os dados foram obtidos do Kaggle, do dataset **LinkedIn Jobs Listing 2023**.
- O download é feito através da API do Kaggle.

### 2. Pipeline de ETL (Extração, Transformação e Carga)
O processo de ETL é definido no notebook `ETL.ipynb` e implementado no script `scripts/preprocess.py`.

- **Extração:** Os dados brutos são carregados a partir do arquivo CSV baixado do Kaggle.
- **Transformação e Limpeza:**
    - **Seleção de Colunas:** Apenas colunas relevantes como `title`, `description`, `company_name`, `location`, etc., são mantidas.
    - **Limpeza de Texto:** A descrição das vagas passa por um rigoroso processo de limpeza:
        - Conversão para minúsculas.
        - Remoção de pontuação, caracteres especiais e URLs.
        - Remoção de *stopwords* (palavras comuns como "e", "ou", "de") utilizando a biblioteca NLTK.
    - **Normalização de Salário:** Os salários são normalizados para um valor anual para permitir a filtragem.
    - **Engenharia de Features:** O texto limpo da descrição da vaga se torna a principal *feature* para o modelo.

### 3. Feature Extraction (TF-IDF)
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Esta técnica é utilizada para converter o texto das descrições das vagas em vetores numéricos.
- O `TfidfVectorizer` do Scikit-learn é treinado com o corpus de todas as descrições de vagas. Ele aprende o vocabulário e a importância de cada palavra.
- O resultado é uma **matriz TF-IDF**, onde cada linha representa uma vaga e cada coluna representa uma palavra do vocabulário, com seu valor TF-IDF.

### 4. Mecanismo de Recomendação (Similaridade de Cosseno)
- Quando um usuário insere seu perfil, o texto também passa pelo mesmo processo de limpeza.
- O texto do usuário é então transformado em um vetor TF-IDF usando o vetorizador já treinado.
- A **similaridade de cosseno** é calculada entre o vetor do perfil do usuário e todos os vetores de vagas na matriz TF-IDF.
- O resultado é um *score* de similaridade (entre 0 e 1) para cada vaga. As vagas com os maiores scores são as mais recomendadas.

### 5. Artefatos Gerados
O script `scripts/preprocess.py` salva três artefatos essenciais na pasta `artifacts/`:
1.  `vagas_limpas.parquet`: O DataFrame com os dados das vagas já limpos e processados.
2.  `tfidf_vectorizer.pkl`: O objeto `TfidfVectorizer` treinado.
3.  `vagas_matrix.pkl`: A matriz TF-IDF esparsa com os vetores de todas as vagas.

---

## Como Executar o Projeto

Siga os passos abaixo para configurar e executar a aplicação em sua máquina local.

### Passo 1: Pré-requisitos
- **Git** instalado.
- **Python 3.11** ou superior.

### Passo 2: Clonar o Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd Hackathon-CID
```

### Passo 3: Configurar o Ambiente Virtual
É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente
# No Windows
venv\Scripts\activate
# No macOS/Linux
source venv/bin/activate
```

### Passo 4: Instalar as Dependências
Instale todas as bibliotecas necessárias com um único comando:

```bash
pip install -r requirements.txt
```

### Passo 5: Baixar os Dados do Kaggle

**1. Obtenha seu Token da API do Kaggle:**
   - Faça login no [Kaggle](https://www.kaggle.com).
   - Clique no ícone da sua conta no canto superior direito e vá para **"Settings"**.
   - Role a página até a seção **"API"**.
   - Clique no botão **"Create New API Token"**. Isso fará o download de um arquivo chamado `kaggle.json`.

**2. Configure o Token:**
   - **No Windows:**
     - Crie uma pasta chamada `.kaggle` no seu diretório de usuário (ex: `C:\Users\<SeuNome>\.kaggle`).
     - Mova o arquivo `kaggle.json` para dentro desta pasta.
   - **No macOS/Linux:**
     - Crie a pasta: `mkdir -p ~/.kaggle`
     - Mova o arquivo: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
     - Defina as permissões corretas: `chmod 600 ~/.kaggle/kaggle.json`

**3. Baixe e Descompacte o Dataset:**
Execute o comando abaixo na raiz do projeto para baixar os dados:

```bash
kaggle datasets download -d arshsisodiya/linkedin-jobs-listing-2023 -p data/raw --unzip
```
*Isso irá baixar e descompactar os dados na pasta `data/raw`.*

### Passo 6: Executar o Pré-processamento
Este script irá limpar os dados e criar os artefatos do modelo de ML necessários para a aplicação.

```bash
python scripts/preprocess.py
```
Ao final, você verá uma pasta `artifacts/` na raiz do projeto com os arquivos do modelo.

### Passo 7: Executar a Aplicação Streamlit
Finalmente, inicie a aplicação web:

```bash
streamlit run app/app.py
```

A aplicação estará disponível no seu navegador no endereço `http://localhost:8501`.
