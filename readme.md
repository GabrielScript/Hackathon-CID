# 📊 Decifrando o Profissional de Dados: Uma Análise de Vagas no LinkedIn

![Badge Hackathon](https://img.shields.io/badge/Projeto-Hackathon%20%5BNome%20do%20Hackathon%5D-blue?style=for-the-badge)

## 🎯 1. Objetivo do Projeto

Este projeto foi desenvolvido durante o **carrers in data** com o objetivo principal de analisar o mercado de trabalho para profissionais de dados (Data Scientists, Data Analysts, Data Engineers, etc.).

Utilizando um conjunto de dados de vagas publicadas no LinkedIn, buscamos identificar e dissecar quais são as **características, habilidades técnicas (hard skills) e habilidades comportamentais (soft skills)** mais esperadas e valorizadas pelas empresas atualmente.

## ❓ 2. O Problema

O campo de dados está em constante e rápida evolução. Para aspirantes que desejam entrar na área e para profissionais que buscam se manter atualizados, surge a pergunta central:

> **"Quais são as competências essenciais que definem um profissional de dados competitivo no mercado atual?"**

Este projeto visa responder a essa pergunta por meio de uma análise quantitativa e qualitativa das descrições de vagas reais.

## 💾 3. Fonte de Dados

A análise é baseada no conjunto de dados público **"LinkedIn Data Jobs Dataset"**, disponível na plataforma Kaggle.

* **Link do Dataset:** [Kaggle - LinkedIn Data Jobs Dataset](https://www.kaggle.com/datasets/joykimaiyo18/linkedin-data-jobs-dataset)

Este dataset agrega milhares de postagens de vagas, contendo informações valiosas como:
* Título do Cargo (Job Title)
* Descrição da Vaga (Job Description)
* Empresa (Company)
* Localização (Location)
* Nível de Senioridade (Seniority Level)
* E outras informações relevantes.

## 🛠️ 4. Metodologia e Análise

Nossa abordagem para extrair insights dos dados seguiu as seguintes etapas:

1.  **Pré-processamento e Limpeza:**
    * Carregamento dos dados.
    * Tratamento de valores ausentes (NaN) e duplicados.
    * Padronização de textos (lowercase, remoção de stopwords).

2.  **Análise Exploratória de Dados (EDA):**
    * Distribuição de vagas por cargo (Analista vs. Cientista vs. Engenheiro).
    * Análise de vagas por nível de senioridade (Júnior, Pleno, Sênior).
    * Principais empresas contratando.
    * Distribuição geográfica das vagas.

3.  **Processamento de Linguagem Natural (NLP) & Extração de Habilidades:**
    * Utilização de técnicas de NLP (como TF-IDF e N-grams) para extrair as *keywords* mais frequentes das descrições.
    * Mapeamento de *Hard Skills*: Identificação de tecnologias (Python, SQL, R, Tableau, Power BI, Spark, AWS, etc.).
    * Mapeamento de *Soft Skills*: Identificação de competências (Comunicação, Liderança, Proatividade, Resolução de Problemas, etc.).

4.  **Visualização:**
    * Criação de gráficos e dashboards para comunicar os resultados de forma clara e objetiva.

## 📈 5. Principais Descobertas (Resultados)

> 🚧 **Análise em andamento...** 🚧
> *Esta seção será preenchida com os principais insights encontrados durante o hackathon.*

* **Quais são as 10 Hard Skills mais demandadas?**
    * *Ex: 1. SQL, 2. Python...*
* **Qual a proporção de vagas que exigem "Python" vs. "R"?**
    * *...*
* **Quais as Soft Skills mais citadas nas descrições de vagas?**
    * *...*
* **Visualização Principal (Dashboard):**
    * *[Inserir link do dashboard ou imagem estática]*

## 🚀 6. Tecnologias Utilizadas

Este projeto foi construído utilizando as seguintes ferramentas e bibliotecas:

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FFFFFF?style=for-the-badge&logo=matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-88B0C3?style=for-the-badge&logo=seaborn&logoColor=white)
## 📂 7. Como Executar o Projeto

Para replicar esta análise, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-REPOSITÓRIO]
    cd [NOME-DO-REPOSITÓRIO]
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/Mac
    .\venv\Scripts\activate   # Para Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Obs: Pode ser necessário baixar os pacotes de stopwords do NLTK:*
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

4.  **Execute os Notebooks:**
    Abra o Jupyter Lab (ou VS Code) e execute os notebooks na ordem (ex: `01_Limpeza.ipynb`, `02_Analise.ipynb`).
    ```bash
    jupyter lab
    ```

## 👥 8. Equipe

