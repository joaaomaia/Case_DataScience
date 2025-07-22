import os

# Estrutura de diretórios
folders = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "reports/figures",
    "tests"
]

# Arquivos iniciais (caminho: conteúdo)
files = {
    ".gitignore": """# Dados
data/
*.csv
*.parquet

# Jupyter
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
*.egg-info/

# VSCode
.vscode/
""",

    "README.md": "# Case Data Science – Predição de Target\n\nEste projeto visa resolver um desafio de predição de variável target, com foco em qualidade de modelagem, explicabilidade e padronização.",

    "environment.yml": """name: case_ds
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - jupyterlab
  - shap
  - lightgbm
  - xgboost
  - pip:
      - notebook
""",

    "Makefile": """all: setup data eda model report

setup:
\t@echo "Configurar ambiente, instalar dependências"

data:
\t@echo "Processar dados"

eda:
\t@echo "Executar análise exploratória"

model:
\t@echo "Treinar e avaliar modelo"

report:
\t@echo "Gerar relatório final"
""",

    "src/__init__.py": "",
    "tests/__init__.py": "",
    "reports/case_report.md": "# Relatório Final do Case\n\nDocumente aqui metodologia, modelagem, resultados e conclusões.",
}

# Criar diretórios
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Criar arquivos
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Estrutura inicial criada com sucesso.")
