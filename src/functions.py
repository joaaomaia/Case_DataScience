import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging


def criar_novas_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    eps = 1e-6 # evita divisões por zero
    
    df['utilizacao_limite_rotativo'] = df['saldo_rotativo_total'] / (df['limite_rotativo_total'] + eps)

    df['proporcao_emprestimo_vs_limite_rotativo'] = df['valor_total_emprestimos_tomados'] / (df['limite_rotativo_total'] + eps)

    df['alavancagem_patrimonial'] = df['valor_total_emprestimos_tomados'] / (df['patrimonio_total'] + eps)

    df['taxa_recuperacao_credito'] = df['valor_total_recuperacoes_ultimos_2a'] / (df['qtd_atrasos_ultimos_2a'] + 1)

    df['atrasos_por_linhas_credito_abertas'] = df['qtd_atrasos_ultimos_2a'] / (df['qtd_linhas_credito_abertas'] + 1)

    df['consultas_por_linha_aberta'] = df['qtd_consultas_ultimos_6m'] / (df['qtd_linhas_credito_abertas'] + 1)

    df['consultas_por_emprestimo_tomado'] = df['qtd_consultas_ultimos_6m'] / (df['valor_total_emprestimos_tomados'] + eps)

    grade_ordem = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_ordinal'] = df['grade'].map(grade_ordem)

    subgrades_ordenadas = [
        'A1', 'A2', 'A3', 'A4', 'A5',
        'B1', 'B2', 'B3', 'B4', 'B5',
        'C1', 'C2', 'C3', 'C4', 'C5',
        'D1', 'D2', 'D3', 'D4', 'D5',
        'E1', 'E2', 'E3', 'E4', 'E5',
        'F1', 'F2', 'F3', 'F4', 'F5',
        'G1', 'G2', 'G3', 'G4', 'G5'
    ]
    subgrade_ordem = {sg: i + 1 for i, sg in enumerate(subgrades_ordenadas)}
    df['sub_grade_ordinal'] = df['sub_grade'].map(subgrade_ordem)

    # quanto maior, maior o desvio entre classificação principal e detalhada,
    # hipótese de que equipes trabalharam com informações distintas 
    df['diferenca_grade_subgrade'] = df['grade_ordinal'] * 5 - df['sub_grade_ordinal']

    df['renda_presumida'] = df['valor_total_emprestimos_tomados'] / (df['razao_credito_tomado_vs_renda_informada'] + eps)

    df['fonte_renda_verificada'] = df['verificacao_fonte_de_renda'].apply(lambda x: 1 if str(x).strip().lower() in ['sim', 'yes', 'true', '1'] else 0)

    #df.drop(columns=['grade_ordinal', 'sub_grade_ordinal'], inplace=True)

    return df

def correlacao_juros_alavancagem(df):
    df = df.copy()
    df = df[["alavancagem_patrimonial", "taxa_juros_media_emprestimos_tomados"]].dropna()
    df = df[(df["alavancagem_patrimonial"] < np.inf) & (df["alavancagem_patrimonial"] > -np.inf)]
    
    # Spearman (ideal para relações monotônicas, mesmo não lineares)
    corr, pval = spearmanr(df["alavancagem_patrimonial"], df["taxa_juros_media_emprestimos_tomados"])

    # Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df,
                    x="alavancagem_patrimonial",
                    y="taxa_juros_media_emprestimos_tomados",
                    alpha=0.3)
    sns.regplot(data=df,
                x="alavancagem_patrimonial",
                y="taxa_juros_media_emprestimos_tomados",
                scatter=False,
                color="red",
                line_kws={"label": f"ρ = {corr:.2f} (p={pval:.2g})"})
    
    plt.legend()
    plt.title("Correlação: Alavancagem × Taxa de Juros")
    plt.tight_layout()
    plt.show()

    return corr, pval

def gerar_regra_juros_por_alavancagem(df: pd.DataFrame, rho_min=0.2):
    """Retorna uma regra baseada em alavancagem se houver correlação com taxa"""
    df = df[["alavancagem_patrimonial", "taxa_juros_media_emprestimos_tomados"]].dropna()
    df = df[(df["alavancagem_patrimonial"] < np.inf) & (df["alavancagem_patrimonial"] > -np.inf)]

    rho, pval = spearmanr(df["alavancagem_patrimonial"],
                          df["taxa_juros_media_emprestimos_tomados"])

    if rho >= rho_min:
        # Ajuste de reta: y = a*x + b
        a, b, *_ = linregress(df["alavancagem_patrimonial"],
                              df["taxa_juros_media_emprestimos_tomados"])

        def taxa_min(df_inner):
            return a * df_inner["alavancagem_patrimonial"] + b - 0.01  # margem de tolerância

        logging.info(f"[Regra dinâmica] Correlação ρ = {rho:.2f} → ativando regra juros ≥ f(alavancagem)")
        return {
            "juros_vs_alavanc": (
                "taxa_juros_media_emprestimos_tomados",
                op.ge,
                taxa_min
            )
        }
    else:
        logging.info(f"[Regra dinâmica] Correlação ρ = {rho:.2f} → nenhuma regra adicionada")
        return {}


def search_dtypes(df, limite_categorico=50, force_categorical=None, verbose=True, remove_ids=False):
    """
    Identifica colunas numéricas e categóricas em um DataFrame.

    - Força 'client_id' e colunas em `force_categorical` como categóricas, se existirem.
    - Colunas object/string com poucos valores únicos viram 'category'.
    - Colunas numéricas permanecem como estão.

    Parâmetros:
    - df: DataFrame de entrada
    - limite_categorico: máximo de valores únicos para considerar como 'category'
    - force_categorical: lista de colunas que devem ser tratadas como categóricas
    - verbose: se True, imprime detalhes das decisões

    Retorna:
    - num_cols: lista de colunas numéricas
    - cat_cols: lista de colunas categóricas
    """
    num_cols = []
    cat_cols = []

    if force_categorical is None:
        force_categorical = []

    for col in df.columns:
        tipo = df[col].dtype

        # Força colunas explicitamente marcadas como categóricas
        if col == 'client_id' or col in force_categorical:
            cat_cols.append(col)
            if verbose:
                print(f"\nForçando '{col}' como categórica.")
            continue

        if tipo == 'object' or tipo.name == 'string':
            unicos = df[col].nunique(dropna=True)
            if unicos <= limite_categorico:
                cat_cols.append(col)
                if verbose:
                    print(f"Coluna '{col}' classificada como 'category' ({unicos} únicos).")
            else:
                if verbose:
                    print(f"Coluna '{col}' tem muitos valores únicos ({unicos}), ignorada.")
        elif pd.api.types.is_numeric_dtype(tipo):
            num_cols.append(col)
        elif pd.api.types.is_bool_dtype(tipo):
            cat_cols.append(col)
        else:
            if verbose:
                print(f"Coluna '{col}' ignorada (tipo: {tipo}).")

    if remove_ids:
        cat_cols.remove('client_id')

    if verbose:
        print(f'\nVariáveis categóricas ({len(cat_cols)}):')
        for col in cat_cols:
            print('> ', col)

        print(f'\nVariáveis numéricas ({len(num_cols)}):')
        for col in num_cols:
            print('> ', col)
    

    return num_cols, cat_cols
