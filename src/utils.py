from __future__ import annotations

# === Built-in ===
import logging
import pathlib

# === Typing ===
from typing import List, Optional

# === Third-party: General purpose ===
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis, pearsonr

# === Third-party: Visualization ===
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# === Third-party: Machine Learning ===
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
)
from sklearn.inspection import PartialDependenceDisplay

# === Third-party: ML Libraries ===
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
from boruta import BorutaPy

# === Model Explainability ===
import shap


logging.getLogger('matplotlib.category').setLevel(logging.WARNING)


class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.woe_dicts = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y.values if isinstance(y, pd.Series) else y

        for col in df.drop(columns=[self.target_col]):
            grouped = df.groupby(col, observed=False)[self.target_col]
            dist_good = (1 - grouped.mean()).clip(1e-7, 1)
            dist_bad = grouped.mean().clip(1e-7, 1)
            woe = np.log(dist_good / dist_bad)
            self.woe_dicts[col] = woe.to_dict()
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)

    def transform(self, X):
        X_woe = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapper = self.woe_dicts.get(col, {})
            X_woe[col] = X[col].map(mapper).fillna(0.0)
        return X_woe.values  # retorna como numpy para compatibilidade com XGBoost


class CreditRiskEDA:
    """Explorador EDA especializado em risco de crédito.

    Parameters
    ----------
    df : pd.DataFrame
        Conjunto de dados completo.
    target_col : str
        Coluna‑alvo binária (0 = saudável / 1 = inadimplente).
    numerical_cols : list[str]
        Variáveis numéricas.
    categorical_cols : list[str]
        Variáveis categóricas.
    bins : int, default = 10
        Nº de *bins* para cálculo de IV em variáveis contínuas.
    order_by_iv : bool, default = False
        Se True, plota em ordem decrescente de IV.
    remove_outliers : bool, default = False
        Remove outliers |z|>4.
    independent_outlier_removal : bool, default = False
        Se True, remoção de outliers é LOCAL a cada plot numérico.
    numerical_plot / categorical_plot : habilitam/desabilitam blocos específicos.
    """

    COLOR_GOOD = "#77BDD9"
    COLOR_BAD  = "#0A3873"
    PALETTE    = [COLOR_GOOD, COLOR_BAD]
    COLOR_MAP  = {0: COLOR_GOOD, 1: COLOR_BAD, "0": COLOR_GOOD, "1": COLOR_BAD}

    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 numerical_cols: list[str],
                 categorical_cols: list[str],
                 bins: int = 10,
                 order_by_iv: bool = False,
                 remove_outliers: bool = False,
                 independent_outlier_removal: bool = False,
                 numerical_plot: bool = True,
                 categorical_plot: bool = True,
                 plot_shap: bool = False):          # <-- NEW
        # logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        self.df_orig  = df.copy()
        self.target   = target_col
        self.num_cols = list(numerical_cols)
        self.cat_cols = list(categorical_cols)
        self.bins     = bins
        self.order_by_iv = order_by_iv
        self.remove_outliers = remove_outliers
        self.independent_outlier_removal = independent_outlier_removal
        self.numerical_plot   = numerical_plot
        self.categorical_plot = categorical_plot
        self.plot_shap = plot_shap               # <-- NEW

        sns.set_style("white")
        plt.rcParams["axes.grid"] = False

        # valida colunas
        self._filter_available_cols()

        # base de trabalho
        if remove_outliers and not independent_outlier_removal:
            self.df = self._handle_outliers()
        else:
            self.df = self.df_orig

        # IV pré‑cálculo se necessário
        self.iv_series = self._calc_iv_many() if order_by_iv else None

    # ------------------------------------------------------------
    def _filter_available_cols(self):
        miss_num = [c for c in self.num_cols if c not in self.df_orig.columns]
        miss_cat = [c for c in self.cat_cols if c not in self.df_orig.columns]
        if miss_num or miss_cat:
            self.logger.info("Features removidas: %s", miss_num + miss_cat)
        self.num_cols = [c for c in self.num_cols if c in self.df_orig.columns]
        self.cat_cols = [c for c in self.cat_cols if c in self.df_orig.columns]


    # ------------------------------------------------------------------#
    # 1. INFORMATION VALUE                                              #
    # ------------------------------------------------------------------#
    @staticmethod
    def _calc_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10):
        tmp = df[[feature, target]].copy()
        if tmp[feature].dtype.kind in "bifc":
            noise = np.random.uniform(-1e-5, 1e-5, len(tmp))
            tmp[feature] += noise
            try:
                tmp["bin"] = pd.qcut(
                    tmp[feature].rank(method="first"), q=bins, duplicates="drop"
                )
            except ValueError:
                q_alt = max(2, bins // 2)
                try:
                    tmp["bin"] = pd.qcut(
                        tmp[feature].rank(method="first"), q=q_alt, duplicates="drop"
                    )
                except Exception:
                    tmp["bin"] = tmp[feature].astype(str)
        else:
            tmp["bin"] = tmp[feature].astype(str)

        grouped = tmp.groupby("bin", observed=False)
        dist_good = grouped[target].apply(lambda x: (1 - x).sum()) / max(
            1, (1 - tmp[target]).sum()
        )
        dist_bad = grouped[target].sum() / max(1, tmp[target].sum())
        eps = 1e-7
        woe = np.log((dist_good + eps) / (dist_bad + eps))
        iv = ((dist_good - dist_bad) * woe).sum()
        return iv

    def _calc_iv_many(self):
        col_set = []
        if self.numerical_plot:
            col_set.extend(self.num_cols)
        if self.categorical_plot:
            col_set.extend(self.cat_cols)
        iv_dict = {
            col: self._calc_iv(self.df, col, self.target, self.bins) for col in col_set
        }
        return pd.Series(iv_dict).sort_values(ascending=False)

    def plot_iv_heatmap(self, features_per_row: int = 15):
        """
        Plota o Information Value (IV) das features em um heatmap organizado em grade.

        Args:
            features_per_row (int): Número máximo de features a serem exibidas por linha.
        """
        if self.iv_series is None:
            print("Série de IV não calculada. Execute o cálculo primeiro.")
            return

        # 1. Ordena as features pelo IV, da maior para a menor
        iv_series_sorted = self.iv_series.sort_values(ascending=False)

        # 2. Prepara os dados para a grade (grid)
        n_features = len(iv_series_sorted)
        # Calcula o número de linhas necessárias no grid
        n_rows = int(np.ceil(n_features / features_per_row))

        # Pega os valores e nomes das features
        values = iv_series_sorted.values.tolist()
        feature_names = iv_series_sorted.index.tolist()

        # Preenche com valores vazios para que o grid fique completo
        padding_needed = (n_rows * features_per_row) - n_features
        values.extend([np.nan] * padding_needed)
        feature_names.extend([''] * padding_needed)

        # 3. Reorganiza os dados em uma matriz 2D
        heatmap_data = np.array(values).reshape(n_rows, features_per_row)
        
        # Cria as anotações (rótulos) para cada célula
        labels = [f'{name}\n{val:.3f}' if name else '' for name, val in zip(feature_names, values)]
        annot_labels = np.array(labels).reshape(n_rows, features_per_row)

        # 4. Plota o heatmap
        # Ajusta o tamanho da figura dinamicamente
        fig_height = n_rows * 1.5
        fig_width = features_per_row * 1.2
        plt.figure(figsize=(fig_width, fig_height))

        # Define uma paleta de cores (ex: Verde para bom, Amarelo para médio, Vermelho para alto)
        cmap = mpl.colors.ListedColormap(['#2ECC71', '#F1C40F', '#E74C3C', '#9B59B6', '#34495E'])
        bounds = [0, 0.1, 0.3, 0.5, 1] # Limites de IV: <0.1, 0.1-0.3, 0.3-0.5, >0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        sns.heatmap(
            heatmap_data,
            annot=annot_labels,
            fmt="",  # O formato já está na string de anotação
            cmap=cmap,
            norm=norm, # Aplica a norma de cores customizada
            linewidths=.5,
            cbar=True,
            xticklabels=False,  # Esconde rótulos dos eixos, pois a info está na célula
            yticklabels=False
        )

        plt.title("Information Value (IV) por Feature", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------#
    # 2. GRÁFICOS                                                       #
    # ------------------------------------------------------------------#
    def _plot_numeric(self, col: str, bins: int = 40):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        series = self.df[col].dropna()
        kde_flag = (len(series) > 10) and (series.nunique() > 1)

        sns.histplot(
            data=self.df,
            x=col,
            hue=self.target,
            kde=kde_flag,
            palette=self.PALETTE,
            element="step",
            bins=bins,
            binrange=(series.min(), series.max()),  # garante que não corte
            ax=axes[0],
        )
        axes[0].set_title(f"Distribuição de {col}")
        axes[0].set_ylabel("Contagem")
        # amplia ligeiramente os limites para evitar corte visual
        xmin, xmax = series.min(), series.max()
        margin = 0.02 * (xmax - xmin)
        axes[0].set_xlim(xmin - margin, xmax + margin)

        sns.boxplot(
            x=self.target,
            y=col,
            data=self.df,
            palette=self.PALETTE,
            ax=axes[1],
        )
        axes[1].set_title(f"{col} vs {self.target}")
        axes[1].set_xlabel("")
        axes[0].legend(
            title=self.target,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )
        plt.tight_layout()
        plt.show()

    def _plot_categorical(self, col: str, rotation_threshold: int = 10):
        ordered = (
            self.df.groupby(col)[self.target].mean().sort_values().index
        )
        cross = (
            pd.crosstab(self.df[col], self.df[self.target], normalize="index")
            .loc[ordered]
        )
        colors = [self.COLOR_MAP[v] for v in cross.columns]
        ax = cross.plot(
            kind="bar",
            stacked=True,
            figsize=(max(6, len(ordered) * 0.6), 4),
            width=0.8,
            color=colors,
            edgecolor="none",
        )
        ax.set_title(f"Event Rate por {col}")
        ax.set_ylabel("Proporção")
        for i, (_, row) in enumerate(cross.iterrows()):
            cum = 0
            for j, pct in enumerate(row):
                cum += pct
                label_col = "navy" if cross.columns[j] in [0, "0"] else "white"
                ax.text(i, cum - pct / 2, f"{pct:.1%}", ha="center", va="center", fontsize=8, color=label_col)
        rot = 90 if len(ordered) > rotation_threshold else 0
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha="center", va="top")
        plt.tight_layout()
        plt.show()

    def _plot_shap_summary(self):
        import shap
        import matplotlib.pyplot as plt
        from xgboost import XGBClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.utils.class_weight import compute_sample_weight

        # --- 1. Separação dos dados
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # --- 2. Cálculo de sample weights
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)

        # --- 3. Pré-processador com WoE
        pre = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.num_cols),  # ou RobustScaler()
                ('cat', WoEEncoder(target_col=self.target), self.cat_cols)
            ]
        )

        # --- 4. Modelo
        xgb = XGBClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )

        # --- 5. Pipeline
        pipe = Pipeline([('prep', pre), ('clf', xgb)])

        # --- 6. Ajuste com pesos
        pipe.fit(X, y, clf__sample_weight=sample_weights)

        # --- 7. Explicações com SHAP
        X_trans = pipe['prep'].transform(X)
        explainer = shap.Explainer(pipe['clf'], X_trans)
        shap_values = explainer(X_trans, check_additivity=False)

        self.logger.info("[SHAP] Features categóricas foram convertidas com WoE antes do XGBoost.")

        # --- 8. Gráfico SHAP
        shap.summary_plot(
            shap_values,
            features=X_trans,
            feature_names=pipe['prep'].get_feature_names_out(),
            show=True,
            plot_size=(10, 6),
            color_bar=True
        )


    # ------------------------------------------------------------------#
    # 3. EXECUÇÃO                                                       #
    # ------------------------------------------------------------------#
    def run(self):
        # 1) IV heatmap (opcional)
        if self.order_by_iv:
            self.plot_iv_heatmap()
            ordered_features = list(self.iv_series.index)
        else:
            ordered_features = []  # não utilizado

        # 1.1) SHAP summary plot (opcional)
        if self.plot_shap:
            self._plot_shap_summary()

        # 2) Plotagem seguindo flags
        if self.order_by_iv:
            for feat in ordered_features:
                if feat in self.num_cols and self.numerical_plot:
                    self._plot_numeric(feat)
                elif feat in self.cat_cols and self.categorical_plot:
                    self._plot_categorical(feat)
        else:  # default sem ordenação IV
            if self.numerical_plot:
                for col in self.num_cols:
                    self._plot_numeric(col)
            if self.categorical_plot:
                for col in self.cat_cols:
                    self._plot_categorical(col)

    # ------------------------------------------------------------------#
    # 4. OUTLIERS                                                       #
    # ------------------------------------------------------------------#
    def _handle_outliers(self) -> pd.DataFrame:
        if not self.num_cols:
            return self.df_orig.copy()

        df = self.df_orig.copy()
        z = np.abs(stats.zscore(df[self.num_cols], nan_policy="omit"))
        mask = (z > 4).any(axis=1)
        removed = mask.sum()
        total = len(df)

        if removed:
            pct = 100 * removed / total
            self.logger.info("Outliers removidos (|z| > 4): %d registros (%.2f%% do total)", removed, pct)

        return df.loc[~mask].reset_index(drop=True)


class ScalerSelector(BaseEstimator, TransformerMixin):
    """
    Seleciona e aplica dinamicamente o scaler adequado para cada feature numérica.

    Parâmetros
    ----------
    strategy : {'auto', 'standard', 'robust', 'minmax', 'quantile', None}, default='auto'
        - 'auto'     → decide por coluna com base em normalidade, skew e outliers.
        - demais     → aplica o scaler escolhido a **todas** as colunas.
        - None       → passthrough (sem escalonamento).

    serialize : bool, default=False
        Se True, salva automaticamente o dict de scalers em `save_path` após o fit.

    save_path : str | Path | None
        Caminho do arquivo .pkl a ser salvo (ou sobreposto). Só usado se
        `serialize=True`. Padrão: 'scalers.pkl'.

    random_state : int, default=0
        Usado no QuantileTransformer e em amostragens internas.

    logger : logging.Logger | None
        Logger customizado; se None, cria logger básico.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self,
                 strategy: str = 'auto',
                 serialize: bool = False,
                 save_path: str | pathlib.Path | None = None,
                 random_state: int = 0,
                 logger: logging.Logger | None = None):
        self.strategy = strategy.lower() if strategy else None
        self.serialize = serialize
        self.save_path = pathlib.Path(save_path or "scalers.pkl")
        self.random_state = random_state

        self.scalers_: dict[str, BaseEstimator] = {}
        self.report_:  dict[str, dict] = {}      # estatísticas por coluna

        # logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO,
                                    format="%(levelname)s: %(message)s")

    # ------------------------------------------------------------------
    # MÉTODO INTERNO PARA ESTRATÉGIA AUTO
    # ------------------------------------------------------------------
    def _choose_auto(self, x: pd.Series):
        """
        Decide qual scaler empregar (ou nenhum) para a série x.

        Retorna
        -------
        scaler | None, dict
            Instância já criada (ainda não fitada) e dicionário de métricas.
        """
        sample = x.dropna().astype(float)

        # Coluna constante
        if sample.nunique() == 1:
            return None, dict(reason='constante', scaler='None')

        # ---------------- métricas básicas ----------------
        try:
            p_val = shapiro(sample.sample(min(5000, len(sample)),
                                          random_state=self.random_state))[1]
        except Exception:   # amostra minúscula ou erro numérico
            p_val = 0.0

        sk = skew(sample, nan_policy="omit")
        kt = kurtosis(sample, nan_policy="omit")        # Fisher (0 = normal)

        # ---------------- critérios de NÃO escalonar ----------------
        # (1) variável já em [0,1]
        if 0.95 <= sample.min() <= sample.max() <= 1.05:
            return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
                              reason='já escalada [0-1]', scaler='None')

        # # (2) praticamente normal
        # if abs(sk) < 0.05 and abs(kt) < 0.1 and p_val > 0.90:
        #     return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
        #                       reason='praticamente normal', scaler='None')
        
        # (3) praticamente normal (menos restritivo)
        if abs(sk) < 0.5 and abs(kt) < 1.0 and p_val > 0.05:
            return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
                            reason='aproximadamente normal', scaler='None')


        # ---------------- escolha de scaler ----------------
        if p_val >= 0.05 and abs(sk) <= 0.5:
            scaler = StandardScaler()
            reason = '≈normal'
        elif abs(sk) > 3 or kt > 20:
            scaler = QuantileTransformer(output_distribution='normal',
                                          random_state=self.random_state)
            reason = 'assimetria/kurtosis extrema'
        elif abs(sk) > 0.5:
            scaler = RobustScaler()
            reason = 'skew moderado/outliers'
        else:
            scaler = MinMaxScaler()
            reason = 'default'

        stats = dict(p_value=p_val, skew=sk, kurtosis=kt,
                     reason=reason, scaler=scaler.__class__.__name__)
        return scaler, stats

    # ------------------------------------------------------------------
    # API FIT
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        if self.strategy not in {'auto', 'standard', 'robust',
                                 'minmax', 'quantile', None}:
            raise ValueError(f"strategy '{self.strategy}' não suportada.")

        for col in X_df.columns:
            # --- seleção do scaler -----------------------------------
            if self.strategy == 'auto':
                scaler, stats = self._choose_auto(X_df[col])
            elif self.strategy == 'standard':
                scaler = StandardScaler()
                stats  = dict(reason='global-standard', scaler='StandardScaler')
            elif self.strategy == 'robust':
                scaler = RobustScaler()
                stats  = dict(reason='global-robust', scaler='RobustScaler')
            elif self.strategy == 'minmax':
                scaler = MinMaxScaler()
                stats  = dict(reason='global-minmax', scaler='MinMaxScaler')
            elif self.strategy == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal',
                                             random_state=self.random_state)
                stats  = dict(reason='global-quantile', scaler='QuantileTransformer')
            else:              # None
                scaler = None
                stats  = dict(reason='passthrough', scaler='None')

            # --- ajuste ---------------------------------------------
            if scaler is not None:
                scaler.fit(X_df[[col]])

            self.scalers_[col] = scaler
            self.report_[col]  = stats

            # --- log -------------------------------------------------
            self.logger.info(
                "Coluna '%s' → %s (p=%.3f, skew=%.2f, kurt=%.1f) | motivo: %s",
                col, stats.get('scaler'),
                stats.get('p_value', np.nan),
                stats.get('skew',     np.nan),
                stats.get('kurtosis', np.nan),
                stats['reason']
            )

        # serialização opcional
        if self.serialize:
            self.save(self.save_path)

        return self

    # ------------------------------------------------------------------
    # TRANSFORM / INVERSE_TRANSFORM
    # ------------------------------------------------------------------
    def transform(self, X, return_df: bool = False):
        X_df = pd.DataFrame(X).copy()

        # Verifica se todas as colunas esperadas estão presentes
        missing = set(self.scalers_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no transform: {missing}")

        for col, scaler in self.scalers_.items():
            if scaler is not None:
                X_df[col] = scaler.transform(X_df[[col]])

        return X_df if return_df else X_df.values

    def inverse_transform(self, X, return_df: bool = False):
        X_df = pd.DataFrame(X, columns=self.scalers_.keys()).copy()
        for col, scaler in self.scalers_.items():
            if scaler is not None:
                X_df[col] = scaler.inverse_transform(X_df[[col]])
        return X_df if return_df else X_df.values

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)

    def report_as_df(self) -> pd.DataFrame:
        """Devolve o relatório de métricas/decisões como DataFrame."""
        return pd.DataFrame.from_dict(self.report_, orient='index')

    # ------------------------------------------------------------------
    # SERIALIZAÇÃO
    # ------------------------------------------------------------------
    def save(self, path: str | pathlib.Path | None = None):
        """Serializa scalers + relatório + metadados."""
        path = pathlib.Path(path or self.save_path)
        joblib.dump({
            'scalers': self.scalers_,
            'report':  self.report_,
            'strategy': self.strategy,
            'random_state': self.random_state
        }, path)
        self.logger.info("Scalers salvos em %s", path)

    def load(self, path: str | pathlib.Path):
        """Restaura scalers + relatório + metadados já treinados."""
        data = joblib.load(path)
        self.scalers_  = data['scalers']
        self.report_   = data.get('report', {})
        self.strategy  = data.get('strategy', self.strategy)
        self.random_state = data.get('random_state', self.random_state)
        self.logger.info("Scalers carregados de %s", path)
        return self

from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
import xgboost as xgb
import lightgbm as lgb

# --- Configuração do Logging ---
framework_logger = logging.getLogger("CreditFeatureSelector")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def _calc_iv_table(df_woe: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    ivs = []
    if df_woe.empty:
        framework_logger.warning("_calc_iv_table: df_woe de entrada está vazio.")
        return pd.DataFrame(columns=["feature", "IV"])
        
    common_index = df_woe.index.intersection(y.index)
    if common_index.empty:
        framework_logger.warning("_calc_iv_table: Nenhum índice em comum entre df_woe e y.")
        return pd.DataFrame(columns=["feature", "IV"])
        
    df_woe_aligned = df_woe.loc[common_index]
    y_aligned = y.loc[common_index]

    if y_aligned.isnull().any():
        not_nan_y_mask = y_aligned.notna()
        y_aligned = y_aligned[not_nan_y_mask]
        df_woe_aligned = df_woe_aligned.loc[not_nan_y_mask]

    if df_woe_aligned.empty or y_aligned.empty:
        framework_logger.warning("_calc_iv_table: df_woe_aligned ou y_aligned ficou vazio.")
        return pd.DataFrame(columns=["feature", "IV"])

    for col in df_woe_aligned.columns:
        wo = df_woe_aligned[col]
        valid_mask = wo.notna() & y_aligned.notna()
        wo_clean = wo[valid_mask]
        current_y_clean = y_aligned[valid_mask]

        if wo_clean.empty or wo_clean.nunique() < 1 or current_y_clean.nunique() < 2 :
            iv = 0.0
        else:
            mean_woe_good = wo_clean[current_y_clean == 0].mean()
            mean_woe_bad = wo_clean[current_y_clean == 1].mean()
            if pd.isna(mean_woe_good) or pd.isna(mean_woe_bad): iv = 0.0
            else: iv = (mean_woe_good - mean_woe_bad) * (wo_clean.max() - wo_clean.min())
            iv = iv if pd.notna(iv) else 0.0
        ivs.append(iv)

    if not ivs: return pd.DataFrame(columns=["feature", "IV"])
    return pd.DataFrame({"feature": df_woe_aligned.columns, "IV": ivs}).sort_values("IV", ascending=False).reset_index(drop=True)

# # %% -----------------------------------------------
# from typing import List
# import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from boruta import BorutaPy
# import xgboost as xgb
# import lightgbm as lgb

# --- utilidades já existentes -------------------------------------------------

def _vif_filter(df: pd.DataFrame, thr: float = 5.0) -> List[str]:
    """
    Remove recursivamente variáveis com VIF > thr.
    Retorna lista final de colunas.
    """
    features = df.columns.tolist()
    while True:
        vif_vals = pd.Series(
            [variance_inflation_factor(df[features].values, i) for i in range(len(features))],
            index=features
        )
        max_vif = vif_vals.max()
        if max_vif < thr:
            break
        drop_feat = vif_vals.idxmax()
        features.remove(drop_feat)
    return features


class CreditFeatureSelector:
    """
    Selecionador de variáveis (IV → VIF → Boruta) reutilizável.
    """
    def __init__(
        self,
        iv_thr: float = 0.02,
        vif_thr: float = 5.0,
        n_boruta_iter: int = 100,
        random_state: int = 42,
        use_lgb: bool = False,            # ← opcional
    ):
        self.iv_thr = iv_thr
        self.vif_thr = vif_thr
        self.n_boruta_iter = n_boruta_iter
        self.random_state = random_state
        self.use_lgb = use_lgb
        self.selected_features_: List[str] = []

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CreditFeatureSelector":
        # 1) Information Value -------------------------------------------
        iv_table = _calc_iv_table(X, y)
        vars_iv = iv_table.query("IV > @self.iv_thr")["feature"].tolist()
        if not vars_iv:
            self.selected_features_ = []
            return self
        X_iv = X[vars_iv].copy()

        # 2) Variance Inflation Factor -----------------------------------
        vars_vif = _vif_filter(X_iv, thr=self.vif_thr)
        if not vars_vif:
            self.selected_features_ = []
            return self
        X_vif = X_iv[vars_vif].copy()

        # 3) Boruta -------------------------------------------------------
        def _boruta(estimator):
            sel = BorutaPy(
                estimator,
                n_estimators="auto",
                max_iter=self.n_boruta_iter,
                random_state=self.random_state,
                verbose=0,
            )
            sel.fit(X_vif.values, y.values)
            return X_vif.columns[sel.support_].tolist()

        # XGBoost base
        xgb_est = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        )
        feats = set(_boruta(xgb_est))

        # Opcional: LightGBM
        if self.use_lgb:
            lgb_est = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                objective="binary",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                min_child_samples=20,
            )
            feats |= set(_boruta(lgb_est))

        self.selected_features_ = sorted(feats)
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_] if self.selected_features_ else pd.DataFrame(index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ===============================================================
#  FeatureRelationshipInspector
#  Avalia linearidade vs. não‑linearidade de features com y
# ===============================================================

class FeatureRelationshipInspector:
    """
    Analisa se as variáveis possuem relação linear ou não‑linear com o target binário.

    Métodos suportados
    ------------------
    1. PDP (Partial Dependence Plot)
    2. Event‑Rate por bin (barplot)
    3. SHAP dependence plot
    4. Correlação de Pearson vs. Event‑Rate shape

    Requerimentos:
    --------------
    - XGBoost (como modelo base)
    - pandas, numpy, seaborn, shap, scikit‑learn
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        num_cols: List[str],
        cat_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.target = target_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols or []
        self.random_state = random_state

        # === modelo baseline para PDP / SHAP ===
        self._build_model()

        # === verificacoes ===
        self.num_cols = [c for c in num_cols if c in df.columns]
        self.cat_cols = [c for c in (cat_cols or []) if c in df.columns]

        missing = set(num_cols + (cat_cols or [])) - set(df.columns)
        if missing:
            logging.warning("Features removidas por não existirem no DataFrame: %s", missing)


    # --------------------------------------------------------- #
    # 1. pipeline + modelo
    # --------------------------------------------------------- #
    def _build_model(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.sample_weight_ = compute_sample_weight("balanced", y=y)

        self.prep_ = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.num_cols),
                (
                    "cat",
                    "passthrough",
                    self.cat_cols,
                ),  # se quiser WoE, troque aqui
            ]
        )

        self.model_ = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

        self.pipe_ = Pipeline([("prep", self.prep_), ("clf", self.model_)])
        self.pipe_.fit(X, y, clf__sample_weight=self.sample_weight_)

        # --- ADD THESE LINES ---
        # Store the transformed data and feature names as attributes
        self.X_transformed_ = self.pipe_.named_steps["prep"].transform(X)
        self.transformed_feature_names_ = self.pipe_.named_steps["prep"].get_feature_names_out()
        # -----------------------

        # SHAP explainer pré‑inicializado
        X_trans = self.pipe_.named_steps["prep"].transform(X)
        self.shap_explainer_ = shap.Explainer(self.pipe_.named_steps["clf"], X_trans)
        self.shap_values_ = self.shap_explainer_(X_trans, check_additivity=False)

    # --------------------------------------------------------- #
    # 2. PDP
    # --------------------------------------------------------- #

    def plot_pdp(self, features: List[str], figsize=(6, 4)):
        """Plots the Partial Dependence Plot for the given features."""
        fig, ax = plt.subplots(figsize=figsize)

        PartialDependenceDisplay.from_estimator(
            # Pass the final XGBoost model, not the whole pipeline
            estimator=self.model_,
            # Pass the pre-processed data
            X=self.X_transformed_,
            # Pass the list of features to plot (original names are fine)
            features=features,
            # Provide all transformed feature names for context
            feature_names=self.transformed_feature_names_.tolist(),
            ax=ax,
            kind="average",
        )
        ax.set_title("Partial Dependence Plot")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------- #
    # 3. Event‑rate por bin
    # --------------------------------------------------------- #
    def plot_event_rate(self, feature: str, bins: int = 10):
        df = self.df[[feature, self.target]].dropna()
        df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
        resumo = df.groupby("bin")[self.target].mean().reset_index()

        plt.figure(figsize=(6, 3))
        sns.barplot(x="bin", y=self.target, data=resumo, color="#4B8BBE")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Event‑rate (y=1)")
        plt.title(f"Event‑Rate por bin – {feature}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------- #
    # 4. SHAP dependence
    # --------------------------------------------------------- #
    def plot_shap_dependence(self, feature: str):
        name_transf = self._map_name(feature)
        shap.plots.scatter(
            self.shap_values_[:, name_transf],
            color=self.shap_values_,
            show=True,
            title=f"SHAP Dependence – {feature}",
        )

    # --------------------------------------------------------- #
    # 5. Correlação x Event‑Rate shape
    # --------------------------------------------------------- #
    def correlation_summary(self, feature: str, bins: int = 10):
        sc, _ = pearsonr(
            self.df[feature].fillna(self.df[feature].median()),
            self.df[self.target],
        )
        print(f"Correlação de Pearson ({feature} vs y): {sc:.3f}")

        # quick shape print
        df = self.df[[feature, self.target]].dropna()
        df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
        er = df.groupby("bin")[self.target].mean()
        print("Event‑rate por bin:")
        display(er)

    # # --------------------------------------------------------- #
    # # util interno
    # # --------------------------------------------------------- #
    def _map_name(self, feat: str) -> str:
        """
        Devolve o nome da feature após o ColumnTransformer.
        Aceita tanto numéricas quanto categóricas.
        """
        names = self.prep_.get_feature_names_out()

        # 1) nome já existe tal qual
        if feat in names:
            return feat

        # 2) prefixos possíveis (caso você mude o transformer no futuro)
        if f"num__{feat}" in names:
            return f"num__{feat}"
        if f"cat__{feat}" in names:
            return f"cat__{feat}"

        raise ValueError(f"'{feat}' não encontrado em feature_names_out().")
    # ---------------------------------------------
