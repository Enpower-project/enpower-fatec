import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from dowhy import CausalModel
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from econml.dml import LinearDML
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)


class ElasticityModel:
    def __init__(self, data, model_y_type='RandomForest', model_t_type='RandomForest',
                 cv=5, random_state=42, mc_iters=3, dir_name=''):
        self.data = data
        self.dir_name = dir_name
        self.n_periods = data.n_periods
        self.list_hours = data.list_hours
        self.cv = cv
        self.random_state = random_state
        self.mc_iters = mc_iters

        # Store model type parameters
        self.model_y_type = model_y_type
        self.model_t_type = model_t_type

        # Initialize model
        self.model = None
        self.model_y = None
        self.model_t = None
        self.elasticity = None
        self.elasticity_transposed = None

    def __repr__(self):
        return f"ElasticityModel(model_y_type={self.model_y_type}, model_t_type={self.model_t_type}, cv={self.cv}, " \
               f"random_state={self.random_state}, mc_iters={self.mc_iters})"

    def process(self, plot=0):
        self.create_model()
        self.evaluate_model()
        if plot:
            self.plot_elasticities()
        logger.success(f"Model trained and evaluated successfully.")

    def set_model_y(self):
        if self.model_y_type == 'RandomForest':
            self.model_y = RandomForestRegressor()
        elif self.model_y_type == 'LinearRegression':
            self.model_y = LinearRegression()
        elif self.model_y_type == 'XGBoost':
            self.model_y = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_y_type: {self.model_y_type}")

    def set_model_t(self):
        if self.model_t_type == 'RandomForest':
            self.model_t = RandomForestRegressor()
        elif self.model_t_type == 'LinearRegression':
            self.model_t = LinearRegression()
        elif self.model_t_type == 'XGBoost':
            self.model_t = XGBRegressor()
        else:
            raise ValueError(f"Unsupported model_t_type: {self.model_t_type}")

    def create_model(self):
        logger.debug(
            f"Training DML model with cv={self.cv}, random_state={self.random_state}, mc_iters={self.mc_iters}...")

        # Set model_y using the new method
        self.set_model_y()
        self.set_model_t()

        logger.debug('...Instantiate DoWhy model...')
        model = CausalModel(
            data=self.data.data_scaled,
            treatment=self.data.target_t,
            outcome=self.data.target_y,
            common_causes=self.data.common_causes,
            instruments=self.data.instruments,
            effect_modifiers=self.data.modifiers,
        )
        logger.debug('...Identifying estimand...')
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        logger.debug(identified_estimand)

        logger.debug('...Fitting model...')
        method_name = "backdoor.econml.dml.CausalForestDML"
        method_params = {
            'init_params': {'model_y': self.model_y,
                            'model_t': self.model_t,
                            'cv': KFold(n_splits=self.cv, shuffle=True),
                            'random_state': self.random_state,
                            'mc_iters': self.mc_iters,
                            'n_estimators': 300,
                            'max_depth': 5
                            },
            'fit_params': {'cache_values': True},
        }
        estimate = model.estimate_effect(identified_estimand,
                                         method_name=method_name,
                                         method_params=method_params,
                                         confidence_intervals=True,
                                         )
        self.model = estimate._estimator_object

    def evaluate_model(self):
        try:
            logger.debug(f"Model scores:")
            logger.debug(f"Model Y (R2): {np.round(self.model.nuisance_scores_y, 3)}")
            logger.debug(f"Model T (R2): {np.round(self.model.nuisance_scores_t, 3)}")
            logger.debug(f'Model summary: {self.model.summary()}')
        except Exception as e:
            logger.debug(f'Scores for nuisance models are unavailable: {e}')
        logger.debug(f"Final model (MSE): {round(self.model.score_, 3)}")

        hours_ = self.data.data_scaled['hour'].unique()
        self.elasticity = self.model.const_marginal_effect(np.array(hours_).reshape(-1, 1))

    def plot_elasticities(self):
        logger.debug(f"Plotting elasticity...")

        try:
            self.elasticity_transposed = np.transpose(self.elasticity[0]).reshape(1, self.n_periods, self.n_periods)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            vmin = np.min(self.elasticity_transposed)
            vmax = np.max(self.elasticity_transposed)
            sns.heatmap(self.elasticity_transposed[0], annot=True, fmt=".2f", center=0, cmap='coolwarm', cbar=True,
                        ax=ax, annot_kws={"size": 6}, vmin=vmin, vmax=vmax)
            ax.set(xlabel='Hour', ylabel='Hour', title='Elasticity')
            ax.set_xticklabels(self.list_hours)
            ax.set_yticklabels(self.list_hours)
            plt.tight_layout()
            plt.savefig(f'{self.dir_name}/elasticity_{self.data.consumers}.png')
            plt.close(fig)
        except ValueError:
            hours_ = self.data.data_scaled['hour'].unique()
            marg_effect = self.model.const_marginal_effect(np.array(hours_).reshape(-1, 1))
            marg_data = pd.DataFrame({'Self-elasticity': marg_effect.ravel()})
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(marg_data.T, cmap='copper', annot=True,
                        fmt=".3f", annot_kws={'rotation': 90},
                        cbar_kws=dict(location="top", shrink=0.5))
            cbar = ax.collections[0].colorbar
            cbar.set_label('Elasticity', labelpad=20)
            plt.tight_layout()
            plt.savefig(f'{self.dir_name}/elasticity_{self.data.consumers}.png')
            plt.close(fig)

    @staticmethod
    def ranking(consumers, elasticities, plot=1, dir_name=''):
        logger.debug(f"Ranking consumers by self-elasticities...")
        df_self = pd.DataFrame()
        magnitudes = []
        for i, consumer in enumerate(consumers):
            if elasticities[i].shape == (24, 1):
                diag = elasticities[i].reshape(1, -1)[0]
            else:
                diag = np.diag(elasticities[i])
            df_self[consumer] = diag
            magnitudes.append(np.sqrt(np.sum(np.square(diag))))
        df_self.loc['Magnitude'] = magnitudes
        df_sorted = df_self.T.sort_values('Magnitude', ascending=False)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            sns.heatmap(df_sorted, annot=True, fmt=".2f", center=0, cmap='coolwarm', cbar=True, ax=ax,
                        annot_kws={"size": 5})
            ax.set_ylabel('Consumer ranking (desc. order)', fontsize=8)
            ax.set_xlabel('Hour', fontsize=5)
            ax.set_yticks(np.arange(df_sorted.shape[0]) + 0.5, minor=False)
            ax.set_yticklabels(df_sorted.index, fontsize=3)
            ax.tick_params(axis='x', labelsize=3)
            ax.tick_params(axis='y', labelsize=3)
            for idx in np.arange(df_sorted.size).reshape(df_sorted.shape)[:, -1]:
                ax.texts[idx].set_fontweight('bold')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/ranking.png')
            plt.close(fig)

        return df_sorted
