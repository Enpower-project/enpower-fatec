import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from econml.dml import LinearDML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)


class ElasticityModel:
    def __init__(self, data, model_y_type='RandomForest', model_t_type='RandomForest',
                 cv=5, random_state=42, mc_iters=5, dir_name=''):
        self.data = data
        self.dir_name = dir_name
        self.Y = self.data.Y
        self.T = self.data.T
        self.W = self.data.W
        self.X = self.data.X
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
            plt.show()
        logger.success(f"Model trained and evaluated successfully.")

    def set_model_y(self):
        if self.model_y_type == 'RandomForest':
            self.model_y = RandomForestRegressor()
        elif self.model_y_type == 'LinearRegression':
            self.model_y = LinearRegression()
        elif self.model_y_type == 'XGBoost':
            self.model_y = XGBRegressor(max_depth=6, n_estimators=300)
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

        # Initialize LinearDML model
        self.model = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
            cv=self.cv,
            random_state=self.random_state,
            mc_iters=self.mc_iters
        )
        self.model.fit(self.Y, self.T, W=self.W, X=self.X, cache_values=True, inference='auto')

    def evaluate_model(self):
        try:
            logger.debug(f"Model scores:")
            logger.debug(f"Model Y (R2): {np.round(self.model.nuisance_scores_y, 3)}")
            logger.debug(f"Model T (R2): {np.round(self.model.nuisance_scores_t, 3)}")
            logger.debug(f'Model summary: {self.model.summary()}')
        except Exception as e:
            logger.debug(f'Scores for nuisance models are unavailable: {e}')
        logger.debug(f"Final model (MSE): {round(self.model.score_, 3)}")
        self.elasticity = self.model.const_marginal_effect(X=self.X)

    def plot_elasticities(self):
        logger.debug(f"Calculating Price Elasticity Matrix...")
        self.elasticity_transposed = np.transpose(self.elasticity[0]).reshape(1, self.n_periods, self.n_periods)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        vmin = np.min(self.elasticity_transposed)
        vmax = np.max(self.elasticity_transposed)
        sns.heatmap(self.elasticity_transposed[0], annot=True, fmt=".2f", center=0, cmap='vlag', cbar=True, ax=ax,
                    annot_kws={"size": 6}, vmin=vmin, vmax=vmax)
        ax.set(xlabel='Hour', ylabel='Hour', title='Elasticity')
        ax.set_xticklabels(self.list_hours)
        ax.set_yticklabels(self.list_hours)
        plt.tight_layout()
        plt.savefig(f'{self.dir_name}/elasticity_{self.data.consumers}.png')
        plt.close(fig)

    @staticmethod
    def ranking(consumers, elasticities, plot=1, dir_name=''):
        logger.debug(f"Ranking consumers by self-elasticities...")
        df_self = pd.DataFrame()
        magnitudes = []
        for i, consumer in enumerate(consumers):
            diag = np.diag(elasticities[i])
            df_self[consumer] = diag
            magnitudes.append(np.sqrt(np.sum(np.square(diag))))
        df_self.loc['Magnitude'] = magnitudes
        df_sorted = df_self.T.sort_values('Magnitude', ascending=False)
        logger.debug(f'\n{df_self}')

        if plot:
            fig, ax = plt.subplots(1, 1)
            sns.heatmap(df_sorted.T, annot=True, fmt=".2f", center=0, cmap='Greens_r', cbar=True, ax=ax,
                        annot_kws={"size": 6})
            ax.set(xlabel='Consumer ranking (desc. order)', ylabel='Hour')
            for idx in np.arange(df_sorted.T.size).reshape(df_sorted.T.shape)[-1, :]:
                ax.texts[idx].set_fontweight('bold')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/ranking.png')
            plt.close(fig)

        return df_sorted['Magnitude']


