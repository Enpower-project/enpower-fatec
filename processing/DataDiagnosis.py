import pandas as pd
from loguru import logger
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Diagnosis:
    def __init__(self, data, target_y, hourly_confounders, daily_confounders):
        self.data = data
        self.target_y = target_y
        self.hourly_confounders = hourly_confounders
        self.daily_confounders = daily_confounders

    def __repr__(self):
        return (f"Diagnosis(data={self.data}, target_y={self.target_y}, hourly_confounders={self.hourly_confounders}, "
                f"daily_confounders={self.daily_confounders})")

    def execute(self):
        logger.debug("Executing data diagnosis...")

        vif_data = self.calculate_vif()
        feature_importance = self.calculate_feature_importance()
        low_loading_vars = self.perform_factor_analysis()

        return vif_data, feature_importance, low_loading_vars

    def calculate_vif(self):
        logger.debug("Calculating VIF for confounders...")

        confounders_df = self.data[self.hourly_confounders + self.daily_confounders].copy()
        confounders_df.dropna(inplace=True)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = confounders_df.columns
        vif_data["VIF"] = [variance_inflation_factor(confounders_df.values, i)
                           for i in range(confounders_df.shape[1])]
        logger.debug(f"{vif_data}")
        return vif_data

    def calculate_feature_importance(self):
        logger.debug("Calculating feature importance using RandomForest...")

        # Prepare data for RandomForest model
        X_features = self.data[self.hourly_confounders + self.daily_confounders].copy()
        y_target = self.data[self.target_y].copy()

        # Drop any rows with NaN values
        X_features.dropna(inplace=True)
        y_target = y_target.loc[X_features.index]

        # Initialize and train the RandomForest model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_features, y_target)

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_features.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        logger.debug(f"{feature_importance}")
        return feature_importance

    def perform_factor_analysis(self):
        logger.debug("Performing Factor Analysis for variable reduction...")

        confounders_df = self.data[self.hourly_confounders + self.daily_confounders].dropna()
        scaler = StandardScaler()
        confounders_scaled = scaler.fit_transform(confounders_df)
        fa = FactorAnalysis(n_components=5, random_state=42)
        fa.fit(confounders_scaled)
        factor_loadings = pd.DataFrame(fa.components_, columns=confounders_df.columns).T
        low_loading_vars = factor_loadings[factor_loadings.abs().max(axis=1) < 0.4].index.tolist()

        logger.debug(f"{low_loading_vars}")
        return low_loading_vars
