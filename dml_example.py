import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from econml.dml import LinearDML
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV, MultiTaskLassoCV

np.random.seed(42)

# Input data
logger.debug(f"Reading input data...")
data = pd.read_csv(f".csv")
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S+00:00')
n_periods = data.index.hour.unique().shape[0]
list_hours = data.index.hour.unique().to_list()
n_panels = int(data.index.shape[0] / n_periods)

# Define targets
target_y = 'log_consumption'
target_t = 'log_price'
target_y_col = 'oc_int_total'
target_t_col = 'tariff_PL1_w_discount'
hourly_confounders = ['temperature', 'consumption_lag', 'price_lag', 'precipitation']
daily_confounders = ['weekday', 'holiday', 'month', 'day', 'year', 'weekend']

# Data preprocessing
logger.debug(f"Processing input data...")
days_lag = 1
data['consumption_lag'] = data[target_y_col].shift(n_periods * days_lag)
data['price_lag'] = data[target_t_col].shift(n_periods * days_lag)
data = data[n_periods * days_lag:]
n_panels = n_panels - days_lag
data_daily = data.resample('D').mean()

# Get outcome Y
data[target_y] = data[target_y_col].apply(lambda x: (np.log(x + 1)))
data['consumption_lag'] = data['consumption_lag'].apply(lambda x: (np.log(x + 1)))
Y = data[[target_y]].values
Y = Y.reshape((n_panels, n_periods))

# Get treatment T
data[target_t] = data[target_t_col].apply(lambda x: np.log(x))
data['price_lag'] = data['price_lag'].apply(lambda x: (np.log(x)))
T = data[[target_t]].values
T = T.reshape((n_panels, n_periods))

# Get confounders W
logger.debug(f"Processing covariates...")
W = data[hourly_confounders].values
W = W.reshape((n_panels, n_periods * W.shape[1]))
W_aux = pd.DataFrame(W)
for confounder in daily_confounders:
    W_aux[confounder] = data_daily[confounder].values
W = W_aux.values
W = StandardScaler().fit_transform(W)

# DML
logger.debug(f"Training DML model...")
model_t, model_y = RandomForestRegressor(), RandomForestRegressor()
model = LinearDML(model_y=model_y, model_t=model_t,
                  cv=3, random_state=42, mc_iters=3)
model.fit(Y, T, W=W, X=None, cache_values=True, inference='auto')

# Scores
# logger.debug(model.summary())
try:
    logger.debug(f"Model scores:")
    logger.debug(f"Model Y (R2): {np.round(model.nuisance_scores_y, 3)[0]}")
    logger.debug(f"Model T (R2): {np.round(model.nuisance_scores_t, 3)[0]}")
except:
    logger.debug('Scores for nuisance models are unavailable')
logger.debug(f"Final model (MSE): {round(model.score_, 3)}")

# Elasticity matrix
logger.debug(f"Calculating Price Elasticity Matrix...")
elasticity = model.const_marginal_effect(X=None)
elasticity_transposed = np.transpose(elasticity[0]).reshape(1, n_periods, n_periods)

fig, ax = plt.subplots(1, 1)
vmin = min(np.min(elasticity_transposed[0]), np.min(elasticity_transposed))
vmax = max(np.max(elasticity_transposed[0]), np.max(elasticity_transposed))
sns.heatmap(elasticity_transposed[0], annot=True, fmt=".2f",
            center=0, cmap='vlag', cbar=True, ax=ax,
            annot_kws={"size": 6}, vmin=vmin, vmax=vmax)
ax.set(xlabel='Hour', ylabel='Hour', title='Elasticity')
ax.set_xticklabels(list_hours)
ax.set_yticklabels(list_hours)
plt.show()
