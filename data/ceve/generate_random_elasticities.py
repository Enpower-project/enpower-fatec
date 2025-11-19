import os
import re
import json
import numpy as np
from loguru import logger

from causality.elasticity import Elasticity

numbers = []
for f in os.listdir(os.getcwd()):
    m = re.match(r"P(\d+)\.xlsx$", f)
    if m:
        numbers.append(int(m.group(1)))
logger.debug(f"Found files for numbers: {numbers}")

# Generate dict with random elasticities
random_elasticities = {}
for num in numbers:
    logger.debug(f"Generating elasticity for P{num}")
    scaling_factor = np.random.choice([1, 3, 5])
    logger.debug(f"Scaling factor for elasticity: {scaling_factor}")
    self_elasticity = np.array([
        -0.072, -0.082, -0.077, -0.086, -0.126, -0.119, -0.097, -0.083, -0.085, -0.091, -0.085, -0.071,
        -0.066, -0.079, -0.089, -0.074, -0.056, -0.058, -0.061, -0.072, -0.078, -0.079, -0.073, -0.071])
    self_elasticity *= scaling_factor
    self_elasticity += np.random.normal(0, 0.005, size=self_elasticity.shape)
    pem_types = Elasticity(self_elasticity)
    pem_examples = pem_types.generate_pem()
    pem_type = 'SelfElastic'
    logger.debug(f"Selected PEM type: {pem_type}")
    pem = pem_examples[pem_type]
    random_elasticities[f'{num}'] = pem.tolist()

# Save to JSON
with open('elasticity_config.json', 'w') as f:
    json.dump(random_elasticities, f, indent=4)
