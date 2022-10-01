"""
==============================================
hyperparameter optimization with Model class
==============================================
"""

from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import get_version_info
from ai4water.hyperopt import Categorical, Real, Integer

# %%

for k,v in get_version_info().items():
    print(f"{k} version: {v}")

# %%

# prepare the data
data = busan_beach()
print(data.shape)

#%%
input_features = data.columns.tolist()[0:-1]
print(input_features)

# %%
output_features = data.columns.tolist()[-1:]
print(output_features)

#%%
# build the model

model = Model(
    model = {"XGBRegressor": {
        "iterations": Integer(low=10, high=30, name='iterations', num_samples=10),
        "learning_rate": Real(low=0.09, high=0.3, prior='log', name='learning_rate', num_samples=10),
        "l2_leaf_reg": Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=10),
        "model_size_reg": Real(low=0.1, high=10, name='model_size_reg', num_samples=10),
        "rsm": Real(low=0.1, high=0.5, name='rsm', num_samples=10),
        "border_count": Integer(low=32, high=50, name='border_count', num_samples=10),
        "feature_border_type": Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type'),
        "n_jobs": 0,
    }},
    train_fraction=1.0,
    split_random=True,
    input_features=input_features,
    output_features=output_features,
    x_transformation="zscore",
    y_transformation={"method": "log", "replace_zeros": True, "treat_negatives": True},
)

#%%

optimizer = model.optimize_hyperparameters(
    data=data,
    num_iterations=25,
    process_results=False  # we can set it to True if we want post-processing of results
)

#%%
optimizer._plot_convergence()

#%%
optimizer.best_iter()

#%%

optimizer.best_paras()

#%%

print(model.config['model'])

#%%

print(model._model)
