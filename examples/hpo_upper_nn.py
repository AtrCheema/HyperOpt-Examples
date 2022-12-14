"""
====================================================
hyperparameter optimization with Model class for NNs
====================================================
"""


from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import get_version_info
from ai4water.hyperopt import Categorical, Real, Integer

for k,v in get_version_info().items():
    print(f"{k} version: {v}")

# %%

data = busan_beach()
print(data.shape)

# %%
input_features = data.columns.tolist()[0:-1]
print(input_features)

# %%
output_features = data.columns.tolist()[-1:]
print(output_features)

# %%
# build the model

lookback = 14

model = Model(
    model = {"layers": {
        "Input": {"shape": (lookback, len(input_features))},
        "LSTM": {"units": Integer(10, 20, name="units"),
                 "activation": Categorical(["relu", "elu", "tanh"], name="activation")},
        "Dense": 1
    }},
    lr=Real(0.00001, 0.01, name="lr"),
    batch_size=Categorical([4, 8, 12, 16, 24], name="batch_size"),
    train_fraction=1.0,
    split_random=True,
    epochs=50,
    ts_args={"lookback": lookback},
    input_features=input_features,
    output_features=output_features,
    x_transformation="zscore",
    y_transformation={"method": "log", "replace_zeros": True, "treat_negatives": True},
)

# %%

optimizer = model.optimize_hyperparameters(
    data=data,
    num_iterations=25,
    process_results=False,
    refit=False,
)

# %%
_ = optimizer.plot_importance(save=False)
