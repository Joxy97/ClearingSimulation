"""RBM experiment configuration."""

config = {
    "device": "cuda",  # "auto" | "cpu" | "cuda:0" | "mps"

    "data": {
        "csv_path": "data/data_mini.csv",  # Path to training data CSV
        "drop_cols": [],
    },

    "model": {
        "bm_type": "rbm",   # Currently only RBM is supported
        "visible_blocks": {"s": 3, "r_prev": 20, "r": 20},
        "hidden_blocks": {"h1": 2, "h":1, "h2": 2},
        "cross_block_restrictions": [("s", "h1"), ("s", "h2"), ("r_prev", "h2"), ("r", "h1")],
        "initialization": "random",
    },

    "preprocess": {
        "q_low": 0.001,
        "q_high": 0.999,
        "add_missing_bit": True,
        "max_categories": 200,
        "min_category_freq": 2,
    },

    "dataloader": {
        "batch_size": 8000,
        "split": [0.8, 0.1, 0.1],
        "seed": 42,
        "shuffle_train": True,
        "num_workers": 0,
        "drop_last_train": True,
        "pin_memory": "auto",  # "auto" | True | False
    },

    "train": {
        "epochs": 500,
        "lr": 1e-2,
        "k": 10,
        "kind": "mean-field",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "clip_value": 0.05,
        "clip_norm": 5.0,
        "lr_schedule": {"mode": "cosine", "min_lr": 1e-4},
        "sparse_hidden": True,
        "rho": 0.1,
        "lambda_sparse": 0.01,
        "early_stopping": False,
        "es_patience": 8,
    },

    "eval": {
        "recon_k": 1,
    },

    "conditional": {
        "clamp_idx": list(range(0, 2003)),      # clamp s + r_prev
        "target_idx": list(range(2003, 4003)),   # sample r
        "n_samples": 100,
        "burn_in": 500,
        "thin": 10,
    },
}
