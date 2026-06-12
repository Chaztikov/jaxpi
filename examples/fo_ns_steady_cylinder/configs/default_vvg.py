import ml_collections


def get_config():
    """Hyperparameter configuration for the first-order (VVG) NS formulation.

    Network outputs: [u, v, p, ux, vy, uy, vx] where ux=du/dx etc.
    Compatibility and consistency conditions are enforced as additional losses.
    No second-order autodiff is needed — one jacrev pass suffices.
    """
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-NS_steady_cylinder"
    wandb.name = "default_vvg"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # PDE definition file (YAML)
    config.pde_path = "./configs/pde_vvg.yaml"

    # Arch — out_dim must match len(state_vars) in pde_vvg.yaml
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.out_dim = 7
    arch.activation = "gelu"
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 10.0, "embed_dim": 128}
    )
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size_per_device = 1024

    # Weighting — keys must cover all loss terms: 4 BCs + 13 PDE equations
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {
            # Dirichlet / no-slip BCs (hardcoded in losses())
            "u_in": 1.0,
            "v_in": 1.0,
            "u_noslip": 1.0,
            "v_noslip": 1.0,
            # PDE residuals (from pde_vvg.yaml — domain)
            "r_continuity": 1.0,
            "r_momentum_x": 1.0,
            "r_momentum_y": 1.0,
            "r_compatibility_ux": 1.0,
            "r_compatibility_uy": 1.0,
            "r_compatibility_vx": 1.0,
            "r_compatibility_vy": 1.0,
            "r_consistency_grad_trace_velocity_gradient_1": 1.0,
            "r_consistency_grad_trace_velocity_gradient_2": 1.0,
            "r_consistency_curl_velocity_gradient_1": 1.0,
            "r_consistency_curl_velocity_gradient_2": 1.0,
            # Outflow BCs (from pde_vvg.yaml — outflow)
            "u_out": 1.0,
            "v_out": 1.0,
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = None
    saving.num_keep_ckpts = 10

    config.input_dim = 2
    config.seed = 42

    return config
