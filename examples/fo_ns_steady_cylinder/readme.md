@main.py 
@train.py 
@eval.py 
@utils.py 
@models.py  
 
What does each function in this file do?
What does each file in this example do? Give me a one-paragraph summary of each.
What PDE is being solved in this example?
How is the PDE being solved in this example encoded in the loss function?
Explain the boundary conditions and how they're enforced.
Here's a comprehensive breakdown:

File Summaries
main.py — Entry point. Parses CLI flags (--config, --workdir), reads the mode field from config, and dispatches to either train.train_and_evaluate or eval.evaluate. Sets JAX to highest matmul precision and enables deterministic cuDNN ops.

utils.py — Data utilities. parabolic_inflow computes the Poiseuille parabolic velocity profile u = 4·U_max·y·(0.41-y)/0.41² for the channel inflow. get_dataset loads the reference FEM solution from data/ns_steady.npy, returning velocity/pressure fields, all boundary coordinate arrays, and kinematic viscosity ν.

train.py — Training loop. Loads data, optionally nondimensionalizes all quantities by U★=0.2 m/s and L★=0.1 m, constructs the model and evaluator, then runs the main optimization loop: steps the model, optionally updates loss weights (grad_norm/NTK), logs metrics to W&B, and saves checkpoints.

eval.py — Post-training evaluation. Restores a checkpoint, predicts u and v on the full domain, computes relative L2 errors, evaluates aerodynamic quantities (C_D, C_L, Δp) against the Nabh (1998) benchmark values, and saves tricontour comparison plots to PDF.

models.py — Core physics and model. Contains the PINN NavierStokes2D class implementing the network forward pass, PDE residuals via automatic differentiation, all loss terms, NTK diagonal computation, drag/lift integration, and the NavierStokesEvaluator logging wrapper.

Function-by-Function (models.py, the most complex file)
Function	What it does
__init__	Stores boundary arrays, computes Re, creates vmap'd prediction functions
neural_net	Rescales (x,y) into [0,1]², feeds into network, returns (u, v, p)
u_net / v_net / p_net	Thin extractors returning one output from neural_net
r_net	Computes PDE residuals using jacrev (1st derivatives) and hessian (2nd derivatives)
ru_net / rv_net / rc_net / u_out_net / v_out_net	Single-residual extractors from r_net
losses	Assembles all MSE loss terms into a dict (BCs + PDE residuals)
compute_diag_ntk	Computes diagonal NTK values per loss term for adaptive weighting
compute_l2_error	Relative L2 norm of u and v predictions vs. reference
u_v_grads	First-order spatial gradients of u and v (used by drag/lift)
compute_drag_lift	Integrates the full stress tensor on the cylinder surface via trapezoid rule to get F_D, F_L, Δp → C_D, C_L
PDE Being Solved
Steady-state incompressible Navier-Stokes (2D, DFG cylinder benchmark):


x-momentum:  u·∂u/∂x + v·∂u/∂y + ∂p/∂x − (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
y-momentum:  u·∂v/∂x + v·∂v/∂y + ∂p/∂y − (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
continuity:  ∂u/∂x + ∂v/∂y = 0
Flow past a circular cylinder (radius 0.05 m, center (0.2, 0.2)) in a 2.2×0.41 m channel at Re ≈ 20 (creeping flow regime). Reference benchmark values are from Nabh (1998).

PDE Encoding in the Loss Function
The PINN loss (models.py:113) is a sum of mean squared errors over collocation/boundary points:


L_total = w_u_in · ||u_pred − u_in||²       ← inflow u BC
        + w_v_in · ||v_pred||²               ← inflow v = 0
        + w_u_out· ||∂u/∂x − p||²           ← outflow natural BC
        + w_v_out· ||∂v/∂x||²               ← outflow natural BC
        + w_u_ns · ||u_pred||²               ← no-slip u = 0
        + w_v_ns · ||v_pred||²               ← no-slip v = 0
        + w_ru  · ||u·u_x + v·u_y + p_x − (u_xx+u_yy)/Re||²   ← x-momentum
        + w_rv  · ||u·v_x + v·v_y + p_y − (v_xx+v_yy)/Re||²   ← y-momentum
        + w_rc  · ||u_x + v_y||²             ← continuity
The PDE residuals (ru, rv, rc) are computed via jacrev (first derivatives) and hessian (second derivatives) in r_net (models.py:67). The weights w_* are either fixed (uniform) or adapted via gradient normalization or NTK-based schemes.

Boundary Conditions and Their Enforcement
BC	Location	Condition	Loss term	Code
Inflow (Dirichlet)	Left wall	u = parabolic profile, v = 0	u_in_loss, v_in_loss	models.py:114
Outflow (Natural/Neumann)	Right wall	∂u/∂x − p = 0, ∂v/∂x = 0	u_out_loss, v_out_loss	models.py:125
No-slip (Dirichlet)	Top/bottom walls + cylinder surface	u = 0, v = 0	u_noslip_loss, v_noslip_loss	models.py:133
All BCs are soft-enforced — they enter as additional MSE loss terms rather than being baked into the network architecture. The outflow condition ∂u/∂x − p = 0 is the "do-nothing" (stress-free) natural outflow BC, which arises from the weak form of Navier-Stokes; it's computed inside r_net at models.py:87 as u_out = u_x - p and v_out = v_x.

The no-slip coordinates combine both channel walls and the cylinder surface: noslip_coords = vstack(wall_coords, cylinder_coords) at models.py:34.

The explanation above covers all your questions — this was a read-only understanding task so no implementation plan is needed. Let me know if you'd like to dig deeper into any part, such as the NTK weighting scheme, the drag/lift integration, or how ForwardBVP from jaxpi works.

Mathematically describe the operations of the following:
1. NTK weighting scheme
2. the drag/lift integration
3. how ForwardBVP from jaxpi works
Let me read the jaxpi source files to give accurate mathematical descriptions.

Bash List jaxpi Python source files
IN
find /home/chaztikov/git/jaxpi -type f -name "*.py" | head -40

OUT
/home/chaztikov/git/jaxpi/setup.py
/home/chaztikov/git/jaxpi/jaxpi/archs.py
/home/chaztikov/git/jaxpi/jaxpi/models.py
/home/chaztikov/git/jaxpi/jaxpi/utils.py
/home/chaztikov/git/jaxpi/jaxpi/__init__.py
/home/chaztikov/git/jaxpi/jaxpi/samplers.py
/home/chaztikov/git/jaxpi/jaxpi/evaluator.py
/home/chaztikov/git/jaxpi/jaxpi/logging.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/models.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/utils.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/main.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/sweep.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/eval.py
/home/chaztikov/git/jaxpi/examples/allen_cahn/train.py
/home/chaztikov/git/jaxpi/examples/euler_tori/models.py
/home/chaztikov/git/jaxpi/examples/euler_tori/utils.py
/home/chaztikov/git/jaxpi/examples/euler_tori/main.py
/home/chaztikov/git/jaxpi/examples/euler_tori/eval.py
/home/chaztikov/git/jaxpi/examples/euler_tori/train.py
/home/chaztikov/git/jaxpi/examples/ldc/models.py
/home/chaztikov/git/jaxpi/examples/ldc/utils.py
/home/chaztikov/git/jaxpi/examples/ldc/main.py
/home/chaztikov/git/jaxpi/examples/ldc/eval.py
/home/chaztikov/git/jaxpi/examples/ldc/train.py
/home/chaztikov/git/jaxpi/examples/ks/models.py
/home/chaztikov/git/jaxpi/examples/ks/utils.py
/home/chaztikov/git/jaxpi/examples/ks/main.py
/home/chaztikov/git/jaxpi/examples/ks/eval.py
/home/chaztikov/git/jaxpi/examples/ks/train.py
/home/chaztikov/git/jaxpi/examples/burgers/models.py
/home/chaztikov/git/jaxpi/examples/burgers/utils.py
/home/chaztikov/git/jaxpi/examples/burgers/main.py
/home/chaztikov/git/jaxpi/examples/burgers/eval.py
/home/chaztikov/git/jaxpi/examples/burgers/train.py
/home/chaztikov/git/jaxpi/examples/fo_ns_steady_cylinder/models.py
/home/chaztikov/git/jaxpi/examples/fo_ns_steady_cylinder/utils.py
/home/chaztikov/git/jaxpi/examples/fo_ns_steady_cylinder/main.py
/home/chaztikov/git/jaxpi/examples/fo_ns_steady_cylinder/eval.py
/home/chaztikov/git/jaxpi/examples/fo_ns_steady_cylinder/train.py
/home/chaztikov/git/jaxpi/examples/adv/models.py
Read models.py
Read utils.py
Bash Read jaxpi models.py
IN
cat -n /home/chaztikov/git/jaxpi/jaxpi/models.py

OUT
     1	from functools import partial
     2	from typing import Any, Callable, Sequence, Tuple, Optional, Dict
     3	
     4	from flax.training import train_state
     5	from flax import jax_utils
     6	
     7	import jax.numpy as jnp
     8	from jax import lax, jit, grad, pmap, random, jacfwd, jacrev
     9	from jax.tree_util import tree_map, tree_reduce, tree_leaves
    10	
    11	import optax
    12	
    13	from jaxpi import archs
    14	from jaxpi.utils import flatten_pytree
    15	
    16	
    17	class TrainState(train_state.TrainState):
    18	    weights: Dict
    19	    momentum: float
    20	
    21	    def apply_weights(self, weights, **kwargs):
    22	        """Updates `weights` using running average  in return value.
    23	
    24	        Returns:
    25	          An updated instance of `self` with new weights updated by applying `running_average`,
    26	          and additional attributes replaced as specified by `kwargs`.
    27	        """
    28	
    29	        running_average = (
    30	            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
    31	        )
    32	        weights = tree_map(running_average, self.weights, weights)
    33	        weights = lax.stop_gradient(weights)
    34	
    35	        return self.replace(
    36	            step=self.step,
    37	            params=self.params,
    38	            opt_state=self.opt_state,
    39	            weights=weights,
    40	            **kwargs,
    41	        )
    42	
    43	
    44	def _create_arch(config):
    45	    if config.arch_name == "Mlp":
    46	        arch = archs.Mlp(**config)
    47	
    48	    elif config.arch_name == "ModifiedMlp":
    49	        arch = archs.ModifiedMlp(**config)
    50	
    51	    elif config.arch_name == "DeepONet":
    52	        arch = archs.DeepONet(**config)
    53	
    54	    else:
    55	        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")
    56	
    57	    return arch
    58	
    59	
    60	def _create_optimizer(config):
    61	    if config.optimizer == "Adam":
    62	        lr = optax.exponential_decay(
    63	            init_value=config.learning_rate,
    64	            transition_steps=config.decay_steps,
    65	            decay_rate=config.decay_rate,
    66	        )
    67	        tx = optax.adam(
    68	            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
    69	        )
    70	
    71	    else:
    72	        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")
    73	
    74	    # Gradient accumulation
    75	    if config.grad_accum_steps > 1:
    76	        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)
    77	
    78	    return tx
    79	
    80	
    81	def _create_train_state(config):
    82	    # Initialize network
    83	    arch = _create_arch(config.arch)
    84	    x = jnp.ones(config.input_dim)
    85	    params = arch.init(random.PRNGKey(config.seed), x)
    86	
    87	    # Initialize optax optimizer
    88	    tx = _create_optimizer(config.optim)
    89	
    90	    # Convert config dict to dict
    91	    init_weights = dict(config.weighting.init_weights)
    92	
    93	    state = TrainState.create(
    94	        apply_fn=arch.apply,
    95	        params=params,
    96	        tx=tx,
    97	        weights=init_weights,
    98	        momentum=config.weighting.momentum,
    99	    )
   100	
   101	    return jax_utils.replicate(state)
   102	
   103	
   104	class PINN:
   105	    def __init__(self, config):
   106	        self.config = config
   107	        self.state = _create_train_state(config)
   108	
   109	    def u_net(self, params, *args):
   110	        raise NotImplementedError("Subclasses should implement this!")
   111	
   112	    def r_net(self, params, *args):
   113	        raise NotImplementedError("Subclasses should implement this!")
   114	
   115	    def losses(self, params, batch, *args):
   116	        raise NotImplementedError("Subclasses should implement this!")
   117	
   118	    def compute_diag_ntk(self, params, batch, *args):
   119	        raise NotImplementedError("Subclasses should implement this!")
   120	
   121	    @partial(jit, static_argnums=(0,))
   122	    def loss(self, params, weights, batch, *args):
   123	        # Compute losses
   124	        losses = self.losses(params, batch, *args)
   125	        # Compute weighted loss
   126	        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
   127	        # Sum weighted losses
   128	        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
   129	        return loss
   130	
   131	    @partial(jit, static_argnums=(0,))
   132	    def compute_weights(self, params, batch, *args):
   133	        if self.config.weighting.scheme == "grad_norm":
   134	            # Compute the gradient of each loss w.r.t. the parameters
   135	            grads = jacrev(self.losses)(params, batch, *args)
   136	
   137	            # Compute the grad norm of each loss
   138	            grad_norm_dict = {}
   139	            for key, value in grads.items():
   140	                flattened_grad = flatten_pytree(value)
   141	                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)
   142	
   143	            # Compute the mean of grad norms over all losses
   144	            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
   145	            # Grad Norm Weighting
   146	            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)
   147	
   148	        elif self.config.weighting.scheme == "ntk":
   149	            # Compute the diagonal of the NTK of each loss
   150	            ntk = self.compute_diag_ntk(params, batch, *args)
   151	
   152	            # Compute the mean of the diagonal NTK corresponding to each loss
   153	            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)
   154	
   155	            # Compute the average over all ntk means
   156	            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
   157	            # NTK Weighting
   158	            w = tree_map(lambda x: (mean_ntk / x), mean_ntk_dict)
   159	
   160	        return w
   161	
   162	    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
   163	    def update_weights(self, state, batch, *args):
   164	        weights = self.compute_weights(state.params, batch, *args)
   165	        weights = lax.pmean(weights, "batch")
   166	        state = state.apply_weights(weights=weights)
   167	        return state
   168	
   169	    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
   170	    def step(self, state, batch, *args):
   171	        grads = grad(self.loss)(state.params, state.weights, batch, *args)
   172	        grads = lax.pmean(grads, "batch")
   173	        state = state.apply_gradients(grads=grads)
   174	        return state
   175	
   176	
   177	class ForwardIVP(PINN):
   178	    def __init__(self, config):
   179	        super().__init__(config)
   180	
   181	        if config.weighting.use_causal:
   182	            self.tol = config.weighting.causal_tol
   183	            self.num_chunks = config.weighting.num_chunks
   184	            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T
   185	
   186	
   187	class ForwardBVP(PINN):
   188	    def __init__(self, config):
   189	        super().__init__(config)
Bash Read jaxpi archs.py for network architecture details
IN
cat -n /home/chaztikov/git/jaxpi/jaxpi/archs.py

OUT
     1	from functools import partial
     2	from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
     3	
     4	from flax import linen as nn
     5	from flax.core.frozen_dict import freeze
     6	
     7	from jax import random, jit, vmap
     8	import jax.numpy as jnp
     9	from jax.nn.initializers import glorot_normal, normal, zeros, constant
    10	
    11	activation_fn = {
    12	    "relu": nn.relu,
    13	    "gelu": nn.gelu,
    14	    "swish": nn.swish,
    15	    "sigmoid": nn.sigmoid,
    16	    "tanh": jnp.tanh,
    17	    "sin": jnp.sin,
    18	}
    19	
    20	
    21	def _get_activation(str):
    22	    if str in activation_fn:
    23	        return activation_fn[str]
    24	
    25	    else:
    26	        raise NotImplementedError(f"Activation {str} not supported yet!")
    27	
    28	
    29	def _weight_fact(init_fn, mean, stddev):
    30	    def init(key, shape):
    31	        key1, key2 = random.split(key)
    32	        w = init_fn(key1, shape)
    33	        g = mean + normal(stddev)(key2, (shape[-1],))
    34	        g = jnp.exp(g)
    35	        v = w / g
    36	        return g, v
    37	
    38	    return init
    39	
    40	
    41	class PeriodEmbs(nn.Module):
    42	    period: Tuple[float]  # Periods for different axes
    43	    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    44	    trainable: Tuple[
    45	        bool
    46	    ]  # Specifies whether the period for each axis is trainable or not
    47	
    48	    def setup(self):
    49	        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
    50	        period_params = {}
    51	        for idx, is_trainable in enumerate(self.trainable):
    52	            if is_trainable:
    53	                period_params[f"period_{idx}"] = self.param(
    54	                    f"period_{idx}", constant(self.period[idx]), ()
    55	                )
    56	            else:
    57	                period_params[f"period_{idx}"] = self.period[idx]
    58	
    59	        self.period_params = freeze(period_params)
    60	
    61	    @nn.compact
    62	    def __call__(self, x):
    63	        """
    64	        Apply the period embeddings to the specified axes.
    65	        """
    66	        y = []
    67	
    68	        for i, xi in enumerate(x):
    69	            if i in self.axis:
    70	                idx = self.axis.index(i)
    71	                period = self.period_params[f"period_{idx}"]
    72	                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
    73	            else:
    74	                y.append(xi)
    75	
    76	        return jnp.hstack(y)
    77	
    78	
    79	class FourierEmbs(nn.Module):
    80	    embed_scale: float
    81	    embed_dim: int
    82	
    83	    @nn.compact
    84	    def __call__(self, x):
    85	        kernel = self.param(
    86	            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
    87	        )
    88	        y = jnp.concatenate(
    89	            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
    90	        )
    91	        return y
    92	
    93	
    94	class Dense(nn.Module):
    95	    features: int
    96	    kernel_init: Callable = glorot_normal()
    97	    bias_init: Callable = zeros
    98	    reparam: Union[None, Dict] = None
    99	
   100	    @nn.compact
   101	    def __call__(self, x):
   102	        if self.reparam is None:
   103	            kernel = self.param(
   104	                "kernel", self.kernel_init, (x.shape[-1], self.features)
   105	            )
   106	
   107	        elif self.reparam["type"] == "weight_fact":
   108	            g, v = self.param(
   109	                "kernel",
   110	                _weight_fact(
   111	                    self.kernel_init,
   112	                    mean=self.reparam["mean"],
   113	                    stddev=self.reparam["stddev"],
   114	                ),
   115	                (x.shape[-1], self.features),
   116	            )
   117	            kernel = g * v
   118	
   119	        bias = self.param("bias", self.bias_init, (self.features,))
   120	
   121	        y = jnp.dot(x, kernel) + bias
   122	
   123	        return y
   124	
   125	
   126	# TODO: Make it more general, e.g. imposing periodicity for the given axis
   127	
   128	
   129	class Mlp(nn.Module):
   130	    arch_name: Optional[str] = "Mlp"
   131	    num_layers: int = 4
   132	    hidden_dim: int = 256
   133	    out_dim: int = 1
   134	    activation: str = "tanh"
   135	    periodicity: Union[None, Dict] = None
   136	    fourier_emb: Union[None, Dict] = None
   137	    reparam: Union[None, Dict] = None
   138	
   139	    def setup(self):
   140	        self.activation_fn = _get_activation(self.activation)
   141	
   142	    @nn.compact
   143	    def __call__(self, x):
   144	        if self.periodicity:
   145	            x = PeriodEmbs(**self.periodicity)(x)
   146	
   147	        if self.fourier_emb:
   148	            x = FourierEmbs(**self.fourier_emb)(x)
   149	
   150	        for _ in range(self.num_layers):
   151	            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
   152	            x = self.activation_fn(x)
   153	
   154	        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
   155	        return x
   156	
   157	
   158	class ModifiedMlp(nn.Module):
   159	    arch_name: Optional[str] = "ModifiedMlp"
   160	    num_layers: int = 4
   161	    hidden_dim: int = 256
   162	    out_dim: int = 1
   163	    activation: str = "tanh"
   164	    periodicity: Union[None, Dict] = None
   165	    fourier_emb: Union[None, Dict] = None
   166	    reparam: Union[None, Dict] = None
   167	
   168	    def setup(self):
   169	        self.activation_fn = _get_activation(self.activation)
   170	
   171	    @nn.compact
   172	    def __call__(self, x):
   173	        if self.periodicity:
   174	            x = PeriodEmbs(**self.periodicity)(x)
   175	
   176	        if self.fourier_emb:
   177	            x = FourierEmbs(**self.fourier_emb)(x)
   178	
   179	        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
   180	        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
   181	
   182	        u = self.activation_fn(u)
   183	        v = self.activation_fn(v)
   184	
   185	        for _ in range(self.num_layers):
   186	            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
   187	            x = self.activation_fn(x)
   188	            x = x * u + (1 - x) * v
   189	
   190	        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
   191	        return x
   192	
   193	
   194	class MlpBlock(nn.Module):
   195	    num_layers: int
   196	    hidden_dim: int
   197	    out_dim: int
   198	    activation: str
   199	    reparam: Union[None, Dict]
   200	    final_activation: bool
   201	
   202	    def setup(self):
   203	        self.activation_fn = _get_activation(self.activation)
   204	
   205	    @nn.compact
   206	    def __call__(self, x):
   207	        for _ in range(self.num_layers):
   208	            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
   209	            x = self.activation_fn(x)
   210	
   211	        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
   212	        if self.final_activation:
   213	            x = self.activation_fn(x)
   214	
   215	        return x
   216	
   217	
   218	class DeepONet(nn.Module):
   219	    arch_name: Optional[str] = "DeepONet"
   220	    num_branch_layers: int = 4
   221	    num_trunk_layers: int = 4
   222	    hidden_dim: int = 256
   223	    out_dim: int = 1
   224	    activation: str = "tanh"
   225	    periodicity: Union[None, Dict] = None
   226	    fourier_emb: Union[None, Dict] = None
   227	    reparam: Union[None, Dict] = None
   228	
   229	    def setup(self):
   230	        self.activation_fn = _get_activation(self.activation)
   231	
   232	    @nn.compact
   233	    def __call__(self, u, x):
   234	        u = MlpBlock(
   235	            num_layers=self.num_branch_layers,
   236	            hidden_dim=self.hidden_dim,
   237	            out_dim=self.hidden_dim,
   238	            activation=self.activation,
   239	            final_activation=False,
   240	            reparam=self.reparam,
   241	        )(u)
   242	
   243	        x = Mlp(
   244	            num_layers=self.num_trunk_layers,
   245	            hidden_dim=self.hidden_dim,
   246	            out_dim=self.hidden_dim,
   247	            activation=self.activation,
   248	            periodicity=self.periodicity,
   249	            fourier_emb=self.fourier_emb,
   250	            reparam=self.reparam,
   251	        )(x)
   252	
   253	        y = u * x
   254	        y = self.activation_fn(y)
   255	        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
   256	        return y
Now I have everything needed. Here's the full mathematical treatment:

1. NTK Weighting Scheme
The Diagonal NTK Element
For a scalar-valued function f(θ; x) (one of the loss residuals), the diagonal NTK element at point x is (jaxpi/utils.py:27):


J(x) = ∂f/∂θ  ∈ ℝ^P          (gradient flattened to a P-dim vector)
K(x) = J(x)ᵀ J(x) = ‖∂f/∂θ‖²  ∈ ℝ
This is the diagonal of the full NTK matrix K(xᵢ, xⱼ) = J(xᵢ)ᵀJ(xⱼ). The full NTK governs the training dynamics: d θ/dt = -η K θ in the infinite-width limit.

NTK Weight Computation (jaxpi/models.py:148)
For each loss term k (u_in, v_in, ru, rc, …) with associated sample points {xᵢᵏ}:


μ_k = (1/Nₖ) Σᵢ K_k(xᵢᵏ)         (mean diagonal NTK for loss k)

μ̄  = (1/K) Σ_k μ_k                (global mean across all K loss terms)

w_k = μ̄ / μ_k                     (NTK weight for loss k)
Interpretation: a loss term with a large average NTK value is already learning quickly. Dividing by μ_k downweights it, equalizing effective learning rates across all terms.

Grad-Norm Weighting (jaxpi/models.py:133)
Alternatively, weights can be based on gradient norms:


G_k = ‖∂L_k/∂θ‖                   (Euclidean norm of loss-k gradient)

Ḡ   = (1/K) Σ_k G_k

w_k = Ḡ / G_k
This enforces equal gradient magnitudes across all loss terms.

Exponential Moving Average (jaxpi/models.py:29)
Weights are not applied immediately — they are smoothed with momentum β:


w_k^(t+1) = β · w_k^(t) + (1 − β) · ŵ_k^(t)
where ŵ_k^(t) is the newly computed NTK or grad-norm weight.

Total Weighted Loss

L(θ) = Σ_k w_k · L_k(θ)
where each L_k is an MSE over its sample set.

2. Drag/Lift Integration
Stress Tensor
For incompressible Newtonian flow, the Cauchy stress tensor is:


σ_ij = −p δ_ij + ν (∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
The traction on a surface with outward normal n = (nₓ, nᵧ) is t = σ·n. In 2D:


tₓ = (−p + 2ν ∂u/∂x) nₓ  +  ν (∂u/∂y + ∂v/∂x) nᵧ

tᵧ = ν (∂u/∂y + ∂v/∂x) nₓ  +  (−p + 2ν ∂v/∂y) nᵧ
Cylinder Parameterization (models.py:240)
The cylinder surface is parameterized by θ ∈ [0, 2π]:


x(θ) = r cos θ + xc ,   y(θ) = r sin θ + yc
nₓ(θ) = cos θ ,          nᵧ(θ) = sin θ
ds = r dθ
Force Integrals

F_D = ∮ tₓ ds = ∫₀²π tₓ(θ) · r dθ

F_L = ∮ tᵧ ds = ∫₀²π tᵧ(θ) · r dθ
Trapezoidal Rule (models.py:278)
With N discrete points and arc-length element ds = r·dθ:


F_D ≈ (ds/2) Σᵢ₌₀^{N−2} [Iₓ(θᵢ) + Iₓ(θᵢ₊₁)]
F_L ≈ (ds/2) Σᵢ₌₀^{N−2} [Iᵧ(θᵢ) + Iᵧ(θᵢ₊₁)]
where Iₓ(θ) = tₓ(θ) and Iᵧ(θ) = tᵧ(θ).

Nondimensionalization (models.py:297)

C_D = 2 F_D / (U★² L★)

C_L = 2 F_L / (U★² L★)
The pressure difference is evaluated at the front and back stagnation points:


Δp = p(xc − r, yc) − p(xc + r, yc)     [nondim → dimensional via × U★²]
3. ForwardBVP and the Full PINN Stack
ForwardBVP is a thin wrapper that inherits entirely from PINN — it adds no logic, just signals "this is a boundary value problem" (vs. ForwardIVP which adds causal time-marching weights). The full mathematical stack is:

Network Architecture
MLP (archs.py:129):


z⁽⁰⁾ = E(x)                          optional: Fourier or periodic embedding
z⁽ˡ⁾ = σ(W⁽ˡ⁾ z⁽ˡ⁻¹⁾ + b⁽ˡ⁾),  l = 1,…,L
ŷ    = W⁽ᴸ⁺¹⁾ z⁽ᴸ⁾ + b⁽ᴸ⁺¹⁾
ModifiedMLP (Wang et al. 2022, archs.py:158):


u = σ(Wᵤ x + bᵤ)
v = σ(Wᵥ x + bᵥ)
z⁽ˡ⁾ = σ(W⁽ˡ⁾ z⁽ˡ⁻¹⁾ + b⁽ˡ⁾) ⊙ u  +  (1 − σ(·)) ⊙ v
The u,v gating from the input layer helps preserve gradient flow through the network.

Weight Factorization (optional reparameterization, archs.py:29):


W = g ⊙ v,    g = exp(μ + ε),  ε ~ N(0, σ²)
This decouples the weight magnitude (g) from direction (v), aiding optimization conditioning.

Fourier Embedding (archs.py:79):


E(x) = [cos(Bx), sin(Bx)],   B ~ N(0, σ²ₑ)
where B ∈ ℝ^{d × m/2} is a fixed random matrix. This projects inputs into a high-frequency feature space, mitigating spectral bias.

Training Step (jaxpi/models.py:169)
Each step computes:


g = ∂L(θ; w)/∂θ = Σ_k w_k · ∂L_k/∂θ

θ^(t+1) = Adam(θ^(t), g)
with Adam using exponential LR decay:


α(t) = α₀ · r^(t/T)
Adam first and second moment estimates (β₁, β₂) and the pmap over devices average gradients before the update:


g ← (1/D) Σ_d g_d    (all-reduce mean across D devices)
Summary of Class Hierarchy

PINN                    — loss weighting, optimizer step, weight update
  └── ForwardBVP        — no additions (BVP marker)
        └── NavierStokes2D  — neural_net, r_net, losses, compute_diag_ntk, compute_drag_lift
The only problem-specific code lives in NavierStokes2D. Everything else — architecture, optimizer, weighting, training loop — is reusable infrastructure from jaxpi.
