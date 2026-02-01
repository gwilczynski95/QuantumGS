from .encoders import *

shared_encoders = {
    "none": SharedIdentity,
    "mlp": SharedMLPEncoder,
    "positional": SharedPositionalMLPEncoder,
    "rff": SharedRFFMLPEncoder,
    "lff": lambda hidden_dim=32: SharedRFFMLPEncoder(hidden_dim=hidden_dim, is_learnable=True),
    "hashgrid": SharedHashGridMLPEncoder
}
