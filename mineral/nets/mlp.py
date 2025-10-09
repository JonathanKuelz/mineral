from typing import Any, Dict, Literal, Optional, Sequence

import torch
import torch.nn as nn


def Norm(norm_type, size, **norm_kwargs):
    if norm_type is None:
        return nn.Identity()

    module = torch.nn
    Cls = getattr(module, norm_type)
    norm = Cls(size, **norm_kwargs)
    return norm


def Act(act_type, **act_kwargs):
    if act_type is None:
        return nn.Identity()

    module = torch.nn.modules.activation
    Cls = getattr(module, act_type)
    act = Cls(**act_kwargs)
    return act


class MLP(nn.Module):
    """A thin wrapper around nn.Sequential to create an MLP with various options."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_units: Sequence[int],
        dropout: Optional[float] = None,
        dropout_kwargs: Optional[Dict[str, Any]] = None,
        where_dropout: Literal['every', 'first', 'last'] = "every",
        norm_type = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act_type: str = "ReLU",
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        plain_last: bool = True,
    ):
        """
        Create an MLP with the given specifications.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_units (Sequence[int]): Sizes of hidden layers.
            dropout (float, optional): Dropout probability. If None, no dropout is applied.
            dropout_kwargs (dict, optional): Additional arguments for the Dropout layers.
            where_dropout (str): Where to apply dropout. Options are "every", "first", "last".
            norm_type (str, optional): Type of normalization layer to use. If None, no normalization is applied.
            norm_kwargs (dict, optional): Additional arguments for the normalization layers.
            act_type (str): Type of activation function to use. If None, no activation is applied.
            act_kwargs (dict, optional): Additional arguments for the activation functions.
            bias (bool): Whether to include bias terms in the linear layers.
            plain_last (bool): If True, the last layer will not have activation, normalization, or dropout.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        if act_kwargs is None:
            act_kwargs = {}
        if dropout_kwargs is None:
            dropout_kwargs = {}

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.units = [*hidden_units, out_dim]

        in_size = in_dim
        layers = []
        for i, out_size in enumerate(self.units):
            lin = nn.Linear(in_size, out_size, bias=bias)
            layers.append(lin)
            if plain_last and i == len(self.units) - 1:
                break

            if dropout is not None:
                add_dropout = False
                if i == 0 and where_dropout in ("every", "first"):
                    add_dropout = True
                if (i != 0 and i != len(self.units) - 1) and where_dropout in ("every",):
                    add_dropout = True
                if i == len(self.units) - 1 and where_dropout in ("every", "last"):
                    add_dropout = True
                if add_dropout:
                    dp = nn.Dropout(dropout, **dropout_kwargs)
                    layers.append(dp)
            if norm_type is not None:
                norm = Norm(norm_type, out_size, **norm_kwargs)
                layers.append(norm)
            if act_type is not None:
                act = Act(act_type, **act_kwargs)
                layers.append(act)
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
