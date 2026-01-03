import torch
import torch.nn as nn
from einops import rearrange

from model.CamC2V_base.epipolar import EpipolarCrossAttention
from model.CamC2V_base.util_modules import CrossNormalization




class InsertionCrossAttention(nn.Module):

    def __init__(self,
            query_dim: int,
            head_dim: int,
            context_dim: int = None,
            norm_num_groups: int = 32,
            dropout: float = 0.25,
            num_register_tokens: int = 0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.head_dim = head_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        self.norm_num_groups = norm_num_groups
        self.num_register_tokens = num_register_tokens

        self.proj_in = nn.Linear(self.query_dim, self.query_dim)
        self.proj_out = nn.Linear(self.query_dim, self.query_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = EpipolarCrossAttention(
            query_dim = self.query_dim,
            context_dim = self.context_dim,
            dim_head = head_dim,
            num_register_tokens=self.num_register_tokens,
        )

        self.group_norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
        self.cross_norm = CrossNormalization(dims=(-3, -2, -1))


    def forward(self, hidden_state, past_hidden_state=None, attn_mask=None, batch_size = None):
        """
        
            Arguments:
                hidden_state: hidden state from skip connection [B, T, D, H, W]
                past_hidden_state: hidden state from same layer from previous auto-regressive generation [B, T, D, H, W]
        
        """
        #import ipdb; ipdb.set_trace()
        #_hidden_state_squeeze = False; _past_hidden_state_squeeze = False
        #if len(hidden_state.shape)==4:
        #    hidden_state = hidden_state.unsqueeze(0)  
        #    _hidden_state_squeeze = True
        #if past_hidden_state is not None and len(past_hidden_state.shape)==4:
        #    past_hidden_state = past_hidden_state.unsqueeze(0)  
        #    _past_hidden_state_squeeze = True

        T, D, H, W = hidden_state.shape
        T = T//batch_size

        if batch_size is None:
            B = 1
        else:
            B = batch_size

        past_hidden_state = past_hidden_state if past_hidden_state is not None else hidden_state

        hidden_state_norm = self.group_norm(hidden_state)
        past_hidden_state = self.cross_norm(past_hidden_state, hidden_state_norm)

        hidden_state_norm = rearrange(hidden_state_norm, '(B T) D H W -> B (T H W) D', B=B, T=T)
        past_hidden_state = rearrange(past_hidden_state, '(B T) D H W -> B (T H W) D', B=B, T=T)

        hidden_state_norm = self.proj_in(hidden_state_norm)
        attn_out = self.attn(hidden_state_norm, past_hidden_state, attn_mask=attn_mask)
        attn_out = self.proj_out(attn_out)

        attn_out = rearrange(attn_out, "B (T H W) D -> (B T) D H W", T=T, H=H, W=W)

        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state + attn_out

        #if _hidden_state_squeeze:
        #    hidden_state = hidden_state.squeeze(0)

        return hidden_state

