import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from typing import Literal
from lvdm.common import default
from typing import List 
import numpy as np

from diffusers.models.attention import FeedForward
from timm.models.layers import trunc_normal_

class PoseGuidedConditionEncoder(nn.Module):
    def __init__(self, 
                 input_dims: List[int],
                 num_tokens: List[int],
                 context_tokens: int,
                 context_dim: int,
                 num_attention_heads=8,
                 inner_dim=None,
                 output_dim=None,
                 dropout=0.0,
                 num_blocks=4,
                 activation_fn="geglu",
                 video_length: int = 16
                 ):
        super().__init__()

        #output_dim = output_dim if output_dim is not None else input_dims[0]
        inner_dim = inner_dim if inner_dim is not None else output_dim

        attn_blocks = []

        for i in range(num_blocks):
            odim = inner_dim if i < num_blocks - 1 else input_dims # TODO: fix this
            block = PoseAgentAttention(
                query_dims=input_dims,
                num_tokens=num_tokens,
                context_tokens=context_tokens,
                context_dim=context_dim,
                heads=num_attention_heads,
                dropout=dropout,
                out_dim=None
            )
            attn_blocks.append(block)

        self.attention_blocks = nn.ModuleList(attn_blocks)
        self.video_length = video_length

        self.ff_out_lw = []
        self.norms = []
        for i, qdim in enumerate(input_dims):
            self.ff_out_lw.append(
                nn.Sequential(nn.LayerNorm(qdim), FeedForward(qdim, dropout=dropout, activation_fn=activation_fn))
            ) 
            self.norms.append(nn.LayerNorm(qdim))

        self.ff_out_lw = nn.ModuleList(self.ff_out_lw)
        self.norms = nn.ModuleList(self.norms)

    def precompute_kv_attention(self, context: Tensor, pose_embedding: Tensor, attn_mask: Tensor = None):
        for block in self.attention_blocks:
            block.compute_kv_attention(context, pose_embedding, attn_mask)
        

    def forward(self, hidden_state, context=None, pose=None, attention_mask=None, block_index:int=0):
        
        #import ipdb; ipdb.set_trace()
        #print(f"Block {block_index}")
        #print(f"Input: {hidden_state.shape}")
        #hidden_state = rearrange(hidden_state, 'B N H D -> B N (H D)')
        #return hidden_state
        #hidden_state = rearrange(hidden_state, 'B D T H W -> B T (H W) D')
        hidden_state = rearrange(hidden_state, '(B T) N H D -> B T N (H D)', T=self.video_length)
        
        _layer_wise_cross_attn = False
        if isinstance(context, list):
            assert len(context) == len(self.attention_blocks), "Please provide same number of context conditions as layers if multiple are provided"
            _layer_wise_cross_attn = True
        for i, attn_block in enumerate(self.attention_blocks):
            cur_context = context[i] if _layer_wise_cross_attn else context
            norm_hidden_state = self.norms[block_index](hidden_state)
            hidden_state = attn_block(
                x = norm_hidden_state,   
                context = cur_context,
                pose_embedding = pose,
                attn_mask = attention_mask,
                block_index = block_index
            ) + hidden_state
            
        hidden_state = self.ff_out_lw[block_index](hidden_state) + hidden_state
        hidden_state = rearrange(hidden_state, 'B T N D -> (B T) N D')

        #print(f"Output: {hidden_state.shape}")
        return hidden_state

class PoseAgentAttention(nn.Module):

    def __init__(self, 
                    query_dims: List[int],
                    num_tokens: List[int],
                    context_tokens: int,
                    context_dim,
                    out_dim=None,
                    attention_mode: Literal["softmax","linear"]="linear",
                    heads=8,
                    attn_drop=0., 
                    proj_drop=0.,
                    dim_head=64,
                    window=14,
                    agent_num=10,
                    dropout=0.0,
                    num_register_tokens=0):
        assert len(query_dims) == len(num_tokens), "Please provide same number of query and token dimensions"
        super().__init__()
        inner_dim = dim_head * heads # TODO: This is set in up-stream module and ignored here
        self.context_dim = context_dim

        self.unique_dims = sorted(list(set(query_dims)))
        self.unique_tokens = sorted(list(set(num_tokens)))[::-1]
        self.num_tokens = num_tokens
        self.query_dims = query_dims

        self.scale = dim_head**-0.5
        self.num_heads = heads
        self.dim_head = dim_head
        self.window = window
        self.inner_dim = inner_dim
        self.num_blocks = len(query_dims)
        self.to_q_lw = []
        self.to_a_lw = {}
        for i, q_dim in enumerate(query_dims):
            self.to_q_lw.append(nn.Linear(q_dim, inner_dim, bias=False))
        
        #import ipdb; ipdb.set_trace()
        for i, (a_dim, n_tokens) in enumerate(zip(self.unique_dims, self.unique_tokens)):
            self.to_a_lw[str(a_dim)] = nn.Linear(a_dim, inner_dim, bias=False)
            setattr(self, f"agent_bias_{a_dim}", nn.Parameter(torch.zeros(1, 1, n_tokens, context_tokens)))
        
        self.to_a_lw = nn.ModuleDict(self.to_a_lw)
        self.to_q_lw = nn.ModuleList(self.to_q_lw)

        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out_lw = []
        self.dwc_lw = []
        for i, q_dim in enumerate(query_dims):
            odim = out_dim if out_dim is not None else q_dim
            self.proj_out_lw.append(nn.Linear(self.inner_dim, odim, bias=False))
            setattr(self, f"query_bias_{i}", nn.Parameter(torch.zeros(1, 1, num_tokens[i], num_tokens[i])))
            self.dwc_lw.append(nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=(3, 3), padding=1, groups=inner_dim))

        self.proj_out_lw = nn.ModuleList(self.proj_out_lw)
        self.dwc_lw = nn.ModuleList(self.dwc_lw)

        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = dropout
        # TODO: Check how this is used in the original agent attention

        self.agent_kv = None
        self.agent_tokens = None
        self.v_dwc = None

    def compute_kv_attention(self, context: Tensor, pose_embedding: List[Tensor], attn_mask: Tensor = None):
        pose_embedding = [rearrange(p, 'B D T H W -> B T (H W) D') for p in pose_embedding]
        # Retrieve shape parameters
        B, C, N, D = context.shape
        H = int(N ** 0.5)
        W = int(N ** 0.5)
        T = pose_embedding[0].shape[1]
        M = context.shape[2]

        pose_embedding = pose_embedding[:len(self.to_a_lw.keys())]
        #context = rearrange(context, 'B C N D -> B C N D')
        #pose_embedding = rearrange(pose_embedding, 'B T N D -> B T N D')
        ##-- Projection In --##
        k = self.to_k(context) # [(B, C), N, D]
        v = self.to_v(context) # [(B, C), N, D]
        self.agent_tokens = {p.shape[-1]: self.to_a_lw[str(p.shape[-1])](p) for p in pose_embedding} # [(B, T), N, D]

        # TODO: Check if this dim manipulation is valid
        k = rearrange(k, f'B C N D -> B (C N) D')
        k = repeat(k, f'B N D -> B {T} N D')
        
        v = rearrange(v, 'B C N D -> (B C) N D')
        v = repeat(v, f'B N D -> B {T} N D')

        self.agent_kv = {}
        #import ipdb; ipdb.set_trace()
        for i, (name, a_token) in enumerate(self.agent_tokens.items()):
            M = pose_embedding[i].shape[-2]
            agent_attn = (a_token * self.scale) @ k.transpose(-2, -1)
            agent_attn = agent_attn.view(B, T, M, N, C)
            agent_attn = rearrange(agent_attn, 'B T M N C -> (B C) T M N')

            agent_attn = agent_attn + getattr(self, f"agent_bias_{name}")
            agent_attn = self.softmax(agent_attn)
            agent_attn = self.attn_drop(agent_attn)


            agent_kv = agent_attn @ v # (B C) T (H W) INNER_DIM
            agent_kv = agent_kv.view(B, C, T, M, self.inner_dim)
            if attn_mask is not None:
                agent_kv *= attn_mask
            agent_kv = torch.max(agent_kv, dim=1)[0]
            self.agent_kv[name] = agent_kv

        #import ipdb; ipdb.set_trace()
        v = rearrange(v, 'B T N D -> (B T) D N')
        v = v.view(B*C*T, self.inner_dim, H, W)
        #v_dwc = [dwc(v) for dwc in self.dwc_lw]
        #v_dwc = torch.stack(v_dwc, dim=0)
        #v_dwc = v_dwc.view(self.num_blocks, B, C, T, self.inner_dim, H, W)
        #if attn_mask is not None:
        #    v_dwc = v_dwc * attn_mask
        #v_dwc = torch.max(v_dwc, dim=2)[0]
        #scale_factor = 2
        self.v_dwc = []
        #import ipdb; ipdb.set_trace()
        for i, (num_tokens, token_dim) in enumerate(zip(self.num_tokens, self.query_dims)):
            v_dwc = self.dwc_lw[i](v)
            v_dwc = v_dwc.view(B, C, T, self.inner_dim, H, W)
            if attn_mask is not None:
                v_dwc = v_dwc * attn_mask
            v_dwc = torch.max(v_dwc, dim=1)[0]

            tokens = v_dwc.shape[-2]*v_dwc.shape[-1]
            scale_factor = np.sqrt(num_tokens/tokens)
            if scale_factor != 1:
                scale_factor = int(scale_factor) if scale_factor > 1 else scale_factor
                self.v_dwc.append(F.interpolate(rearrange(v_dwc, "B T D H W -> (B T) D H W"), scale_factor=scale_factor, mode='bilinear'))
                self.v_dwc[i] = self.v_dwc[i].view(B, T, self.inner_dim, int(H*scale_factor), int(W*scale_factor))
            else:
                self.v_dwc.append(v_dwc)
            
            self.v_dwc[i] = rearrange(self.v_dwc[i], 'B T D H W -> B T (H W) D')


    def forward(self, x: Tensor, context: Tensor = None, pose_embedding: Tensor = None, attn_mask: Tensor = None, block_index: int=0):
        """
        Shape parameters: 
            B: batch size
            T: sequence length
            N: Number of tokens, (H W)
            C: Number of conditioning frames
        Args:
            x: latent features from the UNet [B, T, N, D_x]
            c_embedding: CLIP-embedded features of the conditioning frames [B, C, 256, D_c]
            pose_embedding: Encoded pluecker pose embedding [B, T, N, D_p]
            attn_mask: Mask indicating the epipolar line of in each 


        """
        #import ipdb; ipdb.set_trace()
        # Retrieve shape parameters
        if context is None or pose_embedding is None:
            assert self.agent_kv is not None or len(self.agent_kv)>0, "If 'context' or 'pose_embedding' is not provided, please precompute the key-value agent attention."
        else:
            self.compute_kv_attention(context, pose_embedding, attn_mask)

        B, T, N, D = x.shape
        #n = self.agent_kv.shape[-2]

        ##-- Projection In --##
        input_dim = x.shape[-1]
        q = self.to_q_lw[block_index](x) # [(B, T), N, D]

    
        agent_q = (q * self.scale) @ self.agent_tokens[input_dim].transpose(-2, -1) + getattr(self, f"query_bias_{block_index}")
        if attn_mask is not None:
            agent_q += attn_mask
        agent_q = self.softmax(agent_q)
        agent_q = self.attn_drop(agent_q)

        x = agent_q @ self.agent_kv[input_dim]
        #print(f"Block {block_index}: x: {x.shape}")
        #import ipdb; ipdb.set_trace()
        x = x + self.v_dwc[block_index]

        x = self.proj_out_lw[block_index](x)
        x = self.proj_drop(x)
        return x
    
