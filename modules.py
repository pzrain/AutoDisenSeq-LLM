import numpy as np

import copy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from typing import Optional, Tuple
from ops import PRIMITIVES, OPS, ConvBN

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor_, attention_mask):
        input_tensor = self.LayerNorm(input_tensor_)
        
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        
        hidden_states = torch.mul(
            torch.bmm(mixed_query_layer, mixed_key_layer.transpose(1, 2)),
            attention_mask
            ) / math.sqrt(self.attention_head_size)

        hidden_states[hidden_states < 1e-3] = -10000
        
        hidden_states = torch.softmax(
            hidden_states,
            dim=2
        )
        hidden_states = torch.bmm(hidden_states, mixed_value_layer)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states += input_tensor_

        return hidden_states


class PointWiseFeedForward(nn.Module):
    def __init__(self, args):
        super(PointWiseFeedForward, self).__init__()
        self.conv1d_1 = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=(1,))
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=(1,))
        self.linear_2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout_1 = nn.Dropout(args.hidden_dropout_prob)
        self.dropout_2 = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor_):
        
        hidden_states = self.LayerNorm(input_tensor_)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.dropout_1(hidden_states)
        hidden_states += input_tensor_

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = PointWiseFeedForward(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class SASEncoder(nn.Module):
    def __init__(self, args):
        super(SASEncoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            return hidden_states
        return all_encoder_layers


class DSSEncoder(nn.Module):
    def __init__(self, args):
        super(DSSEncoder, self).__init__()
        # self.sas_encoder = SASEncoder(args)
        # prototypical intention vector for each intention
        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(args.hidden_size) *
                                                         (1 / np.sqrt(args.hidden_size)))
                                            for _ in range(args.num_intents)])

        self.layernorm1 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm3 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm4 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm5 = LayerNorm(args.hidden_size, eps=1e-12)

        self.w = nn.Linear(args.hidden_size, args.hidden_size)

        self.b_prime = nn.Parameter(torch.zeros(args.hidden_size))
        # self.b_prime = BiasLayer(args.hidden_size, 'zeros')

        # individual alpha for each position
        self.alphas = nn.Parameter(torch.zeros(args.max_seq_length, args.hidden_size))

        self.beta_input_seq = nn.Parameter(torch.randn(args.num_intents, args.hidden_size) *
                                           (1 / np.sqrt(args.hidden_size)))

        self.beta_label_seq = nn.Parameter(torch.randn(args.num_intents, args.hidden_size) *
                                           (1 / np.sqrt(args.hidden_size)))
        
        nn.init.normal_(self.beta_input_seq, mean=0, std=1/np.sqrt(args.hidden_size))
        nn.init.normal_(self.beta_label_seq, mean=0, std=1/np.sqrt(args.hidden_size))
        
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _intention_clustering(self,
                              z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely the primary intention at position i
        is related with kth latent category
        :param z:
        :return:
        """
        z = self.layernorm1(z)
        hidden_size = z.shape[2]
        exp_normalized_numerators = list()
        i = 0
        for prototype_k in self.prototypes:
            prototype_k = self.layernorm2(prototype_k)  # [D]
            
            # numerator = torch.matmul(z, prototype_k)  # [B, S]
            numerator = self.cos(z, prototype_k)
            exp_normalized_numerator = torch.exp(numerator / np.sqrt(hidden_size))  # [B, S]
            exp_normalized_numerators.append(exp_normalized_numerator)
            if i == 0:
                denominator = exp_normalized_numerator
            else:
                denominator = torch.add(denominator, exp_normalized_numerator)
            i = i + 1

        all_attentions_p_k_i = [torch.div(k, denominator)
                                for k in exp_normalized_numerators]  # [B, S] K times
        all_attentions_p_k_i = torch.stack(all_attentions_p_k_i, -1)  # [B, S, K]

        return all_attentions_p_k_i

    def _intention_weighting(self,
                             z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely primary intention at position i
        is important for predicting user's future intentions
        :param z:
        :return:
        """
        hidden_size = z.shape[2]
        keys_tilde_i = self.layernorm3(z + self.alphas)  # [B, S, D]
        keys_i = keys_tilde_i + torch.relu(self.w(keys_tilde_i))  # [B, S, D]
        query = self.layernorm4(self.b_prime + self.alphas[-1, :] + z[:, -1, :])  # [B, D]
        query = torch.unsqueeze(query, -1)  # [B, D, 1]
        # numerators = self.cos(keys_i, query)
        # numerators = numerators.unsqueeze(-1)
        numerators = torch.bmm(keys_i, query)  # [B, S, 1]
        exp_normalized_numerators = torch.exp(numerators / np.sqrt(hidden_size))
        sum_exp_normalized_numerators = exp_normalized_numerators.sum(1).unsqueeze(-1)  # [B, 1] to [B, 1, 1]
        all_attentions_p_i = exp_normalized_numerators / sum_exp_normalized_numerators  # [B, S, 1]
        all_attentions_p_i = all_attentions_p_i.squeeze(-1)  # [B, S]

        return all_attentions_p_i

    def _intention_aggr(self,
                        z: torch.Tensor,
                        attention_weights_p_k_i: torch.Tensor,
                        attention_weights_p_i: torch.Tensor,
                        is_input_seq: bool) -> torch.Tensor:
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param z:
        :param attention_weights_p_k_i:
        :param attention_weights_p_i:
        :param is_input_seq:
        :return:
        """
        attention_weights_p_i = attention_weights_p_i.unsqueeze(-1)  # [B, S, 1]
        attention_weights = torch.mul(attention_weights_p_k_i, attention_weights_p_i)  # [B, S, K]
        attention_weights_transpose = attention_weights.transpose(1, 2)  # [B, K, S]
        if is_input_seq:
            disentangled_encoding = self.beta_input_seq + torch.matmul(attention_weights_transpose, z)
        else:
            disentangled_encoding = self.beta_label_seq + torch.matmul(attention_weights_transpose, z)

        disentangled_encoding = self.layernorm5(disentangled_encoding)

        return disentangled_encoding  # [K, D]

    def forward(self,
                is_input_seq: bool,
                z: torch.Tensor):

        attention_weights_p_k_i = self._intention_clustering(z)  # [B, S, K]
        attention_weights_p_i = self._intention_weighting(z)  # [B, S]
        disentangled_encoding = self._intention_aggr(z,
                                                     attention_weights_p_k_i,
                                                     attention_weights_p_i,
                                                     is_input_seq)

        return disentangled_encoding


class MultiheadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout=0., add_zero_attn=False,
                batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        I = torch.eye(embed_dim)
        I3 = torch.cat([torch.eye(embed_dim), torch.eye(embed_dim), torch.eye(embed_dim)], dim=0)

        self.in_proj_weight = nn.Parameter(I3, requires_grad=False)
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.register_parameter('in_proj_bias', None)
        self.out_proj_weight = nn.Parameter(I, requires_grad=False)
        self.out_proj_bias = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
        
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None):
        # B D S -> S B D
        query, key, value = [x.permute(2, 0, 1) for x in (query, key, value)]

        attn_output, _ = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj_weight, self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=False,
            attn_mask=attn_mask)
        return attn_output.permute(1, 2, 0)

def attention_origin(query, key, value, mask, head=8):
    b, d, s = key.size()
    d_h = d // head
    
    # query, key, value. B S D
    x1 = query.reshape(b * head, d_h, s)
    x2 = key.reshape(b * head, d_h, s)
    x3 = value.reshape(b * head, d_h, s)
    # => B*H D_H S

    attn = x2.permute(0, 2, 1) @ x1 # s_k x s_q
    attn = attn / np.sqrt(d_h)
    attn = attn.masked_fill(~mask[:,None,:,None].repeat(1, head, 1, 1).reshape(b * head, s, 1), -np.inf)
    # attn = F.dropout(attn, p=self.dropout, training=self.training)
    attn = F.softmax(attn, dim=1)

    x = x3 @ attn
    x = x.reshape(b, d, s)
    return x

class AggLayer(nn.Module):
    def __init__(self, args, dim, head, dropout=0.0, norm='ln', aug_dropouts=[0.0, 0.1]):
        super().__init__()
        self.args = args
        self.dim = dim
        self.head = head
        self.dropout = dropout
        self.aug_dropouts = aug_dropouts
        self.norm = norm
        if self.norm == 'ln':
            self.ln = nn.LayerNorm(self.dim)
        elif self.norm == 'bn':
            self.bn = nn.BatchNorm1d(self.dim)
        # self.ln = nn.LayerNorm(self.dim)
        # self.attention = MultiheadAttention(dim, head, dropout=self.aug_dropouts[0])
        self.attention = SelfAttention(args)
        self.intermediate = PointWiseFeedForward(args)
    
    def forward(self, x1, x2, x3, mask, type=0):
        if type == 0:
            x = x1 + x2
            x = F.dropout(x, p=self.dropout, training=self.training)

        else:
            x = self.attention(x1, mask)
            x = self.intermediate(x)
        return x

class Encoder(nn.Module):
    def __init__(self, args, dim, head, layer, edgeops, nodeops, arch=None, dropout=0.1, context='fc', act='nn.ReLU', norm='ln', pre=True, aug_dropouts=[0.0, 0.1]):
        super().__init__()
        self.args = args
        self.arch = arch
        self.edgeops = edgeops
        self.nodeops = nodeops
        self.layer = layer if arch is None else len(arch)
        self.dim = dim
        self.head = head
        self.dropout = dropout
        self.context = context
    
        op_map = {}
        
        def O(idx):
            return OPS[PRIMITIVES[idx]](self.dim, self.dropout, act=eval(act), norm=norm, pre=pre)

        if arch is None:
            # include all edges according to contexts
            for i in range(1, layer + 1):
                for j in range(i):
                    for op in edgeops:
                        for ftype in range(2):
                            for ttype in range(2):
                                op_map[self._get_path_name(j, i, op, ftype, ttype)] = O(op)
                # agg layer
                op_map[f'layer-{i}'] = AggLayer(self.args, self.dim, self.head, self.dropout, norm, aug_dropouts=aug_dropouts)
            
        else:
            for i, a in enumerate(arch):
                o1, node, o2, o3, n = a
                cur_id = i + 1
                ftype_prev = None if node <= 0 else arch[node - 1][-1]
                ftype_i = None if i == 0 else arch[i - 1][-1]
                if n == 0:
                    op_map[self._get_path_name(i, cur_id, o1, ftype_i, 0)] = O(o1)
                    if node >= 0:
                        op_map[self._get_path_name(node, cur_id, o2, ftype_prev, 0)] = O(o2)
                else:
                    op_map[self._get_path_name(i, cur_id, o1, ftype_i, 1)] = O(o1)
                    op_map[self._get_path_name(node, cur_id, o2, ftype_prev, 2)] = O(o2)
                    op_map[self._get_path_name(node, cur_id, o3, ftype_prev, 3)] = O(o3)

                # agg layer
                op_map[f'layer-{cur_id}'] = AggLayer(self.args, self.dim, self.head, self.dropout, norm, aug_dropouts=aug_dropouts)

        self.op_map = nn.ModuleDict(op_map)

    def _get_path_name(self, fid, tid, op, ftype=None, ttype=None):
        if self.context == 'fc':
            if fid == 0:
                return f'0-{tid}-{op}-{ttype}'
            return f'{fid}-{tid}-{op}-{ftype}-{ttype}'
        elif self.context == 'tc':
            return f'{fid}-{tid}-{op}-{ttype}'
        elif self.context == 'sc':
            if fid == 0:
                return f'0-{tid}-{op}'
            return f'{fid}-{tid}-{op}-{ftype}'
        return f'{fid}-{tid}-{op}'
    
    def get_path_parameters_name(self, arch):
        key_set = set()
        for i, a in enumerate(arch):
            o1, prev, o2, o3, n = a
            f_prev = None if prev <= 0 else arch[prev-1][-1]
            f_i = None if i == 0 else arch[i - 1][-1]
            if n == 0:
                key_set.add(self._get_path_name(i, i + 1, o1, f_i, 0))
                if prev >= 0:
                    key_set.add(self._get_path_name(prev, i + 1, o2, f_prev, 0))
            else:
                key_set.add(self._get_path_name(i, i + 1, o1, f_i, 1))
                key_set.add(self._get_path_name(prev, i + 1, o2, f_prev, 2))
                key_set.add(self._get_path_name(prev, i + 1, o3, f_prev, 3))
        return key_set
    
    def get_parameters_by_name(self, names):
        param_list = []
        for name in names:
            param_list.extend(list(self.op_map[name].parameters()))
        return param_list

    def get_path_parameters(self, arch):
        param_list = []
        key_list = []
        def add_param(path):
            if path not in key_list:
                key_list.append(path)
                param_list.extend(list(self.op_map[path].parameters()))

        for i, a in enumerate(arch):
            o1, prev, o2, o3, n = a
            ftype = None if prev <= 0 else arch[prev - 1][-1]
            ftype_prev = None if prev <= 0 else arch[prev - 1][-1]
            ftype_i = None if i == 0 else arch[i - 1][-1]
            if n == 0:
                add_param(self._get_path_name(i, i + 1, o1, ftype_i, 0))
                if prev >= 0:
                    add_param(self._get_path_name(prev, i + 1, o2, ftype_prev, 0))
            elif n == 1:
                add_param(self._get_path_name(i, i + 1, o1, ftype_i, 1))
                add_param(self._get_path_name(prev, i + 1, o2, ftype_prev, 2))
                add_param(self._get_path_name(prev, i + 1, o3, ftype_prev, 3))

        return param_list

    def forward(self, x, mask, arch=None):
        
        if arch is None or self.arch is not None:
            arch = self.arch

        x_list = [x] + [torch.zeros_like(x) for _ in range(len(arch))]

        for i, a in enumerate(arch):
            cur_idx = i + 1
            o1, prev, o2, o3, n = a
            ftype_prev = None if prev <= 0 else arch[prev - 1][-1]
            ftype_i = None if i == 0 else arch[i - 1][-1]
            if n == 0:
                inp1 = x_list[i]
                # inp1 = F.layer_norm(inp1, self.dim)
                # inp1 = F.dropout(inp1, p=self.dropout, training=self.training)
                feat1 = self.op_map[self._get_path_name(i, i + 1, o1, ftype_i, 0)](inp1, mask=mask)
                if prev >= 0:
                    inp2 = x_list[prev]
                    # inp2 = F.layer_norm(inp2, self.dim)
                    # inp2 = F.dropout(inp2, p=self.dropout, training=self.training)
                    feat2 = self.op_map[self._get_path_name(prev, i + 1, o2, ftype_prev, 0)](inp2, mask=mask)
                else:
                    feat2 = 0
                feat3 = 0
            
            else:
                inp1 = x_list[i]
                # inp1 = F.layer_norm(inp1, self.dim)
                # inp1 = F.dropout(inp1, p=self.dropout, training=self.training)
                inp2 = x_list[prev]
                # inp2 = F.layer_norm(inp2, self.dim)
                # inp2 = F.dropout(inp2, p=self.dropout, training=self.training)
                inp3 = x_list[prev]
                # inp3 = F.layer_norm(inp3, self.dim)
                # inp3 = F.dropout(inp3, p=self.dropout, training=self.training)
                feat1 = self.op_map[self._get_path_name(i, i + 1, o1, ftype_i, 1)](inp1, mask=mask)
                feat2 = self.op_map[self._get_path_name(prev, i + 1, o2, ftype_prev, 1)](inp2, mask=mask)
                feat3 = self.op_map[self._get_path_name(prev, i + 1, o3, ftype_prev, 1)](inp3, mask=mask)
            
            x = self.op_map[f'layer-{cur_idx}'](feat1, feat2, feat3, mask, n)

            x_list[cur_idx] = x

        out = x_list[-1]
        # out = F.layer_norm(out, self.dim)
        # out = F.dropout(out, p=self.dropout, training=self.training)
        return out
