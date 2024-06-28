import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DSSEncoder, SASEncoder, LayerNorm, Encoder

torch.set_printoptions(profile="full")


class SASRecModel(nn.Module):
    """
    Class implementing SASRec model
    """

    def __init__(self, args, dim, head, layer, edgeops=None, nodeops=None, arch=None, dropout=0.0, context='fc', act='nn.ReLU', norm='ln', pre=True, freeze=True, pad_idx=0, aug_dropouts=[0.0, 0.1]):
        super().__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        if args.use_auto:
            self.item_encoder = Encoder(args, dim, head, layer, edgeops, nodeops, arch=arch, dropout=dropout, context=context, act=act, norm=norm, pre=pre, aug_dropouts=aug_dropouts)
        else:
            self.item_encoder = SASEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass

    def _add_position_embedding(self,
                                sequence: torch.Tensor):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def _get_embedding_and_mask(self, input_ids):
        sequence_emb = self._add_position_embedding(input_ids)
        attention_mask = (input_ids > 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = extended_attention_mask.squeeze(1)
        return sequence_emb, extended_attention_mask

    def finetune(self, input_ids, arch=None):
        sequence_emb, extended_attention_mask = self._get_embedding_and_mask(input_ids)
        if self.args.use_auto:
            item_encoded_layer = self.item_encoder(sequence_emb,
                                                   extended_attention_mask,
                                                   arch)
        else:
            item_encoded_layer = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=False)
        return item_encoded_layer


class DSSRecModel(SASRecModel):
    """
    Version 1 of Disentangled Self-Supervision
    """

    def __init__(self, args, dim, head, layer, edgeops=None, nodeops=None, arch=None, dropout=0.0, context='fc', act='nn.ReLU', norm='ln', pre=True, freeze=True, pad_idx=0, aug_dropouts=[0.0, 0.1]):
        super().__init__(args,  dim, head, layer, edgeops, nodeops, arch, dropout, context, act, norm, pre, freeze, pad_idx, aug_dropouts)
        self.disentangled_encoder = DSSEncoder(args)
        
    def scale(self, x, max_x, maxx):
        new_x = (x / max_x) * maxx
        return new_x

    def __seq2seqloss(self,
                      inp_subseq_encodings: torch.Tensor,
                      label_subseq_encodings: torch.Tensor):
        cur_batch_size = inp_subseq_encodings.size(0)
        sqrt_hidden_size = np.sqrt(self.args.hidden_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)  # [B, K, D]

        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size  # [B, K]
        normalized_dot_product = torch.flatten(normalized_dot_product)

        all_product = torch.matmul(inp_subseq_encodings.view(-1, self.args.hidden_size), label_subseq_encodings.view(-1, self.args.hidden_size).transpose(0, 1)) / sqrt_hidden_size
        all_product = torch.logsumexp(all_product, dim=1)

        seq2seq_loss_k = -normalized_dot_product + all_product
        thresh_th = int(np.floor(self.args.lambda_ * cur_batch_size * self.args.num_intents))
        thresh = torch.kthvalue(seq2seq_loss_k, thresh_th)[0] if thresh_th > 0 else np.inf
        conf_indicator = seq2seq_loss_k <= thresh
        
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, conf_indicator)

        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)

        return seq2seq_loss
    
    # def checknan(self, x, name):
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print('checknan', name, 'nan' if torch.isnan(x).any() else 'inf')
    #         print(x[0])
    #         # exit(1)

    # re-checked the math behind the code, it is CORRECT
    def __seq2itemloss(self,
                       inp_subseq_encodings: torch.Tensor,
                       next_item_emb: torch.Tensor):
        sqrt_hidden_size = np.sqrt(self.args.hidden_size)
        max_dot_product = torch.bmm(inp_subseq_encodings, next_item_emb.transpose(1, 2))
        max_dot_product = torch.max(max_dot_product.squeeze(2), dim=1)[0] / sqrt_hidden_size

        next_item_emb = next_item_emb.squeeze(1).transpose(1, 0)
        dot_product = torch.matmul(inp_subseq_encodings, next_item_emb) / sqrt_hidden_size  # [B, K, 1]
        denominator = torch.logsumexp(torch.flatten(dot_product, start_dim=1), dim=1)
        seq2item_loss_k = -max_dot_product + denominator

        seq2item_loss = torch.sum(seq2item_loss_k)

        return seq2item_loss

    def pretrain(self,
                 inp_subseq: torch.Tensor,
                 label_subseq: torch.Tensor,
                 next_item: torch.Tensor, arch):
        
        next_item_emb = self.item_embeddings(next_item)  # [B, 1, D]

        inp_subseq_emb, inp_subseq_ext_attn_mask = self._get_embedding_and_mask(inp_subseq)

        if self.args.use_auto:
            input_subseq_encoding = self.item_encoder(inp_subseq_emb,
                                                    inp_subseq_ext_attn_mask,
                                                    arch)
        else:
            input_subseq_encoding = self.item_encoder(inp_subseq_emb,
                                                    inp_subseq_ext_attn_mask,
                                                    output_all_encoded_layers=False)

        label_subseq_emb, label_subseq_ext_attn_mask = self._get_embedding_and_mask(label_subseq)
        
        if self.args.use_auto:
            label_subseq_encoding = self.item_encoder(label_subseq_emb,
                                                    label_subseq_ext_attn_mask,
                                                    arch)
        else:
            label_subseq_encoding = self.item_encoder(label_subseq_emb,
                                                  label_subseq_ext_attn_mask,
                                                  output_all_encoded_layers=False)

        disent_inp_subseq_encodings = self.disentangled_encoder(True,
                                                                input_subseq_encoding)
        disent_label_seq_encodings = self.disentangled_encoder(False,
                                                               label_subseq_encoding)

        # seq2item loss
        seq2item_loss = self.__seq2itemloss(disent_inp_subseq_encodings, next_item_emb)
        # seq2seq loss
        seq2seq_loss = self.__seq2seqloss(disent_inp_subseq_encodings, disent_label_seq_encodings)
        # seq2seq_loss = torch.tensor(0)

        
        return seq2item_loss, seq2seq_loss


class DSSRecModel2(DSSRecModel):
    """
    Version 2 of Disentangled Self-Supervision
    Here the K (number of intents) encodings collectively represent
    a sequence
    """

    def __init__(self, args,  dim, head, layer, edgeops=None, nodeops=None, arch=None, dropout=0.0, context='fc', act='nn.ReLU', norm='ln', pre=True, freeze=True, pad_idx=0, aug_dropouts=[0.0, 0.1]):
        super().__init__(args,  dim, head, layer, edgeops, nodeops, arch, dropout, context, act, norm, pre, freeze, pad_idx, aug_dropouts)

    def finetune(self, input_ids, arch=None):
        sequence_emb, extended_attention_mask = self._get_embedding_and_mask(input_ids)

        if self.args.use_auto:
            item_encoded_layer = self.item_encoder(sequence_emb,
                                                   extended_attention_mask,
                                                   arch)
        else:
            item_encoded_layer = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=False)
        sequence_encodings = self.disentangled_encoder(True,
                                                       item_encoded_layer)

        return sequence_encodings
