import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric, sample_valid_archs
from ops import PRIMITIVES
from llm import MyLLM


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        if self.cuda_condition:
            self.model.to(self.device)

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False, arch=None):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False, arch=arch)

    def test(self, epoch, full_sort=False, arch=None):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False, arch=arch)

    def iteration(self, epoch, dataloader, full_sort=False, train=True, arch=None):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        # pred_list = pred_list[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):

        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        seq_emb = seq_out
        pos_emb = pos_emb.transpose(1, 2)
        neg_emb = neg_emb.transpose(1, 2)

        istarget = (pos_ids > 0).float()  # [batch*seq_len]
        neg_istarget = istarget.view(istarget.size(0), istarget.size(1), 1).repeat(1, 1, self.args.max_seq_length)
        
        pos_logits = torch.diagonal(torch.bmm(seq_emb, pos_emb), dim1=1, dim2=2)
        neg_logits = torch.bmm(seq_emb, neg_emb)
        
        loss = torch.sum(torch.mul(-torch.log(torch.sigmoid(pos_logits) + 1e-24), istarget)) + \
            torch.sum(torch.mul(-torch.log(1 - torch.sigmoid(neg_logits) + 1e-24), neg_istarget))
        loss /= torch.sum(istarget)

        return loss

    def dss_cross_entropy(self, seq_out, pos_ids, neg_ids):
        """
        Cross-entropy loss using DSS 2.0 model
        :param seq_out:
        :param pos_ids:
        :param neg_ids:
        :return:
        """
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids[:,-1])
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        seq_emb = seq_out
        pos_emb = pos_emb.unsqueeze(2)
        neg_emb = neg_emb.transpose(1, 2)

        pos_logits = torch.max(torch.bmm(seq_emb, pos_emb).squeeze(2), dim=-1, keepdim=True)[0]
        neg_logits = torch.max(torch.bmm(seq_emb, neg_emb).transpose(1, 2), dim=-1)[0]
        
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24)) + \
            torch.sum(-torch.log(1 - torch.sigmoid(neg_logits) + 1e-24))
        loss /= pos_ids.size(0)
        
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample.cuda())
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def dss_predict_sample(self, seq_out, test_neg_sample):
        """
        Predit using DSS 2.0 model
        :param seq_out:
        :param test_neg_sample:
        :return:
        """
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        mul = torch.matmul(test_item_emb, seq_out.transpose(1, 2))  # [B 100 K]
        test_logits = torch.max(mul, dim=-1)[0]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # seq_out = F.normalize(seq_out, dim=1, p=2)
        test_item_emb = test_item_emb.transpose(0, 1)
        # test_item_emb = F.normalize(test_item_emb, dim=1, p=2)
        rating_pred = torch.matmul(seq_out, test_item_emb)
        return rating_pred

    def dss_predict_full(self, seq_out):
        """
        Predict using DSS 2.0 model
        :param seq_out:
        :return:
        """
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch num_intents hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        # [batch hidden_size]
        rating_pred = torch.max(rating_pred, dim=1)[0]
        return rating_pred


class DSSPretrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(DSSPretrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader, edgeop, nodeop):
        desc = f'S2I-{self.args.s2i_weight}-' \
               f'S2S-{self.args.s2s_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} "
                                            f"Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        seq2item_loss_avg = 0.0
        seq2seq_loss_avg = 0.0
        total_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            inp_pos_items, label_pos_items, next_pos_item = batch

            for arch in sample_valid_archs(self.args.auto_layer, edgeop, nodeop, self.args.auto_arch_batch_size, PRIMITIVES):
                # inp_pos_items[:,:-5]=0
                seq2item_loss, seq2seq_loss = self.model.pretrain(inp_pos_items,
                                                                label_pos_items,
                                                                next_pos_item, arch)

                joint_loss = self.args.s2i_weight * seq2item_loss + \
                            self.args.s2s_weight * seq2seq_loss

                self.optim.zero_grad()
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50.0)
                self.optim.step()
                seq2item_loss_avg += seq2item_loss.item()
                seq2seq_loss_avg += seq2seq_loss.item()
                total_loss_avg += joint_loss.item()

            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Gradient {name} contains NaN values!")
                    exit(1)

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "seq2item_loss_avg": '{:.4f}'.format(seq2item_loss_avg / num),
            "seq2seq_loss_avg": '{:.4f}'.format(seq2seq_loss_avg / num),
            "dss_loss_avg": '{:.4f}'.format(total_loss_avg / num)
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')


class FineTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FineTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.llm = MyLLM()
        self.features = np.load('data/%s-feature.npy' % (args.data_name), allow_pickle=True).item()
        self.features = np.array([self.features[key] for key in self.features.keys()])

    def iteration(self, epoch, dataloader, full_sort=False, train=True, arch=None):

        str_code = "train" if train else "test"
        import sys
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}",
                                  file=sys.stdout)
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, answer = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids, arch)
                # sequence_output = sequence_output[:, -1, :]
                if self.args.loss_type in ['DSS-2']:
                    loss = self.dss_cross_entropy(sequence_output, target_pos, target_neg)
                else:
                    loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            # if (epoch + 1) % self.args.log_freq == 0:
            print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)
                    if self.args.loss_type in ['DSS-2']:
                        rating_pred = self.dss_predict_full(recommend_output)
                    else:
                        recommend_output = recommend_output[:, -1, :]
                        rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                input_ids_list, pred_list, test_items_list = [], [], []
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    if i == 0:
                        input_ids_list = input_ids.cpu().detach().numpy().copy()
                    else:
                        input_ids_list = np.append(input_ids_list, input_ids.cpu().detach().numpy().copy(), axis=0)
                    sample_negs = []
                    for _ in range(100):
                        t = np.random.randint(1, self.args.item_size)
                        while t in answers: t = np.random.randint(1, self.args.item_size)
                        sample_negs.append(t)
                    sample_negs = torch.tensor(sample_negs).repeat(answers.size(0), 1).to(self.device)
                    
                    # input_ids[:,:-5] = 0
                    recommend_output = self.model.finetune(input_ids, arch)
                    test_neg_items = torch.cat((answers, sample_negs), -1)

                    if self.args.loss_type in ['DSS-2']:
                        test_logits = self.dss_predict_sample(recommend_output, test_neg_items)
                    else:
                        recommend_output = recommend_output[:, -1, :]
                        test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                        test_items_list = test_neg_items.cpu().detach().numpy()
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                        test_items_list = np.append(test_items_list, test_neg_items.cpu().detach().numpy(), axis=0)
                rk_list = (-pred_list).argsort()
                pred_list = rk_list.argsort()[:,0]

                return self.get_sample_scores(epoch, pred_list)
