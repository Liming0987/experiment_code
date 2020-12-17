import torch
from torch import nn
from collections import defaultdict
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os

class STAMP(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, attention_size):
        super(STAMP, self).__init__()
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        self.attention_size = attention_size

        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.neg_emb = 1
        if self.neg_emb == 1:
            self.item_emb_neg = torch.nn.Embedding(num_items, embedding_size)

        # self.user_emb = nn.Embedding(num_users, embedding_size)
        self.attention_wxi = torch.nn.Linear(embedding_size, self.attention_size, bias=True)
        self.attention_wxt = torch.nn.Linear(embedding_size, self.attention_size, bias=True)
        self.attention_wms = torch.nn.Linear(embedding_size, self.attention_size, bias=True)
        self.attention_out = torch.nn.Linear(self.attention_size, 1, bias=False)

        self.mlp_a = torch.nn.Linear(embedding_size, embedding_size, bias=True)
        self.mlp_b = torch.nn.Linear(embedding_size, embedding_size, bias=True)

    def init_paras(self, m):
        if 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)

    def forward(self, feed_dict):
        y = feed_dict['y']

        prediction = self.predict(feed_dict)
        # y = torch.cat([y, torch.from_numpy(np.array([0.0] * len(y), dtype=float)).type(torch.FloatTensor)])
        # loss = torch.nn.BCELoss(reduction='mean')(prediction.view([-1, 1]), y.view([-1, 1]))

        loss = self.rank_loss(prediction, y, feed_dict['real_batch_size'])

        feed_dict['prediction'] = prediction
        feed_dict['loss'] = loss
        return feed_dict

    def predict(self, feed_dict):
        iids = feed_dict['iid']
        history = feed_dict['history_mix']
        history_lengths = feed_dict['history_mix_length']

        valid_his = history.abs().gt(0).long() # B * H
        his_pos_neg = history.ge(0).unsqueeze(-1).float() # B * H * 1
        his_length = valid_his.sum(dim=-1) # B

        # embedding layer 之后的结果
        # batch_size, sequence, item_embedding
        # [[[1,1,2,3,4,..], [], ...], []]
        pos_his_vectors = self.item_emb(history.abs() * valid_his) * valid_his.unsqueeze(dim=-1).float()

        if self.neg_emb == 1:
            neg_his_vectors = self.item_emb_neg(history.abs() * valid_his) * valid_his.unsqueeze(dim=-1).float()
            his_vectors = pos_his_vectors * his_pos_neg + (1 - his_pos_neg) * neg_his_vectors

        # Prepare Vectors
        # Xt 是序列的最后一个元素
        xt = his_vectors[range(len(history_lengths)), his_length - 1, :] # B * V
        # Ms 是 session representation
        ms = his_vectors.sum(dim=1)/his_length.view([-1, 1]).float()

        # Attention
        att_wxi_v = self.attention_wxi(his_vectors) # B * H * V
        att_wxt_v = self.attention_wxt(xt).unsqueeze(dim=1) # B * 1 * V
        att_wms_v = self.attention_wms(ms).unsqueeze(dim=1) # B * 1 * V
        att_v = self.attention_out((att_wxi_v + att_wxt_v + att_wms_v).sigmoid()) # B * H * 1
        ma = (his_vectors * att_v * valid_his.unsqueeze(dim=-1).float()).sum(dim=1) # B * V

        # Output Layer
        hs = self.mlp_a(ma).tanh()
        ht = self.mlp_b(xt).tanh()

        target_emb = self.item_emb(iids)

        pred_vector = hs * ht
        prediction = (pred_vector * target_emb).sum(dim=-1).view([-1])

        return prediction

    def rank_loss(self, prediction, label, real_batch_size):
        """
        计算 Bayesian Personalized Ranking Loss

        :param prediction: size: (total_batch_size, )
        :param label: size: (real_batch_size, )
        :param real_batch_size:
        :return: scalar (loss value)
        """
        pos_neg_tag = (label - 0.5) * 2
        observed, sample = prediction[:real_batch_size],prediction[real_batch_size:]
        # sample的第一维表示有多少个 neg samples
        sample = sample.view([-1, real_batch_size])
        sample_softmax = (sample * pos_neg_tag.view([1, real_batch_size])).softmax(dim=0)
        sample = (sample * sample_softmax).sum(dim=0) # 取 neg samples的加权平均
        loss = -(pos_neg_tag * (observed - sample)).sigmoid().log().sum()

        return loss

    def cross_entropy(self, prediction, y):
        y = torch.cat([y, torch.from_numpy(np.array([0.0] * len(y), dtype=float)).type(torch.FloatTensor)])
        loss = torch.nn.BCELoss(reduction='mean')(prediction, y)
        return loss

    def evaluate_method(self, predictions, data, metrics):
        y = data['y']
        evaluations = []
        rank = False
        for metric in metrics:
            if '@' in metric:
                rank = True

        split_y, split_p, split_y_sum = None, None, None

        if rank:
            uids, times = data['uid'].reshape([-1]), data['time'].reshape([-1])
            sorted_idx = np.lexsort((-y, -predictions, times, uids))
            sorted_uid, sorted_time = uids[sorted_idx], times[sorted_idx]

            sorted_uid_unique, sorted_uid_idx = np.unique([sorted_uid, sorted_time], axis=1,return_index=True)

            sorted_y, sorted_p = y[sorted_idx], predictions[sorted_idx]
            split_y, split_p = np.split(sorted_y, sorted_uid_idx[1:]), np.split(sorted_p, sorted_uid_idx[1:])

            split_y_sum = [np.sum((d > 0).astype(float)) for d in split_y]

        for metric in metrics:
            try:
                if metric == 'rmse':
                    evaluations.append(np.sqrt(mean_squared_error(y, predictions)))
                elif metric == 'mae':
                    evaluations.append(mean_absolute_error(y, predictions))
                elif metric == 'precision':
                    evaluations.append(precision_score(y, np.around(predictions)))
                elif metric == 'recall':
                    evaluations.append(recall_score(y, np.around(predictions)))
                elif metric == 'accuracy':
                    evaluations.append(accuracy_score(y, np.around(predictions)))
                else:
                    k = int(metric.split('@')[-1])
                    if metric.startswith('hit@'):
                        k_data = [(list(d) + [0] * k)[:k] for d in split_y]
                        hits = (np.sum(k_data, axis=1) > 0).astype(float)
                        evaluations.append(np.average(hits))
                    elif metric.startswith('ndcg@'):
                        max_k = max([len(d) for d in split_y])
                        k_data = np.array([(list(d) + [0] * max_k)[:max_k] for d in split_y])
                        best_rank = -np.sort(-k_data, axis=1)
                        best_dcg = np.sum(best_rank[:, :k]/np.log2(np.arange(2, k+2)), axis=1)

                        dcg = np.sum(k_data[:, :k] / np.log2(np.arange(2, k + 2)), axis=1)
                        ndcgs = dcg / best_dcg
                        evaluations.append(np.average(ndcgs))
                    elif metric.startswith('precision@'): #还没有完全理解
                        k_data = [d[:k] for d in split_y]
                        k_data_dict = defaultdict(list)
                        for d in k_data:
                            k_data_dict[len(d)].append(d)
                        precisions = [np.average((np.array(d) > 0).astype(float), axis=1) for d in k_data_dict.values()]
                        evaluations.append(np.average(np.concatenate(precisions)))
            except Exception as e:
                raise e

        return evaluations

    def save_model(self, model_path):
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        print('save model to ' + model_path)

    def load_model(self, model_path, cpu=False):
        if cpu:
            self.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(model_path))
        self.eval()

        print('load model from ' + model_path)


