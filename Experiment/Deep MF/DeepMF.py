import torch
from torch import nn
from collections import defaultdict
from sklearn.metrics import *
import numpy as np

class DeepMF(nn.Module):
    def __init__(self, data_processor,num_users, num_items, embedding_size):
        super(DeepMF, self).__init__()

        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        train_data = data_processor.get_train_data(-1)
        self.user_emb = np.zeros([num_users, num_items], dtype=np.float)
        uids = train_data['uid']
        iids = train_data['iid']
        y = train_data['y']
        for uid, iid, y in list(zip(uids, iids, y)):
            if y == 1:
                self.user_emb[uid][iid] = 1

        self.item_emb = np.transpose(self.user_emb)

        self.user_emb = torch.tensor(np.array(self.user_emb), dtype=torch.float)
        self.user_emb = nn.Embedding.from_pretrained(self.user_emb)

        self.item_emb = torch.tensor(np.array(self.item_emb), dtype=torch.float)
        self.item_emb = nn.Embedding.from_pretrained(self.item_emb)

        print('num_items: ' + str(num_items))
        self.user_layer_1 = torch.nn.Linear(num_items, embedding_size)
        self.user_layer_2 = torch.nn.Linear(embedding_size, embedding_size)

        print('num_users: ' + str(num_users))
        self.item_layer_1 = torch.nn.Linear(num_users, embedding_size)
        self.item_layer_2 = torch.nn.Linear(embedding_size, embedding_size)

        self.similarity_layer = torch.nn.CosineSimilarity()


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


    def init_paras(self, m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, feed_dict):
        y = feed_dict['y']
        prediction = self.predict(feed_dict)
        y = torch.cat([y, torch.from_numpy(np.array([0.0] * len(y), dtype=float)).type(torch.FloatTensor)])
        loss = torch.nn.BCELoss(reduction='mean')(prediction, y)

        feed_dict['prediction'] = prediction
        feed_dict['loss'] = loss
        return feed_dict

    def predict(self, feed_dict):
        uids = feed_dict['uid']
        iids = feed_dict['iid']
        u = self.user_emb(uids)
        v = self.item_emb(iids)

        u = self.user_layer_1(u)
        u = self.user_layer_2(u)

        v = self.item_layer_1(v)
        v = self.item_layer_2(v)

        similarity = self.similarity_layer(u, v)
        similarity = torch.clamp(similarity, min=1e-8)

        return similarity

