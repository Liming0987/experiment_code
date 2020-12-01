import torch
from torch import nn
from collections import defaultdict
from sklearn.metrics import *
import numpy as np

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)

    def evaluate_method(self, predictions, data, metrics):
        y = data['y']
        evaluations = []
        rank = False
        for metric in metrics:
            if '@' in metric:
                rank = True

        split_y, split_p = None, None

        if rank:
            uids = data['uid'].reshape([-1])
            sorted_idx = np.lexsort((-y, -predictions,uids))
            sorted_uid = uids[sorted_idx]
            sorted_uid_unique, sorted_uid_idx = np.unique(sorted_uid, return_index=True)
            sorted_y, sorted_p = y[sorted_idx], predictions[sorted_idx]
            split_y, split_p = np.split(sorted_y, sorted_uid_idx[1:]), np.split(sorted_p, sorted_uid_idx[1:])

        for metric in metrics:
            try:
                if metric == 'precision':
                    evaluations.append(precision_score(y, np.around(predictions)))
                elif metric == 'recall':
                    evaluations.append(recall_score(y, np.around(predictions)))
                elif metric == 'accuracy':
                    evaluations.append(accuracy_score(y, np.around(predictions)))
                else:
                    k = int(metric.split('@')[-1])
                    if metric.startswith('precision@'):
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
        if 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, feed_dict):
        y = feed_dict['y']
        prediction = self.predict(feed_dict)

        loss = torch.nn.MSELoss(reduction='mean')(prediction, y)

        feed_dict['prediction'] = prediction
        feed_dict['loss'] = loss
        return feed_dict

    def predict(self, feed_dict):
        uids = feed_dict['uid']
        iids = feed_dict['iid']
        u = self.user_emb(uids)
        v = self.item_emb(iids)
        # u * v element-wise multiplication
        return (u * v).sum(1)
