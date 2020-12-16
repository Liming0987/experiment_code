import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import *
import os
import numpy as np
from collections import defaultdict


class BaseModel(torch.nn.Module):

    def __init__(self, total_item_num, total_user_num, vector_size, layers, r_weight=0.1):
        super(BaseModel, self).__init__()
        self.optimizer = None
        self.r_weight = r_weight
        self.sim_scale = 10
        self.layers = layers
        self.vector_size = vector_size
        self.item_embeddings = torch.nn.Embedding(total_item_num, vector_size)
        self.user_embeddings = torch.nn.Embedding(total_user_num, vector_size)

        self.empty = torch.nn.Parameter(self.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.vector_size]).astype(np.float32)), requires_grad=False)

        self.intersection_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'intersection_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.difference_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'difference_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.union_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'union_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.sim_layer = torch.nn.Linear(vector_size * 2, 1)
        for i in range(layers):
            setattr(self, 'sim_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

    def numpy_to_torch(self, d, gpu=True, requires_grad=True):
        t = torch.from_numpy(d)
        if d.dtype is np.float:
            t.requires_grad = requires_grad
        if gpu:
            if torch.cuda.device_count() > 0:
                t = t.cuda()
        return t

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


    def init_paras(self, m):
        # print(str(type(m)))
        if 'Linear' in str(type(m)): #???
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def uniform_size(self, vector1, vector2):
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        else:
            vector2 = vector2.expand_as(vector1)
        return vector1, vector2

    def intersection(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2)
        vector = torch.cat((vector1, vector2), dim=-1)
        # print('intersection vector: ' + str(vector.size()))

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'intersection_layer_%d' % i)(vector))
        vector = self.intersection_layer(vector)

        return vector

    def union(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2)
        vector = torch.cat((vector1, vector2), dim=-1)

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'union_layer_%d' % i)(vector))
        vector = self.union_layer(vector)

        return vector

    def difference(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2)
        vector = torch.cat((vector1, vector2), dim=-1)

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'difference_layer_%d' % i)(vector))
        vector = self.difference_layer(vector)

        return vector

    def similarity_MLP(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'sim_layer_%d' % i)(vector))
        sim = self.sim_layer(vector)
        return sim

    def similarity_COSINE(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1) #result 的 范围是在 【-1， 1】
        result = result * self.sim_scale #self.sim_scal = 10 => result 的 范围是在 【-10， 10】

        if sigmoid:
            result = result.sigmoid()
        return result

    def linear_attention(self, user_vector, items_vector, x_sample_num, batch_size):
        # user_vector: [batch_size, embedding_size]
        # user_l1 = torch.norm(user_vector, 1, -1) # [batch]
        # user_vector = user_vector / user_l1.view(-1, 1)
        #
        # # items_vector: [batch_size, x_sample_num * 2,embedding_size]
        # items_l1 = torch.norm(items_vector, 1, -1)
        # item_vector = items_vector / items_l1.view(batch_size, x_sample_num, 1)

        # linear_sim: [batch_size, x_sample_num * 2]
        linear_sim = F.cosine_similarity(user_vector.view(batch_size, 1, self.vector_size),
                            items_vector, dim=-1) + 1
        # print('linear_sim size: ' + str(linear_sim.size()))

        linear_sim = linear_sim / linear_sim.sum(dim=-1).view([batch_size, 1])
        linear_sim = linear_sim.view([batch_size, -1, 1])

        # print('linear_sim size: ' + str(linear_sim.size()))

        # print('item_vector: ' + str(items_vector.size()))

        # item_vector: [batch_size, embedding_size]
        item_vector = (items_vector * linear_sim).sum(dim=1)

        # print('item_vector: ' + str(item_vector.size()))

        return item_vector

    def regularizer(self, out_dict):
        # r_length_weight = 0.001

        constraint = out_dict['constraint']
        constraint_valid = out_dict['constraint_valid']

        # intersecction
        r_intersection_self = 1 - self.similarity_COSINE(self.intersection(constraint, constraint), constraint)
        r_intersection_self = (r_intersection_self * constraint_valid).sum()

        r_intersection_empty = 1 - self.similarity_COSINE(self.intersection(constraint, self.empty), constraint)
        r_intersection_empty = (r_intersection_empty * constraint_valid).sum()

        # union
        r_union_self = 1 - self.similarity_COSINE(self.union(constraint, constraint), constraint)
        r_union_self = (r_union_self * constraint_valid).sum()

        r_union_empty = 1 - self.similarity_COSINE(self.union(constraint, self.empty), constraint)
        r_union_empty = (r_union_empty * constraint_valid).sum()

        # difference
        r_difference_empty = 1 - self.similarity_COSINE(self.difference(constraint, self.empty), constraint)
        r_difference_empty = (r_difference_empty * constraint_valid).sum()

        r_loss = 0
        r_loss += r_intersection_self + r_intersection_empty + r_union_self + r_union_empty + r_difference_empty

        if self.r_weight > 0:
            r_loss = r_loss * self.r_weight
        else:
            r_loss = self.numpy_to_torch(np.array(0.0, dtype=np.float32))

        # r_loss += r_length * r_length_weight

        out_dict['r_loss'] = r_loss

        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        loss = self.rank_loss(out_dict['prediction'], feed_dict['y'], feed_dict['real_batch_size'])

        # 重新写
        out_dict = self.regularizer(out_dict)

        # prediction, label = out_dict['prediction'], feed_dict['y']
        # print('prediction: ' + str(prediction.size()))
        r_loss = out_dict['r_loss']

        # loss = torch.nn.BCELoss(reduction='mean')(prediction.view([-1, 1]), label.view([-1, 1]))

        out_dict['loss'] = loss + r_loss

        return out_dict

    def predict(self, feed_dict):
        real_batch_size = feed_dict['real_batch_size']
        total_batch_size = feed_dict['total_batch_size']
        train = feed_dict['train']
        pos_history = feed_dict['history']
        neg_history = feed_dict['history_neg']
        pos_history_length = feed_dict['history_length']
        neg_history_length = feed_dict['history_neg_length']

        pos_his_valid = pos_history.abs().gt(0).float() # B * H
        # print(pos_his_valid.size())
        neg_his_valid = neg_history.abs().gt(0).float() # B * H

        pos_elements = self.item_embeddings(pos_history) # B * H * V
        constraint = [pos_elements.view([total_batch_size, -1, self.vector_size])] # B * H * V
        constraint_valid = [pos_his_valid.view([total_batch_size, -1])] # B * H
        # 如果 pos_his_valid 对应的是 0，则 pos_elements 对应的 vector 全部变成 0
        pos_elements = pos_elements * pos_his_valid.unsqueeze(-1) # pos_his_valid.unsqueeze(-1) => (B * H * 1)

        neg_elements = self.item_embeddings(neg_history)
        constraint.append(neg_elements.view([total_batch_size, -1, self.vector_size]))
        constraint_valid.append(neg_his_valid.view([total_batch_size, -1]))
        neg_elements = neg_elements * neg_his_valid.unsqueeze(-1)

        tmp_o = None
        for i in range(max(pos_history_length)):
            tmp_o_valid = pos_his_valid[:, i].unsqueeze(-1)
            if tmp_o is None:
                tmp_o = pos_elements[:, i, :] * tmp_o_valid
            else:
                tmp_o = self.union(tmp_o, pos_elements[:, i, :]) * tmp_o_valid + tmp_o * (1 - tmp_o_valid)
                constraint.append(tmp_o.view([total_batch_size, 1, self.vector_size])) # B * 1 * V
                constraint_valid.append(tmp_o_valid)

        for i in range(max(neg_history_length)):
            tmp_o_valid = neg_his_valid[:, i].unsqueeze(-1)
            tmp_o = self.difference(tmp_o, neg_elements[:, i, :]) * tmp_o_valid + tmp_o * (1 - tmp_o_valid)
            constraint.append(tmp_o.view([total_batch_size, 1, self.vector_size]))
            constraint_valid.append(tmp_o_valid)

        his_vector = tmp_o # B * V

        all_valid = feed_dict['iid'].gt(0).float().view([total_batch_size, 1])

        right_vector = self.item_embeddings(feed_dict['iid'])
        constraint.append(right_vector.view([total_batch_size, 1, self.vector_size])) # B * 1 * V
        constraint_valid.append(all_valid)

        prediction = self.similarity_MLP(his_vector, right_vector).view([-1])

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)
        out_dict = {
            'prediction': prediction,
            'constraint': constraint,
            'constraint_valid': constraint_valid,
        }

        return out_dict

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











