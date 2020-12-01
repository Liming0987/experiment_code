import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import *
import os
import numpy as np
from collections import defaultdict


class BaseModel(torch.nn.Module):

    def __init__(self, total_item_num, total_user_num, vector_size, layers):
        super(BaseModel, self).__init__()
        self.optimizer = None
        self.sim_scale = 10
        self.layers = layers
        self.vector_size = vector_size

        torch.manual_seed(10)
        torch.cuda.manual_seed(10)

        self.item_embeddings = torch.nn.Embedding(total_item_num, vector_size)

        self.user_embeddings = torch.nn.Embedding(total_user_num, vector_size)

        self.intersection_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'intersection_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.difference_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'difference_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.union_layer = torch.nn.Linear(vector_size * 2, vector_size)
        for i in range(layers):
            setattr(self, 'union_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

        self.predict_layer = torch.nn.Linear(vector_size * 2, 1)
        for i in range(layers):
            setattr(self, 'predict_layer_%d' % i, torch.nn.Linear(vector_size * 2, vector_size * 2))

    def evaluate_method(self, predictions, data, metrics):
        y = data['y']
        print('y average: ' + str(y.sum()/len(y)))
        evaluations = []
        rank = False
        for metric in metrics:
            if '@' in metric:
                rank = True

        split_y, split_p = None, None

        if rank:
            #uids = data['uid'].reshape([-1])
            uids = data['uid']
            uids = np.array(uids)
            #print('uid shape: ' + str(uids.shape))
            #print('y shape: ' + str(y.shape))
            y = np.array(y)
            
            predictions = predictions.reshape([-1])
            #print('prediction shape: ' + str(predictions.shape))

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
        # print(str(type(m)))
        if 'Linear' in str(type(m)): #???
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def intersection(self, vector1, vector2):
        vector = torch.cat((vector1, vector2), dim=-1) #[batch_size, 5, 128]
        # print('intersection vector: ' + str(vector.size()))

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'intersection_layer_%d' % i)(vector))
        vector = self.intersection_layer(vector)

        return vector

    def union(self, vector1, vector2):
        vector = torch.cat((vector1, vector2), dim=-1)

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'union_layer_%d' % i)(vector))
        vector = self.union_layer(vector)

        return vector

    def difference(self, vector1, vector2):
        vector = torch.cat((vector1, vector2), dim=-1)

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'difference_layer_%d' % i)(vector))
        vector = self.difference_layer(vector)

        return vector

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    def predict_result(self, vector1, vector2, sigmoid=True):
        vector = torch.cat((vector1, vector2), dim=-1)
        # print(vector.size())

        for i in range(self.layers):
            vector = F.relu(getattr(self, 'predict_layer_%d' % i)(vector))

        prediction = self.predict_layer(vector)

        if sigmoid:
            return prediction.sigmoid()

        return prediction

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

        r_weight = 0.1
        # r_length_weight = 0.001

        constraint = out_dict['constraint']
        # print('constraint tensor 0 size: ' + str(constraint[0].size()))
        # print('constraint tensor 1 size: ' + str(constraint[1].size()))
        # print('constraint tensor 2 size: ' + str(constraint[2].size()))
        # print('constraint tensor 3 size: ' + str(constraint[3].size()))

        constraint_ = None
        for i, tensor in enumerate(constraint):
            if i == 0:
                constraint_ = tensor.view([-1, self.vector_size])
            else:
                tensor = tensor.view([-1, self.vector_size])
                constraint_ = torch.cat((constraint_, tensor))

        constraint = constraint_

        # print('constraint size: ' + str(constraint.size()))
        constraint2 = constraint[torch.randperm(len(constraint))]

        # vector length regularizer loss
        # r_length = constraint.norm(dim=2).sum()

        # intersecction
        r_intersection = 1 - self.similarity(self.intersection(constraint, constraint), constraint)
        r_intersection = r_intersection.sum()

        # difference
        difference_vector = self.difference(constraint, constraint2)
        r_difference = 1 - self.similarity(self.intersection(constraint, difference_vector), difference_vector)
        r_difference = r_difference.sum()

        r_loss = 0
        r_loss += r_intersection + r_difference
        r_loss = r_loss * r_weight

        # r_loss += r_length * r_length_weight

        out_dict['r_loss'] = r_loss

        return out_dict

    def forward(self, feed_dict):

        # feed_dict
        # real_batch_size
        # x_sample_num
        # y
        # x
        # iid
        # uid
        out_dict = self.predict(feed_dict)
        out_dict = self.regularizer(out_dict)

        prediction, label = out_dict['prediction'], feed_dict['y']
        # print('prediction: ' + str(prediction.size()))
        r_loss = out_dict['r_loss']

        loss = torch.nn.BCELoss(reduction='mean')(prediction.view([-1, 1]), label.view([-1, 1]))
        r_loss = 0

        out_dict['loss'] = loss + r_loss

        return out_dict



    def predict(self, feed_dict):
        # train = feed_dict['train']
        batch_size = feed_dict['real_batch_size']
        x_sample_num = feed_dict['x_sample_num']
        x = feed_dict['x']
        # pos_items = x[:, :x_sample_num]
        # neg_items = x[:, x_sample_num:]
        # batch_size 个 target items [batch_size, embedding_size]
        target_item = feed_dict['iid']
        users = feed_dict['uid']

        # batch_size * x_sample_num * 2 * embeddings_size
        # pos_items_embeddings = self.item_embeddings(pos_items)
        # neg_items_embeddings = self.item_embeddings(neg_items)
        items_embeddings = self.item_embeddings(x)
        constraint = [items_embeddings.view([batch_size, -1, self.vector_size])]
        pos_items_embeddings = items_embeddings[:, :x_sample_num, :]
        neg_items_embeddings = items_embeddings[:, x_sample_num:, :]

        # batch_size * embeddings_size
        target_item_embedding = self.item_embeddings(target_item)
        constraint.append(target_item_embedding.view([batch_size, 1, self.vector_size]))
        target_item_embedding = target_item_embedding.view([batch_size, 1, self.vector_size])
        target_item_embedding = target_item_embedding.expand_as(pos_items_embeddings)

        # print('pos_items_embeddings size: ' + str(pos_items_embeddings.size()))
        # print('target_item_embedding size: ' + str(target_item_embedding.size()))
        intersection_vector = self.intersection(pos_items_embeddings, target_item_embedding)
        constraint.append(intersection_vector.view([batch_size, -1, self.vector_size]))
        difference_vector = self.difference(target_item_embedding, neg_items_embeddings)
        constraint.append(difference_vector.view([batch_size, -1, self.vector_size]))

        # item_feature_vector: [batch_size, x_sample_num * 2, embedding_size]
        item_feature_vector = torch.cat((intersection_vector, difference_vector), dim=1)


        # intersection_vector = intersection_vector.view([batch_size, -1])
        # difference_vector = difference_vector.view([batch_size, -1])
        # item_feature_vector = torch.cat((intersection_vector, difference_vector), dim=-1)

        # users_embeddings: [batch_size, users_embeddings]
        users_embeddings = self.user_embeddings(users)

        # item_feature_vector: [batch_size, embedding_size]
        item_feature_vector = self.linear_attention(users_embeddings, item_feature_vector, x_sample_num, batch_size)

        prediction = self.predict_result(users_embeddings, item_feature_vector)

        out_dict = {
            'prediction': prediction,
            'constraint': constraint
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



if __name__ == '__main__':
    model = BaseModel(100, 10, 32, 2)

    # init models' params
    model.apply(model.init_paras)











