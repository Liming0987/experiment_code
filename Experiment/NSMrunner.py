import torch
from tqdm import tqdm
import numpy as np
import os
from NeuralSetModel import BaseModel
from data_processor import DataProcessor
from data_loader import DataLoader
import logging
import gc

class BaseRunner(object):

    def __init__(self, optimizer='GD', lr=0.01, epoch=1, batch_size=128, eval_batch_size=128, l2=1e-5,
                 dropout=0.2, grad_clip=10):
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.l2_weight = l2
        # whether add l2 to bias
        self.l2_bias = 0
        self.dropout = dropout
        self.grad_clip = grad_clip

        self.metrics = ['hit@5', 'ndcg@10']

        self.train_results, self.valid_results, self.test_results = [], [], []
        self.model_path = './model/NSM/NSM.pt'

    def _build_optimizer(self, model):
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]


        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    # for loop epoches
    def train(self, model, data_processor):
        train_data = data_processor.get_train_data(epoch=-1)
        validation_data = data_processor.get_validation_data()
        test_data = data_processor.get_test_data()

        try:
            for epoch in range(self.epoch):
                # 每一轮都重新获得训练数据，重新进行负采样
                epoch_train_data = data_processor.get_train_data(epoch=epoch)

                # 负采样是在 fit 过程中进行的
                train_predictions, mean_loss = self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                # evaluate 模型效果
                # 这里 train_data 不能用 epoch_train_data 是因为 epoch_train_data 是打乱顺序的
                # train_prediction是重新排列顺序的

                # 这里的 train_result 包含来 Bayesian Personalized Ranking Loss 和 RMSE
                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, ['rmse'])
                validation_result = self.evaluate(model, validation_data, data_processor)
                test_result = self.evaluate(model, test_data, data_processor)

                self.train_results.append(train_result)
                self.valid_results.append(validation_result)
                self.test_results.append(test_result)

                # 输出当前模型效果
                print("Epoch %5d \t train= %s validation= %s test= %s "
                      % (epoch + 1, self.format_metric(train_result), self.format_metric(validation_result),
                         self.format_metric(test_result)) + ','.join(self.metrics))

                # print('precision: ' + self.format_metric(train_result))
                # logging.info("Epoch %5d \t train = %s " % (epoch + 1, self.format_metric(train_result)))
                # return train_predictions, epoch_train_data

                # 还没有写 early stopping
                if self.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    model.save_model()

                if self.eva_termination(model):
                    print("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            save_here = input('Save here? (1/0): ')
            if str(save_here).lower().startswith('1'):
                model.save_model(self.model_path)

        # find the best valiation result across interations
        best_valid_result = self.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_result)
        print("Best Iter(valiation) = %5d\t train= %s valid= %s test= %s"
              % (best_epoch + 1,
                 self.format_metric(self.train_results[best_epoch]),
                 self.format_metric(self.valid_results[best_epoch]),
                 self.format_metric(self.test_results[best_epoch])) + ','.join(self.metrics))

        best_test_score = self.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        print("Best Iter(valiation) = %5d\t train= %s valid= %s test= %s"
              % (best_epoch + 1,
                 self.format_metric(self.train_results[best_epoch]),
                 self.format_metric(self.valid_results[best_epoch]),
                 self.format_metric(self.test_results[best_epoch])) + ','.join(self.metrics))

    def best_result(self, metric, results_list):
        return max(results_list)

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model:
        :return:
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if len(valid) > 20 and self.strictly_increasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(self.best_result(metric, valid)) > 20:
            return True
        return False

    def strictly_increasing(self, l):
        return all(x < y for x, y in zip(l, l[1:]))

    def format_metric(self, metric):
        format_str = []
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)

        return ','.join(format_str)

    def predict(self, model, data, data_process):
        """

        :param model:
        :param data:
        :param data_process:
        :return: 排序后的预测结果
        """
        gc.collect()
        batches = data_process.prepare_batches(data, self.batch_size, train=False)
        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = data_process.batch_to_gpu(batch)
            prediction = model.predict(batch)['prediction']
            predictions.append(prediction.detach().cpu().data.numpy())

        predictions = np.concatenate(predictions)

        # 按 sample id 重新进行排序, 这样排序之后 prediction 就是对应 的 np.concat(data, neg_data) 的顺序
        # neg data 是按batch 来取的，每个batch的 正样本是按顺序取的，但 负样本是在 neg data 中跳着取的
        sample_ids = np.concatenate([b['sample_id'] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data['sample_id']])

        gc.collect()

        # 如果是 training 的数据，对应 np.concat(data, neg_data) 的顺序
        # 如果是 validation/test 的数据，对应 原先的顺序 因为 data 包含来 neg_data
        return predictions

    def evaluate(self, model, data, data_processor):
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=self.metrics)

    # for loop batches, only for training
    def fit(self, model, data, data_processor, epoch=-1):

        gc.collect()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        # prepare_batches 函数中 将 neg_data 加入batch 中 如果是training
        batches = data_processor.prepare_batches(data, self.batch_size, train=True)

        batch_size = self.batch_size * 2

        model.train()
        accumulate_size = 0
        prediction_list = []

        loss_list = []

        for i, batch in tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            batch = data_processor.batch_to_gpu(batch)

            # print(batch.keys())
            accumulate_size += len(batch['y'])
            model.optimizer.zero_grad()

            output_dict = model(batch)

            # loss = rank_loss + r_loss
            loss = output_dict['loss']

            loss.backward()

            loss_list.append(loss.detach().cpu().data.numpy())
            prediction_list.append(output_dict['prediction'].detach().cpu().data.numpy()[:batch['real_batch_size']])

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)

            # 不知道为什么要这么做
            if accumulate_size >= batch_size or i == len(batches) - 1:
                model.optimizer.step()
                accumulate_size = 0

        # 原先的顺序不是按 sample id的顺序进行排列，因为传入 fit 函数的 data 是随机打乱顺序的
        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate([b['sample_id'][:b['real_batch_size']] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data['sample_id']])

        return predictions, np.mean(loss_list)

if __name__ == '__main__':
    dataLoader = DataLoader('./dataset', 'ml100k01-1-5', 'label')
    dataProcessor = DataProcessor(dataLoader)
    model = BaseModel(dataLoader.item_num, dataLoader.user_num, 64, 2)
    model.apply(model.init_paras)

    runner = BaseRunner()
    runner.train(model, dataProcessor)
    # test_data = dataProcessor.get_test_data()
    # prediction = runner.predict(model, test_data, dataProcessor)

    # np.save('prediction', prediction)
    # np.save('y', test_data['y'])
    # np.save('test_data', test_data)

