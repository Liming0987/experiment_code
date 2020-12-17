import torch
from tqdm import tqdm
import numpy as np
from STAMP import STAMP

from data_processor import DataProcessor
from data_loader import DataLoader
import gc

class STAMPRunner(object):
    def __init__(self, model, optimizer='Adam', lr=0.001, epoch=3, batch_size=128, l2=1e-5):
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        self.metrics = ['hit@5', 'ndcg@10']
        self.train_results, self.valid_results, self.test_results = [], [], []
        self.model_path = './model/STAMP/STAMP.pt'

    def train(self, data_processor):
        train_data = data_processor.get_train_data(epoch=-1)
        validation_data = data_processor.get_validation_data()
        test_data = data_processor.get_test_data()

        for epoch in range(self.epoch):
            epoch_train_data = data_processor.get_train_data(epoch)
            prediction, loss = self.fit(epoch_train_data, data_processor, epoch)

            train_result = [loss] + self.model.evaluate_method(prediction.reshape([-1]), train_data, ['rmse'])
            validation_result = self.evaluate(validation_data, data_processor)
            test_result = self.evaluate(test_data, data_processor)

            print("Epoch %5d \t train= %s validation= %s test= %s "
                  % (epoch + 1, self.format_metric(train_result), self.format_metric(validation_result),
                     self.format_metric(test_result)) + ','.join(self.metrics))

            self.train_results.append(train_result)
            self.valid_results.append(validation_result)
            self.test_results.append(test_result)

            # 还没有写 early stopping
            if self.best_result(self.valid_results) == self.valid_results[-1]:
                model.save_model(self.model_path)

            if self.eva_termination():
                print("Early stop at %d based on validation result." % (epoch + 1))
                break

            # find the best valiation result across interations
        best_valid_result = self.best_result(self.valid_results)
        best_epoch = self.valid_results.index(best_valid_result)
        print("Best Iter(valiation) = %5d\t train= %s valid= %s test= %s"
              % (best_epoch + 1,
                 self.format_metric(self.train_results[best_epoch]),
                 self.format_metric(self.valid_results[best_epoch]),
                 self.format_metric(self.test_results[best_epoch])) + ','.join(self.metrics))

        best_test_score = self.best_result(self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        print("Best Iter(test) = %5d\t train= %s valid= %s test= %s"
              % (best_epoch + 1,
                 self.format_metric(self.train_results[best_epoch]),
                 self.format_metric(self.valid_results[best_epoch]),
                 self.format_metric(self.test_results[best_epoch])) + ','.join(self.metrics))

    def best_result(self, results_list):
        return max(results_list)

    def eva_termination(self):
        """
        检查是否终止训练，基于验证集
        :param model:
        :return:
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if len(valid) > 20 and self.strictly_increasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(self.best_result(valid)) > 20:
            return True
        return False

    def strictly_increasing(self, l):
        return all(x < y for x, y in zip(l, l[1:]))

    def format_metric(self, metric):
        # print('evaluation length: ' + str(len(metric)))
        format_str = []
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)

        return ','.join(format_str)

    def fit(self, data, data_processor, epoch):
        gc.collect()
        batches = data_processor.prepare_batches(data, self.batch_size, train=True)
        self.model.train()
        loss_list = []
        prediction_list = []

        for i, batch in tqdm(list(enumerate(batches)), desc='Epoch %5d' % (epoch + 1)):
            batch = data_processor.batch_to_gpu(batch)
            self.optimizer.zero_grad()
            output_dict = self.model(batch)
            loss = output_dict['loss']
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())

            prediction = output_dict['prediction']
            prediction_list.append(prediction.detach().cpu().data.numpy()[:batch['real_batch_size']])

            self.optimizer.step()

        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate([b['sample_id'][:b['real_batch_size']] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data['sample_id']])

        return predictions, np.mean(loss_list)

    def evaluate(self, data, data_processor):
        batches = data_processor.prepare_batches(data, self.batch_size, train=False)
        predictions = []
        self.model.eval()
        for batch in batches:
            batch = data_processor.batch_to_gpu(batch)
            prediction = self.model.predict(batch)
            predictions.append(prediction.detach().cpu().data.numpy())
        predictions = np.concatenate(predictions)
        return self.model.evaluate_method(predictions.reshape([-1]), data, self.metrics)

if __name__ == '__main__':
    dataLoader = DataLoader('../dataset', 'ml100k01-1-5', 'label')
    dataProcessor = DataProcessor(dataLoader)

    model = STAMP(dataLoader.user_num, dataLoader.item_num, 128, 64)
    model.apply(model.init_paras)

    if torch.cuda.device_count() > 0:
        model = model.cuda()

    runner = STAMPRunner(model)

    runner.train(dataProcessor)