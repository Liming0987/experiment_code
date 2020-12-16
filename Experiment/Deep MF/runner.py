import torch
from tqdm import tqdm
import numpy as np
from DeepMF import DeepMF

from data_processor import DataProcessor
from data_loader import DataLoader
import gc

class DeepMFRunner(object):
    def __init__(self, model, lr=0.001, epoch=3, batch_size=128, l2=1e-5):
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        self.metrics = ['precision@1', 'ndcg@10']

    def train(self, data_processor):
        validation_data = data_processor.get_validation_data()
        train_data = data_processor.get_train_data(epoch=-1)
        test_data = data_processor.get_test_data()
        # print('validation mean: ' + str(np.mean(validation_data['y'])))
        # print('len of validation data: ' + str(len(validation_data['y'])))
        # print('The number of 1 in validation data: ' + str(np.sum(validation_data['y'])))

        for epoch in range(self.epoch):
            epoch_train_data = data_processor.get_train_data(epoch=epoch)
            prediction, loss = self.fit(epoch_train_data, data_processor, epoch)

            # train data 的 evaluation 可有可无
            train_result = [loss] + self.model.evaluate_method(prediction.reshape([-1]), train_data, ['rmse'])
            validation_result = self.evaluate(validation_data, data_processor)
            test_result = self.evaluate(test_data, data_processor)

            print("Epoch %5d \t train= %s validation= %s test= %s "
                  % (epoch + 1, self.format_metric(train_result), self.format_metric(validation_result),
                     self.format_metric(test_result)) + ','.join(self.metrics))

    def format_metric(self, metric):
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

        for i, batch in tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
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

    model = DeepMF(dataProcessor, dataLoader.user_num, dataLoader.item_num, 128)
    model.apply(model.init_paras)

    if torch.cuda.device_count() > 0:
        model = model.cuda()

    runner = DeepMFRunner(model)

    runner.train(dataProcessor)