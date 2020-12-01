import torch
from tqdm import tqdm
import numpy as np
from MFModel import MF

from data_processor import DataProcessor
from data_loader import DataLoader

class MFRunner(object):
    def __init__(self, model, optimizer='Adam', lr=0.001, epoch=1, batch_size=128, l2=1e-5):
        self.model = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    def train(self, data_processor):
        validation_data = data_processor.get_validation_data()

        for i in range(self.epoch):
            train_data = data_processor.get_train_data(i)
            prediction, loss = self.fit(train_data, data_processor, i)

            print('prediction')
            print(prediction)

            print('label')
            print(train_data['y'])

            evaluations = self.model.evaluate_method(prediction, train_data, ['precision', 'precision@5'])
            print('train precision: ' + self.format_metric(evaluations))

            validation_evaluations =self.evaluate(validation_data, data_processor, ['precision', 'precision@5'])
            print('evaluation precision: ' + self.format_metric(validation_evaluations))


    def format_metric(self, metric):
        format_str = []
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)

        return ','.join(format_str)

    def fit(self, data, data_processor, epoch):
        batches = data_processor.prepare_batches(data, self.batch_size)
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
            prediction_list.append(prediction.detach().cpu().data.numpy())

            self.optimizer.step()

        predictions = np.concatenate(prediction_list)

        return predictions, np.mean(loss_list)

    def evaluate(self, data, data_processor, metrics):
        batches = data_processor.prepare_batches(data, self.batch_size)
        predictions = []
        self.model.eval()
        for batch in batches:
            batch = data_processor.batch_to_gpu(batch)
            prediction = self.model.predict(batch)
            predictions.append(prediction.detach().cpu().data.numpy())
        predictions = np.concatenate(predictions)
        return self.model.evaluate_method(predictions, data, metrics)

if __name__ == '__main__':
    dataLoader = DataLoader('../dataset', 'ml100k01-1-5', 'label')
    dataProcessor = DataProcessor(dataLoader)

    model = MF(dataLoader.user_num, dataLoader.item_num, 128)
    model.apply(model.init_paras)

    if torch.cuda.device_count() > 0:
        model = model.cuda()

    runner = MFRunner(model)

    runner.train(dataProcessor)









