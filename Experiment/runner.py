import torch
from tqdm import tqdm
import numpy as np
import os
from BaseModel import BaseModel
from data_processor import DataProcessor
from data_loader import DataLoader
import logging

class BaseRunner(object):

    def __init__(self, optimizer='GD', lr=0.01, epoch=1, batch_size=128, eval_batch_size=128, l2=1e-5,
                 dropout=0.2, grad_clip=10):
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.l2_weight = l2
        self.dropout = dropout
        self.grad_clip = grad_clip

        self.train_results, self.valid_results, self.test_results = [], [], []
        self.model_path = './model/'

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        return optimizer

    # for loop epoches
    def train(self, model, data_processor):
        train_data = data_processor.get_train_data(epoch=-1)
        validation_data = data_processor.get_validation_data()
        test_data = data_processor.get_test_data()


        try:
            for epoch in range(self.epoch):
                epoch_train_data = data_processor.get_train_data(epoch=epoch)
                train_predictions, mean_loss = self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                print('mean loss: ' + str(mean_loss))
                # evaluate 模型效果
                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, ['precision'])
                # validation_result = self.evaluate(model, validation_data, data_processor)
                # test_result = self.evaluate(model, test_data, data_processor)
                #
                # self.train_results.append(train_result)
                # self.valid_results.append(validation_result)
                # self.test_results.append(test_result)
                print('precision: ' + self.format_metric(train_result))
                # logging.info("Epoch %5d \t train = %s " % (epoch + 1, self.format_metric(train_result)))
                return train_predictions, epoch_train_data


        except KeyboardInterrupt:
            save_here = input('Save here? (1/0): ')
            if str(save_here).lower().startswith('1'):
                model.save_model(self.model_path)
    def format_metric(self, metric):
        format_str = []
        for m in metric:
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)

        return ','.join(format_str)


    def predict(self, model, data, data_process):
        batches = data_process.prepare_batches(data, self.batch_size)
        model.eval()
        predictions = []
        for batch in tqdm(batches):
            batch = data_process.batch_to_gpu(batch)
            prediction = model.predict(batch)['prediction']
            predictions.append(prediction.detach().cpu().data.numpy())

        predictions = np.concatenate(predictions)
        return predictions

    def evaluate(self, model, data, data_processor):
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data)

    # for loop batches
    def fit(self, model, data, data_processor, epoch):
        model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size)

        model.train()

        loss_list = []
        prediction_list = []

        for i, batch in tqdm(list(enumerate(batches))):
            batch = data_processor.batch_to_gpu(batch)

            model.optimizer.zero_grad()

            output_dict = model(batch)

            loss = output_dict['loss']

            loss.backward()

            loss_list.append(loss.detach().cpu().data.numpy())
            prediction_list.append(output_dict['prediction'].detach().cpu().data.numpy())

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)

            model.optimizer.step()

        predictions = np.concatenate(prediction_list)

        return predictions, np.mean(loss_list)

if __name__ == '__main__':
    dataLoader = DataLoader('./dataset', 'ml100k01-1-5', 'label')
    dataProcessor = DataProcessor(dataLoader)
    model = BaseModel(dataLoader.item_num, dataLoader.user_num, 64, 2)
    model.apply(model.init_paras)

    runner = BaseRunner()
    prediction, data = runner.train(model, dataProcessor)
    # test_data = dataProcessor.get_test_data()
    # prediction = runner.predict(model, test_data, dataProcessor)

    # np.save('prediction', prediction)
    # np.save('y', test_data['y'])
    # np.save('test_data', test_data)

