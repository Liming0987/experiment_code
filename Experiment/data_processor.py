import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from data_loader import DataLoader


class DataProcessor(object):
    data_columns = ['uid', 'iid', 'x']
    info_columns = ['sample_id', 'time']

    def __init__(self, data_loader):
        self.data_loader = data_loader
        #         self.train_sample_n = train_sample_n
        #         self.test_sample_n = test_sample_n
        self.train_data = None

        self.pos_user_item_set = defaultdict(set)
        for uid in data_loader.train_user_pos.keys():
            self.pos_user_item_set[uid] = set(data_loader.train_user_pos[uid])

        for uid in data_loader.validation_user_pos.keys():
            if uid in self.pos_user_item_set:
                self.pos_user_item_set[uid] = self.pos_user_item_set[uid] | set(data_loader.validation_user_pos[uid])
            else:
                self.pos_user_item_set[uid] = set(data_loader.validation_user_pos[uid])

        for uid in data_loader.test_user_pos.keys():
            if uid in self.pos_user_item_set:
                self.pos_user_item_set[uid] = self.pos_user_item_set[uid] | set(data_loader.test_user_pos[uid])
            else:
                self.pos_user_item_set[uid] = set(data_loader.test_user_pos[uid])

        self.neg_user_item_set = defaultdict(set)
        for uid in data_loader.train_user_neg.keys():
            self.neg_user_item_set[uid] = set(data_loader.train_user_neg[uid])

        for uid in data_loader.validation_user_neg.keys():
            if uid in self.neg_user_item_set:
                self.neg_user_item_set[uid] = self.neg_user_item_set[uid] | set(data_loader.validation_user_neg[uid])
            else:
                self.neg_user_item_set[uid] = set(data_loader.validation_user_neg[uid])

        for uid in data_loader.test_user_neg.keys():
            if uid in self.neg_user_item_set:
                self.neg_user_item_set[uid] = self.neg_user_item_set[uid] | set(data_loader.test_user_neg[uid])
            else:
                self.neg_user_item_set[uid] = set(data_loader.test_user_neg[uid])

    def prepare_batches(self, data, batch_size):
        num_example = len(data['y'])

        total_batch = int((num_example + batch_size - 1) / batch_size)

        batches = []
        for batch in tqdm(range(total_batch)):
            batches.append(self.get_feed_dict(data=data, batch_start=batch * batch_size,
                                              batch_size=batch_size))
        return batches

    def numpy_to_torch(self, d, gpu=True, requires_grad=True):
        t = torch.from_numpy(d)
        if d.dtype is np.float:
            t.requires_grad = requires_grad
        if gpu and torch.cuda.device_count() > 0:
            t = t.cuda()
        return t

    def batch_to_gpu(self, batch):
        if torch.cuda.device_count() > 0:
            new_batch = {}
            for c in batch:
                if type(batch[c]) is torch.Tensor:
                    new_batch[c] = batch[c].cuda()
                else:
                    new_batch[c] = batch[c]
            return new_batch
        return batch

    def get_feed_dict(self, data, batch_start, batch_size):
        total_data_num = len(data['y'])
        batch_end = min(len(data['y']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start

        feed_dict = {
            'real_batch_size': real_batch_size,
            'x_sample_num': 5,
        }

        feed_dict['y'] = self.numpy_to_torch(data['y'][batch_start:batch_start + real_batch_size])

        feed_dict['x'] = self.numpy_to_torch(data['x'][batch_start:batch_start + real_batch_size])

        feed_dict['iid'] = self.numpy_to_torch(data['iid'][batch_start:batch_start + real_batch_size])

        feed_dict['uid'] = self.numpy_to_torch(data['uid'][batch_start:batch_start + real_batch_size])

        return feed_dict

        # x: batch_size, x_sample_num * 2
        # iid
        # uid

    def get_validation_data(self):
        df = self.generate_x_samples(self.data_loader.test_df)
        self.test_data = self.format_data_dict(df)
        return self.test_data

    def get_test_data(self):
        df = self.generate_x_samples(self.data_loader.validation_df)
        self.validation_data = self.format_data_dict(df)
        return self.validation_data

    def get_train_data(self, epoch):
        if self.train_data is None:
            df = self.generate_x_samples(self.data_loader.train_df)
            self.train_data = self.format_data_dict(df)
        #             self.train_data['sample_id'] = np.arange(0, len(self.train_data['y']))

        if epoch >= 0:
            np.random.seed(10)  # 保证所有的column shuffle的顺序一样
            rng_state = np.random.get_state()
            for d in self.train_data:
                np.random.set_state(rng_state)
                np.random.shuffle(self.train_data[d])

        return self.train_data

    # 每个 uid iid 产生 5 个 正样本 和 5 个负样本
    def generate_x_samples(self, df, x_number=5, stage='train'):
        x_samples_list = []
        idx_del = []

        for i, uid in enumerate(df['uid'].values):

            iid = df['iid'].values[i]

            # get x_number positive samples
            pos_iids = self.pos_user_item_set[uid]
            if len(pos_iids) == 0:
                idx_del.append(i)
                continue

            if iid in pos_iids:
                pos_iids.remove(iid)
                if x_number < len(pos_iids):
                    pos_samples = np.random.choice(list(pos_iids), x_number, replace=False)
                else:
                    pos_samples = np.random.choice(list(pos_iids), x_number, replace=True)
                pos_iids.add(iid)
            else:
                if x_number < len(pos_iids):
                    pos_samples = np.random.choice(list(pos_iids), x_number, replace=False)
                else:
                    pos_samples = np.random.choice(list(pos_iids), x_number, replace=True)

            # get x_number negative samples (true negative and non-seen movies)
            neg_samples = set()
            for i in range(x_number):
                neg_iid = np.random.randint(1, self.data_loader.item_num)
                while neg_iid in neg_samples or neg_iid in pos_iids or neg_iid == iid:
                    neg_iid = np.random.randint(1, self.data_loader.item_num)
                neg_samples.add(neg_iid)
            neg_samples = list(neg_samples)

            x_samples = np.concatenate([pos_samples, neg_samples]).tolist()
            x_samples_list.append(x_samples)

        #         print('uid delete')
        #         for uid in uids_del:
        #             print(uid)

        df = df.drop(idx_del)
        df['x'] = x_samples_list

        return df

    def format_data_dict(self, df):

        # df = df[df['history'].apply(lambda x: len(x) > 0)]
        data_loader = self.data_loader
        data = {}

        if 'uid' in df:
            data['uid'] = df['uid'].values
        if 'iid' in df:
            data['iid'] = df['iid'].values
        if 'time' in df:
            data['time'] = df['time'].values
        if 'label' in df:
            data['y'] = np.array(df['label'], dtype=np.float32)
        if 'x' in df:
            data['x'] = np.array(df['x'].values.tolist())

        return data

if __name__ == '__main__':
    dataLoader = DataLoader('./dataset', 'ml100k01-1-5', 'label')
    dataprocessor = DataProcessor(dataLoader)
    train_data = dataprocessor.get_train_data(0)
    print(train_data)
