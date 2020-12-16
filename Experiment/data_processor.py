import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from data_loader import DataLoader
import pandas as pd


class DataProcessor(object):
    data_columns = ['uid', 'iid', 'history', 'history_neg', 'history_mix']
    info_columns = ['sample_id', 'time']

    def __init__(self, data_loader):

        np.random.seed(10)

        self.data_loader = data_loader
        self.train_sample_n = 1
        self.test_sample_n = 100
        self.sample_un_p = 1.0 #从 known negative samples 中取样本的概率
        self.train_data, self.validation_data, self.test_data = None, None, None

        # 用于 负采样
        # 1. negative sampling for training process
        # 2. negative sampling for validation/test process
        self.train_history_pos = defaultdict(set)
        for uid in data_loader.train_user_pos.keys():
            self.train_history_pos[uid] = set(data_loader.train_user_pos[uid])

        self.validation_history_pos = defaultdict(set)
        for uid in data_loader.validation_user_pos.keys():
            self.validation_history_pos[uid] = set(data_loader.validation_user_pos[uid])

        self.test_history_pos = defaultdict(set)
        for uid in data_loader.test_user_pos.keys():
            self.test_history_pos[uid] = set(data_loader.test_user_pos[uid])

        self.train_history_neg = defaultdict(set)
        for uid in data_loader.train_user_neg.keys():
            self.train_history_neg[uid] = set(data_loader.train_user_neg[uid])

        self.validation_history_neg = defaultdict(set)
        for uid in data_loader.validation_user_neg.keys():
            self.validation_history_neg[uid] = set(data_loader.validation_user_neg[uid])

        self.test_history_neg = defaultdict(set)
        for uid in data_loader.test_user_neg.keys():
            self.test_history_neg[uid] = set(data_loader.test_user_neg[uid])

        # 用于 负采样
        # self.pos_user_item_set = defaultdict(set)
        # for uid in data_loader.train_user_pos.keys():
        #     self.pos_user_item_set[uid] = set(data_loader.train_user_pos[uid])
        #
        # for uid in data_loader.validation_user_pos.keys():
        #     if uid in self.pos_user_item_set:
        #         self.pos_user_item_set[uid] = self.pos_user_item_set[uid] | set(data_loader.validation_user_pos[uid])
        #     else:
        #         self.pos_user_item_set[uid] = set(data_loader.validation_user_pos[uid])
        #
        # for uid in data_loader.test_user_pos.keys():
        #     if uid in self.pos_user_item_set:
        #         self.pos_user_item_set[uid] = self.pos_user_item_set[uid] | set(data_loader.test_user_pos[uid])
        #     else:
        #         self.pos_user_item_set[uid] = set(data_loader.test_user_pos[uid])
        #
        # self.neg_user_item_set = defaultdict(set)
        # for uid in data_loader.train_user_neg.keys():
        #     self.neg_user_item_set[uid] = set(data_loader.train_user_neg[uid])
        #
        # for uid in data_loader.validation_user_neg.keys():
        #     if uid in self.neg_user_item_set:
        #         self.neg_user_item_set[uid] = self.neg_user_item_set[uid] | set(data_loader.validation_user_neg[uid])
        #     else:
        #         self.neg_user_item_set[uid] = set(data_loader.validation_user_neg[uid])
        #
        # for uid in data_loader.test_user_neg.keys():
        #     if uid in self.neg_user_item_set:
        #         self.neg_user_item_set[uid] = self.neg_user_item_set[uid] | set(data_loader.test_user_neg[uid])
        #     else:
        #         self.neg_user_item_set[uid] = set(data_loader.test_user_neg[uid])



    def prepare_batches(self, data, batch_size, train):
        """
        将data dict全部转换为batch
        :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生 train: [uid, iid, time, y, x, history, history_neg, sample_id]
        :param batch_size: batch大小
        :param train: 训练还是测试
        :return: list of batches
        """
        num_example = len(data['y'])

        total_batch = int((num_example + batch_size - 1) / batch_size)

        neg_data = None
        if train:
            neg_data = self.generate_neg_data(data, self.data_loader.train_df, sample_n=self.train_sample_n,
                                            train=True)

        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='prepare batches'):
            batches.append(self.get_feed_dict(data=data, batch_start=batch * batch_size,
                                              batch_size=batch_size, train=train, neg_data=neg_data))
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

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None):

        total_data_num = len(data['y'])
        batch_end = min(len(data['y']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        total_data_size = real_batch_size * (self.train_sample_n + 1) if train else real_batch_size

        feed_dict = {
            'train': train,
            'real_batch_size': real_batch_size,
            'total_batch_size': total_data_size
        }

        if 'y' in data:
            feed_dict['y'] = self.numpy_to_torch(data['y'][batch_start:batch_start + real_batch_size], gpu=False)
        # data_columns = ['uid', 'iid', 'history', 'history_neg', 'history_mix']
        # info_columns = ['sample_id', 'time']
        for c in self.info_columns + self.data_columns:
            d = data[c][batch_start: batch_start + real_batch_size]

            # 只有 train 的时候才会加 neg samples
            if train:
                # total_data_num 非常重要
                neg_d = np.concatenate([neg_data[c][total_data_num * i + batch_start: total_data_num * i + batch_start + real_batch_size]
                                        for i in range(self.train_sample_n)])
                d = np.concatenate([d, neg_d])

            feed_dict[c] = d

        for c in self.data_columns:
            if c not in ['history', 'history_neg', 'history_mix']:
                feed_dict[c] = self.numpy_to_torch(feed_dict[c])

        # 添加 history_neg_length 和 history_length
        # 填充 history 和 history_neg 使得每个batch中的 sequence 的长度一致
        his_ls = ['history_length', 'history_neg_length', 'history_mix_length']
        for i, c in enumerate(['history', 'history_neg', 'history_mix']):
            lc, d = his_ls[i], feed_dict[c]
            lengths = [len(iids) for iids in d]
            max_length = max(lengths)
            # 为什么要填充 0 是因为 每次都是一个batch 进行处理，如果不填充0使得sequence长度一致的话 会报错
            new_d = np.array([x + [0] * (max_length - len(x)) for x in d])
            feed_dict[c] = self.numpy_to_torch(new_d)
            feed_dict[lc] = lengths

        return feed_dict

        # x: batch_size, x_sample_num * 2
        # iid
        # uid

    # def get_validation_data(self):
    #     df = self.generate_x_samples(self.data_loader.test_df)
    #     self.test_data = self.format_data_dict(df)
    #     return self.test_data
    #
    # def get_test_data(self):
    #     df = self.generate_x_samples(self.data_loader.validation_df)
    #     self.validation_data = self.format_data_dict(df)
    #     return self.validation_data

    # def get_train_data(self, epoch):
    #     if self.train_data is None:
    #         df = self.generate_x_samples(self.data_loader.train_df)
    #         self.train_data = self.format_data_dict(df)
    #     #             self.train_data['sample_id'] = np.arange(0, len(self.train_data['y']))
    #
    #     if epoch >= 0:
    #         np.random.seed(10)  # 保证所有的column shuffle的顺序一样
    #         rng_state = np.random.get_state()
    #         for d in self.train_data:
    #             np.random.set_state(rng_state)
    #             np.random.shuffle(self.train_data[d])
    #
    #     return self.train_data
    def get_train_data(self, epoch):
        if self.train_data is None:
            self.train_data = self.format_data_dict(self.data_loader.train_df)
            self.train_data['sample_id'] = np.arange(0, len(self.train_data['y']))

        if epoch >= 0:
            state = np.random.get_state()
            for key in self.train_data:
                np.random.set_state(state)
                np.random.shuffle(self.train_data[key])

        return self.train_data

    def get_validation_data(self):
        df = self.data_loader.validation_df
        if self.validation_data is None:
            tmp_df = df.rename(columns={'label': 'y'})
            tmp_df = tmp_df.drop(tmp_df[tmp_df['y'] <= 0].index)
            neg_df = self.generate_neg_df(inter_df=tmp_df, df=df, sample_n=self.test_sample_n, train=False)
            df = pd.concat([df, neg_df], ignore_index=True)

            self.validation_data = self.format_data_dict(df)
            self.validation_data['sample_id'] = np.arange(0, len(self.validation_data['y']))

        return self.validation_data

    def get_test_data(self):
        df = self.data_loader.test_df
        if self.test_data is None:
            tmp_df = df.rename(columns={'label': 'y'})
            tmp_df = tmp_df.drop(tmp_df[tmp_df['y'] <= 0].index)
            neg_df = self.generate_neg_df(inter_df=tmp_df, df=df, sample_n=self.test_sample_n, train=False)

            # neg_df: [uid, iid, y, time, history, history_neg]
            # df: [uid, iid, y, time, history, history_neg]
            # 主要区别是
            # 1. df 中是既有 正例 也有 负例
            # 2. neg_df 中只有 负例 from negative sampling
            df = pd.concat([df, neg_df], ignore_index=True)

            self.test_data = self.format_data_dict(df)
            self.test_data['sample_id'] = np.arange(0, len(self.test_data['y']))

        return self.test_data


    # # 每个 uid iid 产生 5 个 正样本 和 5 个负样本
    # def generate_x_samples(self, df, x_number=5, stage='train'):
    #     x_samples_list = []
    #     idx_del = []
    #
    #     for i, uid in enumerate(df['uid'].values):
    #
    #         iid = df['iid'].values[i]
    #
    #         # get x_number positive samples
    #         pos_iids = self.pos_user_item_set[uid]
    #         if len(pos_iids) == 0:
    #             idx_del.append(i)
    #             continue
    #
    #         if iid in pos_iids:
    #             pos_iids.remove(iid)
    #             if x_number < len(pos_iids):
    #                 pos_samples = np.random.choice(list(pos_iids), x_number, replace=False)
    #             else:
    #                 pos_samples = np.random.choice(list(pos_iids), x_number, replace=True)
    #             pos_iids.add(iid)
    #         else:
    #             if x_number < len(pos_iids):
    #                 pos_samples = np.random.choice(list(pos_iids), x_number, replace=False)
    #             else:
    #                 pos_samples = np.random.choice(list(pos_iids), x_number, replace=True)
    #
    #         # get x_number negative samples (true negative and non-seen movies)
    #         neg_samples = set()
    #         for i in range(x_number):
    #             neg_iid = np.random.randint(1, self.data_loader.item_num)
    #             while neg_iid in neg_samples or neg_iid in pos_iids or neg_iid == iid:
    #                 neg_iid = np.random.randint(1, self.data_loader.item_num)
    #             neg_samples.add(neg_iid)
    #         neg_samples = list(neg_samples)
    #
    #         x_samples = np.concatenate([pos_samples, neg_samples]).tolist()
    #         x_samples_list.append(x_samples)
    #
    #     #         print('uid delete')
    #     #         for uid in uids_del:
    #     #             print(uid)
    #
    #     df = df.drop(idx_del)
    #     df['x'] = x_samples_list
    #
    #     return df

    def generate_neg_data(self, data, df, sample_n, train):
        """

        :param data: dict [uid, iid, time, y, history, history_neg, sample_id] sample_id 是后面加的
        :param df: self.data_loader.train_df
        :param sample_n: 1
        :param train: True
        :return:
        """
        inter_df = pd.DataFrame()
        for c in ['uid', 'iid', 'y', 'time']:
            inter_df[c] = data[c]

        # neg_df [uid, iid_neg, iid, time] iid 是正样本，iid_neg 是负样本
        # inter_df: [uid, iid, y, time]
        # df: [uid, iid, y, time, history, history_neg]
        neg_df = self.generate_neg_df(
            inter_df=inter_df, df=df,
            sample_n=sample_n, train=train
        )

        neg_data = self.format_data_dict(neg_df)
        neg_data['sample_id'] = np.arange(0, len(neg_data['y'])) + len(data['sample_id'])

        return neg_data
    def generate_neg_df(self, inter_df, df, sample_n, train):
        other_columns = ['iid', 'time']

        neg_df = self.sample_neg_from_uid_list(uids=inter_df['uid'].tolist(), labels=inter_df['y'].tolist(),
                                               sample_n=sample_n, train=train, other_infos=inter_df[other_columns].to_dict('list'))

        #主要目的是给 neg_df 添加 history and history_neg
        neg_df = pd.merge(neg_df, df, on=['uid'] + other_columns, how='left')
        neg_df = neg_df.drop(columns=['iid'])
        neg_df = neg_df.rename(columns={'iid_neg': 'iid'})

        neg_df = neg_df[df.columns]
        neg_df['label'] = 0
        return neg_df


    def sample_neg_from_uid_list(self, uids, labels, sample_n, train, other_infos=None):
        iid_list = []
        other_infos_list = {}
        for info in other_infos:
            other_infos_list[info] = []

        item_num = self.data_loader.item_num
        for index, uid in enumerate(uids):
            # 避免采到已知的 正样本
            # 这里遍历的每个例子都是正样本，我们需要采集负样本（1.明确的负样本 2.从未接触过的负样本），
            train_history = self.train_history_pos
            validation_history, test_history = self.validation_history_pos, self.test_history_pos
            known_train = self.train_history_neg

            if train:
                # 在train过程中采负样本，有可能采集到该uid 在 validation 和 test 中的正样本 ？为什么要这么做呢
                inter_iids = train_history[uid]
            else:
                inter_iids = train_history[uid] | validation_history[uid] | test_history[uid]

            remain_iids_num = item_num - len(inter_iids)
            assert remain_iids_num >= sample_n

            remain_iids = None
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]

            sampled = set()
            if remain_iids is None:
                unknown_iid_list = []
                for i in range(sample_n):
                    iid = np.random.randint(1, self.data_loader.item_num)
                    while iid in inter_iids or iid in sampled:
                        iid = np.random.randint(1, self.data_loader.item_num)
                    unknown_iid_list.append(iid)
                    sampled.add(iid)
            else:
                unknown_iid_list = np.random.choice(remain_iids, sample_n, replace=False)

            if train and self.sample_un_p < 1:
                known_iid_list = list(np.random.choice(list(known_train[uid]), min(sample_n, len(known_train[uid])), replace=False)) \
                    if len(known_train[uid]) != 0 else []

                # 这样做是为了以防 每次抽到的都是 known_iid_list 中的值，但 known_iid_list 中的元素个数又不够数量
                known_iid_list = known_iid_list + unknown_iid_list

                tmp_iid_list = []
                sampled = set()
                for i in range(sample_n):
                    p = np.random.rand()
                    if p < self.sample_un_p or len(known_iid_list) == 0: # len(known_iid_list)为空的情况可能会出现，当该uid没有负样本时
                        iid = unknown_iid_list.pop(0)
                        while iid in sampled:
                            iid = unknown_iid_list.pop(0)
                    else:
                        iid = known_iid_list.pop(0)
                        while iid in sampled:
                            iid = known_iid_list.pop(0)
                    tmp_iid_list.append(iid)
                    sampled.add(iid)
                iid_list.append(tmp_iid_list)
            else:
                iid_list.append(unknown_iid_list)

        all_uid_list, all_iid_list = [], []

        # 这种遍历顺序非常关键
        # 保证每次都是完整的 inter_df 的顺序 : 表述再清楚点
        for i in range(sample_n):
            for index, uid in enumerate(uids):
                all_uid_list.append(uid)
                all_iid_list.append(iid_list[index][i])

                for info in other_infos:
                    other_infos_list[info].append(other_infos[info][index])

        neg_df = pd.DataFrame(data=list(zip(all_uid_list, all_iid_list)), columns=['uid', 'iid_neg'])
        for info in other_infos:
            neg_df[info] = other_infos_list[info]

        return neg_df




    def format_data_dict(self, df):
        """
        transform the df to dictionary, and delete rows with either empty history or empty history_neg
        :param df: train df/validation df/test df
        :return: data dictionary: [uid, iid, time, y, history, history_neg]
        """
        assert 'history' in df
        data_loader = self.data_loader

        his_cs = ['history', 'history_neg', 'history_mix']

        # 删除 df 中 的 history or history_neg or history_mix 是空 的row
        for c in his_cs:
            df = df[df[c].apply(lambda x: len(x) > 0)]

        data = {}

        if 'uid' in df:
            data['uid'] = df['uid'].values
        if 'iid' in df:
            data['iid'] = df['iid'].values
        if 'time' in df:
            data['time'] = df['time'].values
        if 'label' in df:
            data['y'] = np.array(df['label'], dtype=np.float32)
        # if 'x' in df:
        #     data['x'] = np.array(df['x'].values.tolist())
        for c in his_cs:
            # 转化成 list
            his = df[c].apply(lambda x: eval('[' + x + ']'))
            data[c] = his.values

        return data



# if __name__ == '__main__':
#     dataLoader = DataLoader('./dataset', 'ml100k01-1-5', 'label')
#     dataprocessor = DataProcessor(dataLoader)
#     train_data = dataprocessor.get_train_data(0)
#     print(train_data)
