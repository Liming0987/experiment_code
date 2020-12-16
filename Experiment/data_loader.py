import pandas as pd
import os

TRAIN_SUFFIX = '.train.csv'
VALIDATION_SUFFIX = '.validation.csv'
TEST_SUFFIX = '.test.csv'  # 测试集文件后缀

INFO_SUFFIX = '.info.json'  # 数据集统计信息文件后缀
USER_SUFFIX = '.user.csv'  # 数据集用户特征文件后缀
ITEM_SUFFIX = '.item.csv'  # 数据集物品特征文件后缀

TRAIN_POS_SUFFIX = '.train_pos.csv'  # 训练集用户正向交互按uid合并之后的文件后缀
VALIDATION_POS_SUFFIX = '.validation_pos.csv'  # 验证集用户正向交互按uid合并之后的文件后缀
TEST_POS_SUFFIX = '.test_pos.csv'  # 测试集用户正向交互按uid合并之后的文件后缀

TRAIN_NEG_SUFFIX = '.train_neg.csv'  # 训练集用户负向交互按uid合并之后的文件后缀
VALIDATION_NEG_SUFFIX = '.validation_neg.csv'  # 验证集用户负向交互按uid合并之后的文件后缀
TEST_NEG_SUFFIX = '.test_neg.csv'  # 测试集用户负向交互按uid合并之后的文件后缀


class DataLoader(object):

    def __init__(self, path, dataset, label, sep='\t'):
        self.dataset = dataset
        self.path = os.path.join(path, dataset)

        self.train_file = os.path.join(self.path, dataset + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + TEST_SUFFIX)

        self.info_file = os.path.join(self.path, dataset + INFO_SUFFIX)
        self.user_file = os.path.join(self.path, dataset + USER_SUFFIX)
        self.item_file = os.path.join(self.path, dataset + ITEM_SUFFIX)

        self.train_pos_file = os.path.join(self.path, dataset + TRAIN_POS_SUFFIX)
        self.validation_pos_file = os.path.join(self.path, dataset + VALIDATION_POS_SUFFIX)
        self.test_pos_file = os.path.join(self.path, dataset + TEST_POS_SUFFIX)

        self.train_neg_file = os.path.join(self.path, dataset + TRAIN_NEG_SUFFIX)
        self.validation_neg_file = os.path.join(self.path, dataset + VALIDATION_NEG_SUFFIX)
        self.test_neg_file = os.path.join(self.path, dataset + TEST_NEG_SUFFIX)

        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self.load_user_item()
        self.load_interaction_data()
        self.load_his()
        self.load_info()
        self.append_his()
        # 对于 Top-K Recommendation，只需保留 正样本，负样本是是训练过程中产生的
        self.drop_neg()
        # self.save_info()

    def save_info(self):
        self.train_df.to_csv(os.path.join(self.path, self.dataset + '.train_his.csv'), index=False)
        self.validation_df.to_csv(os.path.join(self.path, self.dataset + '.validation_his.csv'),index=False)
        self.test_df.to_csv(os.path.join(self.path, self.dataset + '.test.csv'), index=False)

    def load_user_item(self):
        self.user_df, self.item_df = None, None
        if os.path.exists(self.user_file):
            self.user_df = pd.read_csv(self.user_file, sep='\t')
        if os.path.exists(self.item_file):
            self.item_df = pd.read_csv(self.item_file, sep='\t')

    def load_interaction_data(self):
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.validation_df = pd.read_csv(self.validation_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file)

    # 此处的数据是在 data_processor 中用到
    def load_his(self):
        # 把 df [uid, iids] 变成 dict {1: [iid, iid, ...] 2: [iid, iid, ...]}
        def build_his(df):
            uids = df['uid'].tolist()

            iids = df['iids'].astype(str).str.split(',').values

            iids = [[int(j) for j in i] for i in iids]
            user_his = dict(zip(uids, iids))
            return user_his

        # 把 df [uid, iids] 变成 dict {1: [iid, iid, ...] 2: [iid, iid, ...]}
        self.train_pos_df = pd.read_csv(self.train_pos_file, sep='\t')
        self.train_user_pos = build_his(self.train_pos_df)

        self.validation_pos_df = pd.read_csv(self.validation_pos_file, sep='\t')
        self.validation_user_pos = build_his(self.validation_pos_df)

        self.test_pos_df = pd.read_csv(self.test_pos_file, sep='\t')
        self.test_user_pos = build_his(self.test_pos_df)

        self.train_neg_df = pd.read_csv(self.train_neg_file, sep='\t')
        self.train_user_neg = build_his(self.train_neg_df)

        self.validation_neg_df = pd.read_csv(self.validation_neg_file, sep='\t')
        self.validation_user_neg = build_his(self.validation_neg_df)

        self.test_neg_df = pd.read_csv(self.test_neg_file, sep='\t')
        self.test_user_neg = build_his(self.test_neg_df)

    # 正例 history 和 负例 history 的 max_history 都是10
    def append_his(self, max_his=10):
        # 包含了 train, validation, test 中的所有的 uid 和 其对应的 iids
        # 正样本是 iid, 负样本是 -iid
        his_dict = {}
        neg_dict = {}
        mix_dict = {}

        for df in [self.train_df, self.validation_df, self.test_df]:

            history, neg_history, mix_history = [], [], []  # 最后加入到 df 中

            # print(df.columns)
            uids, iids, labels = df['uid'].tolist(), df['iid'].tolist(), df['label'].tolist()

            for i, uid in enumerate(uids):
                iid, label = iids[i], labels[i]

                if uid not in his_dict:
                    his_dict[uid] = []

                if uid not in neg_dict:
                    neg_dict[uid] = []

                if uid not in mix_dict:
                    mix_dict[uid] = []

                tmp_his = his_dict[uid] if max_his <= 0 else his_dict[uid][-max_his:]
                tmp_neg = neg_dict[uid] if max_his <= 0 else neg_dict[uid][-max_his:]
                # 可以考虑是否改成 [-max_his * 2:]
                tmp_mix = mix_dict[uid] if max_his <= 0 else mix_dict[uid][-max_his:]

                # 去除 [] 第一个元素是 ‘’， history中的元素是 str 类型
                history.append(str(tmp_his).replace(' ', '')[1:-1])
                neg_history.append(str(tmp_neg).replace(' ', '')[1:-1])
                mix_history.append(str(tmp_mix).replace(' ', '')[1:-1])

                if label <= 0:
                    neg_dict[uid].append(iid)
                    mix_dict[uid].append(-iid)
                elif label > 0:
                    his_dict[uid].append(iid)
                    mix_dict[uid].append(iid)

                # if label <= 0:
                #     his_dict[uid].append(-iid)
                # else:
                #     his_dict[uid].append(iid)
            df['history'] = history #只保存 positive samples
            df['history_neg'] = neg_history #只保存 negative samples
            df['history_mix'] = mix_history #保存 positive and negative samples

    def load_info(self):
        max_dict, min_dict = {}, {}
        for df in [self.train_df, self.validation_df, self.test_df]:
            for c in df.columns:
                # print(c)
                if c not in max_dict:
                    max_dict[c] = df[c].max()
                else:
                    max_dict[c] = max(df[c].max(), max_dict[c])

                if c not in min_dict:
                    min_dict[c] = df[c].min()
                else:
                    min_dict[c] = min(df[c].min(), min_dict[c])

        self.column_max = max_dict
        self.column_min = min_dict

        self.user_num, self.item_num = 0, 0
        if 'uid' in self.column_max:
            self.user_num = self.column_max['uid'] + 1
        if 'iid' in self.column_max:
            self.item_num = self.column_max['iid'] + 1

    def drop_neg(self):
        self.train_df = self.train_df[self.train_df['label'] > 0].reset_index(drop=True)
        self.validation_df = self.validation_df[self.validation_df['label'] > 0].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['label'] > 0].reset_index(drop=True)

# if __name__ == '__main__':
#     dataloader = DataLoader('./dataset', 'ml100k01-1-5', 'label')