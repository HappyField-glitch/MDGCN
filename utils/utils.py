import numpy as np
import torch
import scipy.io as sio
import yaml
import pickle
import time, datetime
import logging
import os
from pathlib import Path
import shutil


def cal_ap(y_pred,y_true):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    # y_pred = y_pred.transpose(1, 0)
    # y_true = y_true.transpose(1, 0)
    ap = torch.zeros(y_pred.size(0))   ## 总样本个数
    # compute average precision for each class
    for k in range(y_pred.size(0)):
        # sort scores
        scores = y_pred[k,:].reshape([1,-1])
        targets = y_true[k,:].reshape([1,-1])
        # compute average precision
        # ap[k] = average_precision(scores, targets, difficult_examples)
        ap[k] = avg_p(scores, targets)
    return ap


def avg_p(output, target):

    sorted, indices = torch.sort(output, dim=1, descending=True)
    tp = 0
    s = 0
    for i in range(target.size(1)):
        idx = indices[0,i]
        if target[0,idx] == 1:
            tp = tp + 1
            pre = tp / (i+1)
            s = s + pre
    if tp == 0:
        AP = 0
    else:
        AP = s/tp
    return AP


'''record configurations'''


class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = self.job_dir / 'config.txt'
        # if not os.path.exists(config_dir):
        if args.use_resume:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')


def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    # file = open(file_path, 'w')
    # file.close()
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def save_checkpoint(epoch, state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'best_model.pth.tar')
        shutil.copyfile(filename, best_filename)


def cal_one_error(output, target):
    output = output.transpose(1, 0)
    target = target.transpose(1, 0)
    error_num = 0
    total_num = output.size(1)
    for i in range(total_num):
        _,indice = torch.max(output[:,i].reshape(1,-1),dim=1)
        if target[indice,i] != 1:
            error_num += 1
    one_error = error_num/total_num
    return torch.from_numpy(np.array(one_error)).to(output.device)


def cal_coverage(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    cover = 0
    for i in range(num_instance):
        Label = []
        not_Label = []
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar == torch.ones([1,num_class]).to(output.device)))
        for j in range(num_class):  # 遍历类别
            if(temp_tar[0,j]==1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp_out = output[i,:]
        _,inde = torch.sort(temp_out)  # 升序
        inde = inde.cpu().numpy().tolist()
        temp_min = num_class
        for m in range(Label_size):
            loc = inde.index(Label[m])
            if loc<temp_min:
                temp_min = loc
        cover = cover + (num_class-temp_min)

    cover_result = (cover/num_instance)-1
    return torch.from_numpy(np.array(cover_result)).to(output.device)


def cal_RankingLoss(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    rankloss = 0
    for i in range(num_instance):
        Label = []  # 存储正标签的索引
        not_Label = []  # 存储负标签的索引
        temp_tar = target[i,:].reshape(1,num_class)
        Label_size = sum(sum(temp_tar ==torch.ones([1,num_class]).to(output.device)))
        for j in range(num_class):
            if (temp_tar[0, j] == 1):
                Label.append(j)
            else:
                not_Label.append(j)
        temp = 0
        for m in range(Label_size):
            for n in range(num_class-Label_size):   ## 比较每一个正标签和所有负标签的预测概率大小，小于等于次数加1
                if output[i,Label[m]]<=output[i,not_Label[n]]: ## n表示负标签总个数
                    temp += 1
        if Label_size==0:
            continue
        else:
            rankloss = rankloss + float(temp) / float((Label_size * (num_class-Label_size)))

    RankingLoss = float(rankloss) / num_instance

    return RankingLoss


def cal_HammingLoss(output, target):
    num_class, num_instance = output.size(1), output.size(0)
    miss_sum = 0
    for i in range(num_instance):
        temp_out = torch.sign(output[i,:])
        temp_out[temp_out==-1] = 0
        miss_pairs = sum(temp_out!=target[i,:])
        miss_sum += miss_pairs
    HammingLoss = float(miss_sum)/(num_class*num_instance)
    # HammingLoss = HammingLoss.data.cpu().numpy()
    # print(HammingLoss)
    return HammingLoss


def load_state_dict(model, ori_state):
    state_dict = model.state_dict()
    tmp = dict()
    for key in state_dict.keys():
        if key in ori_state.keys():
            tmp[key] = ori_state[key]
        else:
            tmp[key] = state_dict[key]
    return tmp


def parse_dicts(args):
    if not isinstance(args, dict):
        return args

    for key, value in args.items():
        if not isinstance(value, dict):
            if not 'dims' in key:
                args[key] = value
            else:
                if key == 'gcn_dims':
                    args[key] = value
                else:
                    args[key] = [int(x) for x in value.split(',')]
        else:
            value = parse_dicts(value)
            args[key] = value
    return args

