# -*- coding: utf-8 -*-
import logging
import argparse
import math
import sys
import random
import numpy
from sklearn import metrics
from time import strftime, localtime
from transformers import BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# default sample extraction method - Interacting Polarity(I_P)
from data_utils_I_P import Tokenizer4Bert, ABSADataset, BalancedBatchSampler
from models import PSI
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Instructor:
    def __init__(self, opt):
        self.opt = opt  # load the Hyper Parameters
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)  # load Bert-Token (default seq_len = 85)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)  # load Bert_Model(default Base-Bert)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        print("self.trainset:", self.trainset)
        print("self.testset:", self.testset)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        list_positive = []
        list_neutral = []
        list_negative = []

        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            rank_criterion = nn.MarginRankingLoss(margin=0.05)
            softmax_layer = nn.Softmax(dim=1).to(device)
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(device) for col in self.opt.inputs_cols]
                targets = batch['polarity'].to(device)
                logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = self.model(inputs, targets,
                                                                                                    flag='train')
                batch_size = logit1_self.shape[0]
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)

                self_logits = torch.zeros(2 * batch_size, 3).to(device)
                other_logits = torch.zeros(2 * batch_size, 3).to(device)
                self_logits[:batch_size] = logit1_self
                self_logits[batch_size:] = logit2_self
                other_logits[:batch_size] = logit1_other
                other_logits[batch_size:] = logit2_other

                # compute loss
                logits = torch.cat([self_logits, other_logits], dim=0)
                targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
                softmax_loss = criterion(logits, targets)

                self_scores = softmax_layer(self_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                         torch.cat([labels1, labels2], dim=0)]
                other_scores = softmax_layer(other_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                           torch.cat([labels1, labels2], dim=0)]
                flag = torch.ones([2 * batch_size, ]).to(device)
                rank_loss = rank_criterion(self_scores, other_scores, flag)

                loss = softmax_loss + rank_loss

                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(logits, -1) == targets).sum().item()
                n_total += len(logits)
                loss_total += loss.item() * len(logits)

            val_acc, val_f1, acc_positive, acc_neutral, acc_negative = self._evaluate_acc_f1(val_data_loader)
            list_positive.append(acc_positive)
            list_neutral.append(acc_neutral)
            list_negative.append(acc_negative)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))

                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return max_val_acc, max_val_f1

    def _evaluate_acc_f1(self, data_loader):
        labels_set = [0, 1, 2]
        n_correct, n_total = 0, 0
        n_correct_positive_sum, n_correct_neutral_sum, n_correct_negative_sum = 0, 0, 0
        n_total_positive, n_total_neutral, n_total_negative = 0, 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                n_correct_positive, n_correct_neutral, n_correct_negative = 0, 0, 0
                t_inputs = [t_batch[col].to(device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(device)
                t_outputs = self.model(t_inputs, targets=None, flag='val')
                t_targets_to_indices = {label: numpy.where(t_targets.cpu().numpy() == label)[0]  # 返回label下标索引数组
                                        for label in labels_set}

                if len(t_targets_to_indices[0]):
                    n_correct_positive = (torch.argmax(t_outputs[t_targets_to_indices[0]], -1) == t_targets[
                        t_targets_to_indices[0]]).sum().item()
                if len(t_targets_to_indices[1]):
                    n_correct_neutral = (torch.argmax(t_outputs[t_targets_to_indices[1]], -1) == t_targets[
                        t_targets_to_indices[1]]).sum().item()
                if len(t_targets_to_indices[2]):
                    n_correct_negative = (torch.argmax(t_outputs[t_targets_to_indices[2]], -1) == t_targets[
                        t_targets_to_indices[2]]).sum().item()

                n_correct += n_correct_positive
                n_correct += n_correct_neutral
                n_correct += n_correct_negative
                n_correct_positive_sum += n_correct_positive
                n_correct_neutral_sum += n_correct_neutral
                n_correct_negative_sum += n_correct_negative

                n_total += len(t_outputs)
                n_total_positive += len(t_targets_to_indices[0])
                n_total_neutral += len(t_targets_to_indices[1])
                n_total_negative += len(t_targets_to_indices[2])

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        acc_positive = n_correct_positive_sum / n_total_positive
        acc_neutral = n_correct_neutral_sum / n_total_neutral
        acc_negative = n_correct_negative_sum / n_total_negative

        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1, acc_positive, acc_neutral, acc_negative

    def run(self):
        # define loss
        criterion = nn.CrossEntropyLoss()
        # load bert params
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # initial new layer params
        self._reset_params()
        # define optimizer
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        # define data extraction method
        train_sampler = BalancedBatchSampler(self.trainset, self.opt.n_classes, self.opt.n_samples)
        train_data_loader = DataLoader(dataset=self.trainset,
                                       batch_sampler=train_sampler)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        # (optional)
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        # train and validate
        max_val_acc, max_val_f1 = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(max_val_acc, max_val_f1))
        return max_val_acc, max_val_f1


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PSI', type=str)
    parser.add_argument('--dataset', default='res14', type=str, help='res14, res15, res16, laptop16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are used when constructing a batch of sentence pairs for PSI
    parser.add_argument('--n_classes', default=3, type=int,
                        help='the number of classes')
    parser.add_argument('--n_samples', default=4, type=int,
                        help='the number of samples per class')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'PSI': PSI,

    }
    dataset_files = {
        'res14': {

            'train': './datasets/semeval14/Res14_train.seg',
            'test': './datasets/semeval14/Res14_test.seg'
        },
        'res15': {
            'train': './datasets/semeval15/Res15_train.seg',
            'test': './datasets/semeval15/Res15_test.seg'
        },
        'res16': {
            'train': './datasets/semeval16/Res16_train.seg',
            'test': './datasets/semeval16/Res16_test.seg'
        },
        'laptop15': {
            'train': './datasets/semeval15/Lap15_train.seg',
            'test': './datasets/semeval15/Lap15_test.seg'
        },
        'laptop16': {
            'train': './datasets/semeval16/Lap16_train.seg',
            'test': './datasets/semeval16/Lap16_test.seg'
        }
    }
    input_colses = {
        'PSI': ['concat_bert_indices', 'concat_segments_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # Build and run the model
    ins = Instructor(opt)
    test_acc, test_f1 = ins.run()
    return test_acc, test_f1


if __name__ == '__main__':
    main()
