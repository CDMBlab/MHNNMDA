import time
import gc
import torch
import random
from datapro import EdgeDataset
from model_tri import SuperedgeLearn
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math

import os


def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)

    return metric


def caculate_metrics(pre_score, real_score):
    y_pre = pre_score
    y_true = real_score

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)

    return metric_result


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))


def train_test(simData, train_data, param, state):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    epo_metric = []
    valid_metric = []

    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']
    m_d_matrix = train_data['true_md']
    md_class = train_data['md_class']

    m_d_matrix[tuple(test_edges.T)] = 0
    md_class[tuple(test_edges.T)] = 0

    kfolds = param.kfold
    torch.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []

        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        for i in range(kfolds):
            a = i+1
            model = SuperedgeLearn(param)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
            # model.load_state_dict(torch.load('./cross_valid_example/fold_{}.pkl'.format(a)))
            # print(f'################Fold {i + 1} of {kfolds}################')
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            trainEdges = EdgeDataset(edges_train, labels_train)
            validEdges = EdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            m_d_class = md_class.copy()
            m_d_class[tuple(edges_valid.T)] = 0.0
            md_m = torch.from_numpy(m_d_class).cuda()

            print("-----training-----")

            for e in range(param.epoch):
                running_loss = 0.0
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()
                for i, item in enumerate(trainLoader):
                    data, label = item
                    trainData = data.cuda()
                    trainLabel = label.cuda()
                    pre_score = model(simData, md_m, trainData)
                    train_loss = torch.nn.BCELoss()
                    loss = train_loss(pre_score, trainLabel)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += loss.item()
                    print(f"After batch {i+1}:loss={loss:.3f};",end='\n')

                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score,batch_score)
                    epo_label = np.append(epo_label,label.numpy())
                end=time.time()
                print('Time:%.2f\n'%(end-start))
                train_result = get_metrics(epo_score,epo_label)


            valid_score, valid_label = [], []
            model.eval()
            with torch.no_grad():
                print("-----validing-----")
                # val_loss = 0.0
                for i, item in enumerate(validLoader):
                    data, label = item
                    validData = data.cuda()
                    # validLabel = label.cuda()
                    pre_score = model(simData, md_m, validData)

                    val_batch_score = pre_score.detach().cpu().numpy()
                    valid_score = np.append(valid_score, val_batch_score)
                    valid_label = np.append(valid_label, label.numpy())

                    # torch.save(model.state_dict(), "./fold_{}.pkl".format(a))
                valid_result = get_metrics(valid_score, valid_label)
                valid_metric.append(valid_result)
                gc.collect()
                torch.cuda.empty_cache()
        print(np.array(valid_metric))
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)


    else:
        test_score, test_label = [], []
        testEdges = EdgeDataset(test_edges, test_labels)
        testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)

        md_ma = torch.from_numpy(md_class).cuda()
        model = SuperedgeLearn(param)
        model.load_state_dict(torch.load('./test_example/test.pkl'))
        model.cuda()
        model.eval()
        with torch.no_grad():
            start = time.time()
            for i, item in enumerate(testLoader):
                data, label = item
                testData = data.cuda()
                pre_score = model(simData, md_ma, testData)
                batch_score = pre_score.cpu().detach().numpy()
                test_score = np.append(test_score, batch_score)
                test_label = np.append(test_label, label.numpy())
            end = time.time()
            print('Time：%.2f \n' % (end - start))
            metrics = get_metrics(test_score, test_label)

    return cv_metric
    # For testing
    # return metrics
