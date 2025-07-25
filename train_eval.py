# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import multi_class_evaluate
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            trains = [x.to(config.device) for x in trains]
            labels = labels.to(config.device)
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        # 训练集top@1/2/3评估
        model.eval()
        train_metrics, train_loss = evaluate(config, model, train_iter, top_k=3)
        dev_metrics, dev_loss = evaluate(config, model, dev_iter, top_k=3)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            # last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        print(
            f'Train Loss: {loss.item():.4f}, Val Loss: {dev_loss:.4f}, Time: {time_dif} {improve}')
        for k in range(1, 4):
            print(
                f'  Train Accuracy@{k}: {train_metrics["acc"][k - 1]:.4f}, F1@{k}: {train_metrics["f1"][k - 1]:.4f}, Precision@{k}: {train_metrics["precision"][k - 1]:.4f}, Recall@{k}: {train_metrics["recall"][k - 1]:.4f}')
            print(
                f'  Val   Accuracy@{k}: {dev_metrics["acc"][k - 1]:.4f}, F1@{k}: {dev_metrics["f1"][k - 1]:.4f}, Precision@{k}: {dev_metrics["precision"][k - 1]:.4f}, Recall@{k}: {dev_metrics["recall"][k - 1]:.4f}')

            writer.add_scalar(f"accuracy/train@{k}", train_metrics["acc"][k - 1], total_batch)
            writer.add_scalar(f"accuracy/dev@{k}", dev_metrics["acc"][k - 1], total_batch)
            writer.add_scalar(f"f1/train@{k}", train_metrics["f1"][k - 1], total_batch)
            writer.add_scalar(f"f1/dev@{k}", dev_metrics["f1"][k - 1], total_batch)
            writer.add_scalar(f"precision/train@{k}", train_metrics["precision"][k - 1], total_batch)
            writer.add_scalar(f"precision/dev@{k}", dev_metrics["precision"][k - 1], total_batch)
            writer.add_scalar(f"recall/train@{k}", train_metrics["recall"][k - 1], total_batch)
            writer.add_scalar(f"recall/dev@{k}", dev_metrics["recall"][k - 1], total_batch)
        writer.add_scalar("loss/train", loss.item(), total_batch)
        writer.add_scalar("loss/dev", dev_loss, total_batch)
        model.train()
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_metrics, test_loss = evaluate(config, model, test_iter, test=True, top_k=3)

    print(f'Test Loss: {test_loss:.4f}')
    for k in range(1, 4):
        print(f'Test Top@{k}:')
        print(f'  F1      : {test_metrics["f1"][k - 1]:.4f}')
        print(f'  Precision: {test_metrics["precision"][k - 1]:.4f}')
        print(f'  Recall   : {test_metrics["recall"][k - 1]:.4f}')
        print(f'  Accuracy : {test_metrics["acc"][k - 1]:.4f}')

    print("Time usage:", get_time_dif(start_time))


def evaluate(config, model, data_iter, test=False, top_k=3):
    model.eval()
    loss_total = 0
    criterion = nn.BCEWithLogitsLoss()
    labels_all = []
    outputs_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = [x.to(config.device) for x in texts]
            labels = labels.to(config.device)
            outputs = model(texts)
            loss = criterion(outputs, labels.float())
            loss_total += loss.item()
            labels_all.append(labels.cpu().numpy())
            outputs_all.append(torch.sigmoid(outputs).detach().cpu().numpy())

    labels_all = np.concatenate(labels_all, axis=0)
    outputs_all = np.concatenate(outputs_all, axis=0)
    metrics_result = multi_class_evaluate(labels_all, outputs_all, max_k=top_k)

    return metrics_result, loss_total / len(data_iter)