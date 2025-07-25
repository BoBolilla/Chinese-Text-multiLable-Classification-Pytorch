import numpy as np

def multi_class_evaluate(y_true,y_pred,max_k):
    acc_list,precision_list,recall_list,f1_list = [],[],[],[]
    for k in range(1,max_k + 1):
        acc_list.append(accuracy_k(y_true,y_pred,k))
        precision_list.append(precision_k(y_true,y_pred,k))
        recall_list.append(recall_k(y_true,y_pred,k))
        f1_list.append(F1_k(y_true,y_pred,k))
    return {'acc':acc_list,'precision':precision_list,'recall':recall_list,'f1':f1_list}

def precision_k(y_true, y_pred, k):
    '''
    Precision@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    y_pred = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        y_pred[i][rank_mat[i, :-k]] = 0
    y_pred[y_pred != 0] = 1
    mat = np.multiply(y_pred, y_true)
    num = np.sum(mat, axis=1)
    precision = np.mean(num / k)
    return precision


def recall_k(y_true, y_pred, k):
    '''
    Recall@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    # print(y_true.shape)
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    all_num = np.sum(y_true, axis=1)  # 所有的正类
    y_pred = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        y_pred[i][rank_mat[i, :-k]] = 0
    y_pred[y_pred != 0] = 1
    mat = np.multiply(y_pred, y_true)
    num = np.sum(mat, axis=1)
    recall = np.mean(num / all_num)
    return recall


def F1_k(y_true, y_pred, k):
    '''
    F1@k
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    p_k = precision_k(y_true, y_pred, k)
    r_k = recall_k(y_true, y_pred, k)
    if p_k + r_k == 0:
        return 0.0
    return (2 * p_k * r_k) / (p_k + r_k)


def Ndcg_k(y_true, y_pred, k):
    '''
    自定义评价指标
    @param: y_true: shape=(None, None)
    @param: y_pred: shape=(None, num_classes)
    @return: (5, 1)
    '''
    top_k = 5
    # CAST YOUR TENSOR & CONVERT IT TO NUMPY ARRAY
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    res = np.zeros((top_k, 1))
    rank_mat = np.argsort(y_pred)
    label_count = np.sum(y_true, axis=1)

    for m in range(top_k):
        y_mat = np.copy(y_true)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-k]] = 0
            for j in range(k):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)[k]


def get_factor(label_count, k):
    res = []
    for i in range(len(label_count)):
        n = int(min(label_count[i], k))
        f = 0.0
        for j in range(1, n+1):
            f += 1/np.log(j+1)
        res.append(f)
    return np.array(res)


def accuracy_k(y_true, y_pred, k):
    """
    Computes the top-k accuracy. The score is calculated by considering if the true label is within the top-k scores.

    :param y_true: 2D list or numpy array of shape [n_samples, n_labels]
    :param y_score: 2D list or numpy array of shape [n_samples, n_labels], confidence scores for each label
    :param k: int, the number of top elements to look at for computing the accuracy
    :return: float, top-k accuracy
    """
    rank_mat = np.argsort(y_pred)
    backup = np.copy(y_pred)
    y_pred = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        y_pred[i][rank_mat[i, :-k]] = 0
    y_pred[y_pred != 0] = 1
    correct = np.sum(np.any(y_pred * y_true, axis=1))
    accuracies = correct / y_true.shape[0] # 正确预测的总数 / 总样本数
    return accuracies