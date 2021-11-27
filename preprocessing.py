import csv
from tqdm import trange
import sys
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
from imblearn.over_sampling import SMOTE
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import time

# 生成文件列表，其中前7个为训练集，后两个为测试集
# 只有第一个文件含有标签头，2601列，2600维数据，最后一列为标签映射关系
file = "D:/dataset_all/data_mining_star/sets_v1_0.csv"
file_label = "D:/dataset_all/data_mining_star/label0.csv"
file_list = [file[:-5] + str(i) + '.csv' for i in range(9)]
file_label_list = [file_label[:-5] + str(i) + '.csv' for i in range(9)]


def get_data(filepath):
    data = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data


# 检查是否有缺失值
def detect_missing(file_list):
    for i in trange(len(file_list)):
        data = []
        with open(file_list[i], 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data.append(row)
        for row in data:
            for item in row:
                if item == '':
                    print('error missing value')
                    sys.exit(0)


# 去除异常点数据--去除每个维度的异常点--超过均值加减三倍标准差即为异常点
def eliminate_outliers(num):
    global file_list
    global file_label_list
    print('开始对第', num, '个文件进行处理')
    flag = []
    data = get_data(file_list[num])
    data = np.array(data)
    data = data[:, :-1].astype(np.float)
    # 对于每个样本数据判断其是否异常点
    for i in trange(len(data[0])):
        mean = np.mean(data[:, i].reshape(-1))
        std = np.std(data[:, i].reshape(-1))
        for j in range(len(data[:, 0])):
            if data[j, i] < mean - 3 * std or data[j, i] > mean + 3 * std:
                flag.append(j)
    print('共找到', len(flag), '个异常点')
    flag = list(set(flag))
    print('去掉重复后还剩', len(flag), '个异常点')
    label = get_data(file_label_list[num])
    label = np.array(label)
    label_delete = label[flag]
    print('这些异常点中的类别分布情况为：', Counter(label_delete.reshape(-1)))
    label = np.delete(label, np.s_[flag], axis=0)
    data = np.delete(data, np.s_[flag], axis=0)
    # 直接删掉这些点并保存成新文件
    with open(file_label_list[num], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(label)
    with open(file_list[num], 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


# 观察数据分布情况
def distribution_visualization():
    global file_list
    data = get_data(file_list[0])
    data = np.array(data)
    demension_0 = data[:, 1].reshape(-1)
    demension_0 = demension_0.astype(np.float)
    demension_0 = np.sort(demension_0)
    # 绘制概率密度图
    sns.displot(demension_0, kde=True, color='royalblue')
    plt.xlabel("demension's pdf")
    plt.show()
    # 绘制散点图
    x = [i for i in range(len(demension_0))]
    plt.scatter(x, demension_0)
    plt.xlabel('index')
    plt.ylabel('values')
    plt.show()

# 观察类别之间的相似度--使用余弦距离
def cosine_with_classes():
    global file_list
    data = get_data(file_list[1])
    label = [10, 27, 26]  # star,qso,galaxy
    data = np.array(data)
    # data = data[[0,12,19],:-1]
    data_star = data[[i for i in range(10)], :-1].astype(np.float)
    data_qso = data[[12, 27, 68, 72, 78, 117, 194, 198, 208, 209], :-1].astype(np.float)
    data_galaxy = data[[19, 20, 26, 37, 38, 41, 67, 73, 75, 100], :-1].astype(np.float)
    # data = data.astype(np.float)
    data = np.concatenate((np.mean(data_star, axis=0).reshape(1, -1), np.mean(data_qso, axis=0).reshape(1, -1)), axis=0)
    data = np.concatenate((data, np.mean(data_galaxy, axis=0).reshape(1, -1)), axis=0)
    Xlabel = ['star', 'qso', 'galaxy']
    Ylabel = ['star', 'qso', 'galaxy']
    cos = cosine_similarity(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(Ylabel)))
    ax.set_yticklabels(Ylabel)
    ax.set_xticks(range(len(Xlabel)))
    ax.set_xticklabels(Xlabel)
    im = ax.imshow(cos, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("Classes_HotMap_10_samples")
    plt.show()

# 对数据进行标准化
# 考虑易实施性这里采用分别对每个文件进行标准化和归一化
def normalization():
    global file_list
    # 使用flag_array保存每一维特征的均值、方差、最小值和最大值
    flag_array = np.zeros((4, 2600))
    for i in trange(9):
        data = np.array(get_data(file_list[i]))
        data = data.astype(np.float64)
        for j in range(2600):
            tmp = data[:,j].reshape(-1)
            flag_array[0, j] += np.mean(tmp)
            flag_array[1, j] += np.std(tmp)
            current_min = np.min(tmp)
            current_max = np.max(tmp)
            if current_min < flag_array[2,j]:
                flag_array[2,j] = current_min
            if current_max > flag_array[3,j]:
                flag_array[3,j] = current_max
    flag_array = np.concatenate((flag_array[:2,:] / 10, flag_array[2:,:]),axis=0)
    # 将min和max转成归一化后的min和max
    for i in range(2600):
        flag_array[2, i] = (flag_array[2, i] - flag_array[0, i]) / flag_array[1, i]
        flag_array[3, i] = (flag_array[3, i] - flag_array[0, i]) / flag_array[1, i]
    # 下面直接逐文件进行标准化归一化操作
    for i in trange(9):
        data = np.array(get_data(file_list[i]))
        data = data.astype(np.float64)
        for j in range(len(data[:,0])):
            for k in range(2600):
                data[j,k] = ((data[j,k] - flag_array[0,k]) / flag_array[1,k] - flag_array[2,k]) / (flag_array[3,k] - flag_array[2,k])
        with open(file_list[i],'w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

# 过采样
def OverSample_for_DNN():
    global file_list
    global file_label_list
    for j in range(9):
        data = np.array(get_data(file_list[j]))
        data_resampled = data.astype(np.float64)
        label = np.array(get_data(file_label_list[j]))
        label_resampled = label.reshape(1,-1)[0]
        # data_resampled, label_resampled = SMOTE().fit_resample(data,label)
        path_train = './data_unbalanced/train/'
        path_val = './data_unbalanced/val/'
        for i in trange(len(data_resampled)):
            if j > 6:
                path = path_val + label_resampled[i] + '/' + label_resampled[i] + str(i+j*20000) + '.png'
            else:
                path = path_train + label_resampled[i] + '/' + label_resampled[i] + str(i+j*20000) + '.png'
            tmp = data_resampled[i].copy()
            tmp *= 255
            tmp = np.concatenate((tmp,np.array([0])),axis=0)
            im = Image.fromarray(tmp.reshape(51,51))
            im.convert('L').save(path)

# 降维--tSNE或PCA
def demension_reduction(method):
    global file_list, pca
    global file_label_list
    for j in trange(9):
        data = np.array(get_data(file_list[j]))
        data = data.astype(np.float64)
        label = np.array(get_data(file_label_list[j])).reshape(1,-1)[0]
        if method == 'PCA' or 'tSNE':
            if j == 0:
                pca = PCA(n_components=200)
                pca.fit(data)
            data_d_r = pca.transform(data)
            if method == 'tSNE':
                start = time.time()
                tsne = TSNE(n_components=3,n_jobs=5)
                data_d_r = tsne.fit_transform(data_d_r)
                end = time.time()
                print('tSNE consumes:',end-start)
            with open(file_list[j][:-4] + '_lower_pca.csv','w',newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(data_d_r)
                # X_train = data_d_r[:int(0.8 * len(data_d_r))]
                # Y_train = label[:int(0.8 * len(data_d_r))]
                # X_test = data_d_r[int(0.8 * len(data_d_r)):]
                # Y_test = label[int(0.8 * len(data_d_r)):]
                # rfc = RandomForestClassifier(random_state=0, class_weight='balanced')
                # rfc.fit(X_train, Y_train)
                # pre = rfc.predict(X_test)
                # print('tSNE output:')
                # print(accuracy_score(pre, Y_test))
                # plt.cla()
                # # 降到二维了
                # X, Y = data_d_r[:, 0], data_d_r[:, 1]
                # for i in range(len(label)):
                #     if label[i] == 'star':
                #         label[i] = 0
                #     elif label[i] == 'qso':
                #         label[i] = 2
                #     else:
                #         label[i] = 1
                # for x, y, s in zip(X, Y, label):
                #     c = cm.rainbow(int(255 / 4 * int(s)))
                #     plt.text(x, y, s, backgroundcolor=c)
                # plt.xlim(X.min(), X.max())
                # plt.ylim(Y.min(), Y.max())
                # plt.title('tSNE output')
                # plt.show()
        else:
            print('PCA or tSNE')

# 处理测试集输入
def test_preprocess(test_file_path):
    test_data = np.array(get_data(test_file_path))
    flag_array = np.array(get_data("D:/python_project/data_mining_stars/flag.csv")).astype(np.float64)
    # 下面直接逐文件进行标准化归一化操作
    test_data = test_data.astype(np.float64)
    for j in range(len(test_data[:, 0])):
        for k in range(2600):
            test_data[j, k] = ((test_data[j, k] - flag_array[0, k]) / flag_array[1, k] - flag_array[2, k]) / (
                        flag_array[3, k] - flag_array[2, k])
    with open('test_data_fine.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(test_data)



if __name__ == '__main__':
    # demension_reduction('PCA')
    # demension_reduction('PCA')
    # OverSample_for_DNN()
    test_preprocess("D:/dataset_all/data_mining_star/sets_small.csv")