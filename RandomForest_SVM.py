import csv
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import joblib

file = "D:/dataset_all/data_mining_star/sets_v1_0.csv"
file_label = "D:/dataset_all/data_mining_star/label0.csv"
file_list = [file[:-5] + str(i) + '_lower_pca.csv' for i in range(9)]
file_label_list = [file_label[:-5] + str(i) + '.csv' for i in range(9)]


def get_whole_data(flag):
    global file_list
    global file_label_list
    data = []
    if flag == 'train':
        flag = 7
        num = 0
    else:
        flag = 2
        num = 7
    for i in range(flag):
        with open(file_list[i + num], 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data.append(row)
    data = np.array(data)
    data = data.astype(np.float64)
    label = []
    for i in range(flag):
        with open(file_label_list[i + num], 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                label.append(row)
    label = np.array(label).reshape(1, -1)[0]
    return data, label


def pre(data, label):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(data)
    label_set = list(set(label.tolist()))
    for i in range(len(label)):
        if label[i] == 'galaxy':
            label[i] = 0
        elif label[i] == 'qso':
            label[i] = 1
        else:
            label[i] = 2
    return data, label


def RF(data, label, test_data, test_label):
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    rf = RandomForestClassifier(random_state=0, class_weight='balanced')
    scores = cross_validate(rf, data, label, scoring=scoring, cv=5, return_train_score=True)
    print('precision_weighted score=', scores['test_precision_macro'])
    print('recall_weighted score=', scores['test_recall_macro'])
    try:
        print('f1_weighted score=', scores['test_f1_macro'])
    except:
        pass
    try:
        print('precision_weighted mean=', scores['test_precision_macro'].mean())
        print('recall_weighted mean=', scores['test_recall_macro'].mean())
        print('f1_weighted mean=', scores['test_f1_macro'].mean())
    except:
        print('no mean')
    rf.fit(data, label)
    pre = rf.predict(test_data)
    print('accuarcy:',accuracy_score(pre,test_label))
    m = confusion_matrix(test_label, pre)
    print('混淆矩阵为：', m, sep='\n')
    joblib.dump(rf,"./model/RF.m")

def SVM(data, label, test_data, test_label):
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    svc = SVC(class_weight='balanced')
    scores = cross_validate(svc, data, label, scoring=scoring, cv=5, return_train_score=True)
    print('precision_weighted score=', scores['test_precision_macro'])
    print('recall_weighted score=', scores['test_recall_macro'])
    try:
        print('f1_weighted score=', scores['test_f1_macro'])
    except:
        pass
    try:
        print('precision_weighted mean=', scores['test_precision_macro'].mean())
        print('recall_weighted mean=', scores['test_recall_macro'].mean())
        print('f1_weighted mean=', scores['test_f1_macro'].mean())
    except:
        print('no mean')
    svc.fit(data, label)
    pre = svc.predict(test_data)
    print('accuarcy:',accuracy_score(pre,test_label))
    m = confusion_matrix(test_label, pre)
    print('混淆矩阵为：', m, sep='\n')
    joblib.dump(svc, "./model/SVM.m")


def RF_test(test_data,test_label):
    rf = joblib.load('./model/RF.m')
    pre = rf.predict(test_data)
    print('accuarcy:', accuracy_score(pre, test_label))
    m = confusion_matrix(test_label, pre)
    print('混淆矩阵为：', m, sep='\n')
    P_1 = m[0, 0] / np.sum(m[0])
    P_2 = m[1, 1] / np.sum(m[1])
    P_3 = m[2, 2] / np.sum(m[2])
    print('macro_P值为：', (P_1 + P_2 + P_3) / 3)
    R_1 = m[0, 0] / np.sum(m[:, 0])
    R_2 = m[1, 1] / np.sum(m[:, 1])
    R_3 = m[2, 2] / np.sum(m[:, 2])
    print('macro_R值为：', (R_1 + R_2 + R_3) / 3)
    F_1 = 2 * P_1 * R_1 / (P_1 + R_1)
    F_2 = 2 * P_2 * R_2 / (P_2 + R_2)
    F_3 = 2 * P_3 * R_3 / (P_3 + R_3)
    print('macro_f1值为：',(F_1 + F_2 + F_3) / 3)

def RF_test_for_one_sample(i):
    test_data, test_label = get_whole_data('test')
    test_data, test_label = pre(test_data, test_label)
    rf = joblib.load('./model/RF.m')
    for j in range(5):
        tmp = rf.predict(test_data[i+j].reshape(1,-1))
        print('真实标签为：',test_label[i+j])
        print('预测标签为：',tmp)

def SVM_test(test_data,test_label):
    rf = joblib.load('./model/SVM.m')
    pre = rf.predict(test_data)
    print('accuarcy:', accuracy_score(pre, test_label))
    m = confusion_matrix(test_label, pre)
    print('混淆矩阵为：', m, sep='\n')

if __name__ == '__main__':
    # data, label = get_whole_data('train')
    # print(data.shape)
    # print(label.shape)
    # data, label = pre(data, label)
    # test_data, test_label = get_whole_data('test')
    # print(test_data.shape)
    # print(test_label.shape)
    # test_data, test_label = pre(test_data, test_label)
    #
    # RF_new(data, label, test_data, test_label)

    # 测试用
    test_data, test_label = get_whole_data('test')
    print(test_data.shape)
    print(test_label.shape)
    test_data, test_label = pre(test_data, test_label)

    RF_test(test_data,test_label)

    # # 测试单样本用
    # RF_test_for_one_sample(170)

