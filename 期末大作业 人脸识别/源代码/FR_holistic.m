close all;
clear all;
clc;

% 载入训练集
load ORL_trainset;
[dim, trainnum] = size(train_data); % dim为样本维数，trainnum为训练集样本数
classnum = length(unique(train_label)); % 类别数
trainnum_eachclass = trainnum / classnum; % 每类目标训练样本数

% 载入测试集
load ORL_testset;
testnum = size(test_data, 2); % 测试集样本数
testnum_eachclass = testnum / classnum; % 每类目标测试样本数

% 载入测试集标签
load ORL_testlabel; 

%------------------------- 数据标准化或归一化 -------------------------%
%本部分为可选项,经过验证,在OLR数据集中,对数据进行预处理提升并不明显
% 标准化
%train_data = zscore(train_data);
%test_data = zscore(test_data);

% 归一化
%train_data = normalize(train_data);
%test_data = normalize(test_data);
%------------------------- 数据标准化或归一化 -------------------------%

%---------------------------- 训练分类器并测试 -------------------------%
% 训练SVM分类器并测试
svm_predicted_labels = svm(train_data, train_label, test_data);

% 训练KNN分类器并测试
k = 1; % 设定KNN算法中的K值
knn_predicted_labels = knn(train_data, train_label, test_data, k);
%---------------------------- 训练分类器并测试 -------------------------%

%----------------------------计算分类精度----------------------------%
svm_accuracy = calculate_accuracy(svm_predicted_labels, label_truth);
knn_accuracy = calculate_accuracy(knn_predicted_labels, label_truth);

svm_accuracy = 100 * svm_accuracy;         % 将分类精度以百分数的形式输出
knn_accuracy = 100 * knn_accuracy;         % 将分类精度以百分数的形式输出

fprintf('使用SVM分类器在测试集上的分类精度为 %.2f%%\n', svm_accuracy);
fprintf('使用KNN分类器在测试集上的分类精度为 %.2f%%\n', knn_accuracy);
%----------------------------计算分类精度----------------------------%



