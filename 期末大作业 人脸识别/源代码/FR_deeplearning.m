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
% 本部分为可选项，根据需要选择是否进行数据标准化或归一化处理
% 标准化
%train_data = zscore(train_data);
%test_data = zscore(test_data);

% 归一化
train_data = normalize(train_data);
test_data = normalize(test_data);
%------------------------- 数据标准化或归一化 -------------------------%

%---------------------------- 训练卷机神经网络并测试 -------------------------%
% 训练卷积神经网络并测试
cnn_predicted_labels = myCNN(train_data, train_label, test_data, trainnum);
%---------------------------- 训练卷机神经网络并测试 -------------------------%

%----------------------------计算分类精度----------------------------%
cnn_accuracy = calculate_accuracy(cnn_predicted_labels, label_truth);
cnn_accuracy = 100 * cnn_accuracy; % 将分类精度以百分数的形式输出
fprintf('使用自建卷积神经网络在测试集上的分类精度为 %.2f%%\n', cnn_accuracy);
%----------------------------计算分类精度----------------------------%

