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

%------------------------- 网格搜索 -------------------------%
% 设置网格搜索的参数范围
pca_dims_range = 1:100; % PCA降维维度的范围
lda_dims_range = 1:39; % LDA降维维度的范围
k_range = 1:5; % KNN中K值的范围

svm_pca_best_accuracy = 0;
svm_pca_best_dims = 0;
svm_lda_best_accuracy = 0;
svm_lda_best_dims = 0;
knn_pca_best_accuracy = 0;
knn_pca_best_dims = 0;
knn_pca_best_k = 0;
knn_lda_best_accuracy = 0;
knn_lda_best_dims = 0;
knn_lda_best_k = 0;

svm_pca_log = [];
svm_lda_log = [];
knn_pca_log = [];
knn_lda_log = [];

% 进行网格搜索
for pca_dims = pca_dims_range
    % 特征提取降维
    [train_pca, test_pca] = pca(train_data, test_data, pca_dims);
    % 训练SVM分类器并测试
    svm_pca_predicted_labels = svm(train_pca, train_label, test_pca);
    % 计算分类精度
    svm_pca_accuracy = calculate_accuracy(svm_pca_predicted_labels, label_truth);
    
    svm_pca_log = [svm_pca_log; pca_dims svm_pca_accuracy];
    
    if svm_pca_accuracy > svm_pca_best_accuracy
        svm_pca_best_accuracy = svm_pca_accuracy;
        svm_pca_best_dims = pca_dims;
    end
    fprintf('SVM分类器中PCA维度为 %d 的准确度为 %.2f%%\n', pca_dims, svm_pca_accuracy*100);
end
fprintf('SVM分类器中最优的PCA维度为 %d，对应准确度为 %.2f%%\n', svm_pca_best_dims, svm_pca_best_accuracy*100);
% 保存日志
log_folder = './Grid_search_log';
save(fullfile(log_folder, 'svm_pca_log.mat'), 'svm_pca_log');
for lda_dims = lda_dims_range
    % 特征提取降维
    [train_lda, test_lda] = lda(train_data, train_label, test_data, lda_dims);
    % 训练SVM分类器并测试
    svm_lda_predicted_labels = svm(train_lda, train_label, test_lda);
    % 计算分类精度
    svm_lda_accuracy = calculate_accuracy(svm_lda_predicted_labels, label_truth);
    
    svm_lda_log = [svm_lda_log; lda_dims svm_lda_accuracy];
    
    if svm_lda_accuracy > svm_lda_best_accuracy
        svm_lda_best_accuracy = svm_lda_accuracy;
        svm_lda_best_dims = lda_dims;
    end
    fprintf('SVM分类器中LDA维度为 %d 的准确度为 %.2f%%\n', lda_dims, svm_lda_accuracy*100);
end
fprintf('SVM分类器中最优的LDA维度为 %d，对应准确度为 %.2f%%\n', svm_lda_best_dims, svm_lda_best_accuracy*100);

for pca_dims = pca_dims_range
    % 特征提取降维
    [train_pca, test_pca] = pca(train_data, test_data, pca_dims);
    for k = k_range
        % 训练KNN分类器并测试
        knn_pca_predicted_labels = knn(train_pca, train_label, test_pca, k);
        % 计算分类精度
        knn_pca_accuracy = calculate_accuracy(knn_pca_predicted_labels, label_truth);
        
        knn_pca_log = [knn_pca_log; pca_dims k knn_pca_accuracy];
        
        if knn_pca_accuracy > knn_pca_best_accuracy
            knn_pca_best_accuracy = knn_pca_accuracy;
            knn_pca_best_dims = pca_dims;
            knn_pca_best_k = k;
        end
        fprintf('KNN分类器中PCA维度为 %d, K值为 %d 的准确度为 %.2f%%\n', pca_dims, k, knn_pca_accuracy*100);
    end
end
fprintf('KNN分类器中最优的PCA维度为 %d, K值为 %d，对应准确度为 %.2f%%\n', knn_pca_best_dims, knn_pca_best_k, knn_pca_best_accuracy*100);

for lda_dims = lda_dims_range
    % 特征提取降维
    [train_lda, test_lda] = lda(train_data, train_label, test_data, lda_dims);
    for k = k_range
        % 训练KNN分类器并测试
        knn_lda_predicted_labels = knn(train_lda, train_label, test_lda, k);
        % 计算分类精度
        knn_lda_accuracy = calculate_accuracy(knn_lda_predicted_labels, label_truth);
        
        knn_lda_log = [knn_lda_log; lda_dims k knn_lda_accuracy];
        
        if knn_lda_accuracy > knn_lda_best_accuracy
            knn_lda_best_accuracy = knn_lda_accuracy;
            knn_lda_best_dims = lda_dims;
            knn_lda_best_k = k;
        end
        fprintf('KNN分类器中LDA维度为 %d, K值为 %d 的准确度为 %.2f%%\n', lda_dims, k, knn_lda_accuracy*100);
    end
end
fprintf('KNN分类器中最优的LDA维度为 %d, K值为 %d，对应准确度为 %.2f%%\n', knn_lda_best_dims, knn_lda_best_k, knn_lda_best_accuracy*100);
%------------------------- 网格搜索 -------------------------%

% 保存日志
log_folder = './Grid_search_log';
save(fullfile(log_folder, 'svm_pca_log.mat'), 'svm_pca_log');
save(fullfile(log_folder, 'svm_lda_log.mat'), 'svm_lda_log');
save(fullfile(log_folder, 'knn_pca_log.mat'), 'knn_pca_log');
save(fullfile(log_folder, 'knn_lda_log.mat'), 'knn_lda_log');

