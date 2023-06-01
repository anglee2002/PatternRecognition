% 导入数据
load('svm_pca_log.mat');
svm_pca_data = svm_pca_log;
load('svm_lda_log.mat');
svm_lda_data = svm_lda_log;
load('knn_pca_log.mat');
knn_pca_data = knn_pca_log;
load('knn_lda_log.mat');
knn_lda_data = knn_lda_log;

% 绘制曲线图
figure;
hold on;

% SVM with PCA
subplot(2, 2, 1);
plot(svm_pca_data(:, 1), svm_pca_data(:, 2), 'ko-', 'MarkerFaceColor', 'b');
xlabel('维度');
ylabel('准确率');
title('(1)');

% SVM with LDA
subplot(2, 2, 2);
plot(svm_lda_data(:, 1), svm_lda_data(:, 2), 'ko-', 'MarkerFaceColor', 'b');
xlabel('维度');
ylabel('准确率');
title('(2)');

% kNN with PCA (k = 1)
subplot(2, 2, 3);
knn_pca_k1 = knn_pca_data(knn_pca_data(:, 2) == 1, :);
plot(knn_pca_k1(:, 1), knn_pca_k1(:, 3), 'ko-', 'MarkerFaceColor', 'b');
xlabel('维度');
ylabel('准确率');
title('(3)');

% kNN with LDA (k = 1)
subplot(2, 2, 4);
knn_lda_k1 = knn_lda_data(knn_lda_data(:, 2) == 1, :);
plot(knn_lda_k1(:, 1), knn_lda_k1(:, 3), 'ko-', 'MarkerFaceColor', 'b'l);
xlabel('维度');
ylabel('准确率');
title('(4)');

hold off;
