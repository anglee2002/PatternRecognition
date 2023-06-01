function [train_pca, test_pca] = pca(train_data, test_data, num_dims)
    % 计算训练集的均值
    train_mean = mean(train_data, 2);

    % 中心化训练集和测试集
    train_centered = train_data - train_mean;
    test_centered = test_data - train_mean;

    % 计算训练集的协方差矩阵
    cov_matrix = cov(train_centered');

    % 对协方差矩阵进行特征值分解
    [eig_vectors, eig_values] = eig(cov_matrix);

    % 对特征值进行排序并选择前num_dims个特征向量
    [~, sorted_idx] = sort(diag(eig_values), 'descend');
    selected_eig_vectors = eig_vectors(:, sorted_idx(1:num_dims));

    % 对训练集和测试集进行降维
    train_pca = selected_eig_vectors' * train_centered;
    test_pca = selected_eig_vectors' * test_centered;
end
