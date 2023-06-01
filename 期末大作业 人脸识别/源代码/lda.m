function [train_lda, test_lda] = lda(train_data, train_label, test_data, num_dims_lda)
    % 计算每个类别的样本均值
    class_means = zeros(size(train_data, 1), max(train_label));

    for i = 1:max(train_label)
        class_means(:, i) = mean(train_data(:, train_label == i), 2);
    end

    % 计算总体均值
    overall_mean = mean(train_data, 2);

    % 计算类内散布矩阵
    Sw = zeros(size(train_data, 1), size(train_data, 1));

    for i = 1:max(train_label)
        class_data = train_data(:, train_label == i);
        class_centered = class_data - class_means(:, i);
        Sw = Sw + class_centered * class_centered';
    end

    % 计算类间散布矩阵
    Sb = zeros(size(train_data, 1), size(train_data, 1));

    for i = 1:max(train_label)
        class_centered = class_means(:, i) - overall_mean;
        Sb = Sb + size(train_data(:, train_label == i), 2) * (class_centered * class_centered');
    end

    % 添加正则化项
    lambda = 0.001;
    Sw_reg = Sw + lambda * eye(size(Sw));

    % 修正类内散布矩阵的特征值
    epsilon = 1e-6;
    Sw_reg = Sw_reg + epsilon * trace(Sw_reg) * eye(size(Sw_reg));

    % 对 (Sw_reg^-1) * Sb 进行特征值分解
    [eig_vectors, eig_values] = eig(inv(Sw_reg) * Sb);

    % 对特征值进行排序并选择前 num_dims_lda 个特征向量
    [~, sorted_idx] = sort(diag(eig_values), 'descend');
    selected_eig_vectors = eig_vectors(:, sorted_idx(1:num_dims_lda));

    % 对训练集和测试集进行降维
    train_lda = selected_eig_vectors' * train_data;
    test_lda = selected_eig_vectors' * test_data;
end
