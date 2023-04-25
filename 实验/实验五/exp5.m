% 读取原始数据
[attrib1, attrib2, attrib3, attrib4, class] = textread('iris.data', '%f%f%f%f%s', 'delimiter', ',');

% 将数据矩阵和类别标签合并为一个矩阵
x = [attrib1, attrib2, attrib3, attrib4]';
y = grp2idx(class);

% 进行LDA降维
[vec, val] = LDA(x, y);
W = vec(:, 1:2); % 取前两个判别分量
y_proj = W' * x; % 投影后的数据

% 画出投影后的数据分布
figure
gscatter(y_proj(1, :), y_proj(2, :), y)
title('Iris数据集投影后的分布')
xlabel('第一判别分量')
ylabel('第二判别分量')
legend('setosa', 'versicolor', 'virginica', 'Location', 'best')
function [vec, val] = LDA(xtr, ytr)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input:
    %     xtr:  data matrix (Each column is a data point)
    %     ytr:  class label (class 1, ..., k)
    % Output:
    %     vec:  sorted discriminative components
    %     val:  corresponding eigenvalues
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [D, ntr] = size(xtr);
    classnum = length(unique(ytr));
    miu = mean(xtr, 2);

    sigmaB = sparse(D, D);

    for i = 1:classnum
        miu_class(:, i) = mean(xtr(:, find(ytr == i)), 2);
        sigmaB = sigmaB + length(find(ytr == i)) * (miu_class(:, i) - miu) * (miu_class(:, i) - miu)';
    end

    sigmaB = (sigmaB + sigmaB') / 2;

    sigmaT = (ntr - 1) * cov(xtr');
    sigmaT = (sigmaT + sigmaT') / 2;

    sigmaW = sigmaT - sigmaB;
    sigmaW = (sigmaW + sigmaW') / 2;

    [eigvector, eigvalue] = eig(sigmaB, sigmaW);
    [val, id] = sort(-diag(eigvalue));
    vec = eigvector(:, id);
    val = -val;

end
