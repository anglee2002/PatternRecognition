N = [1, 16, 256, 2048]; % 生成样本的数量
x = -6:0.01:6; % 画图时x的取值范围
h1 = [0.25, 1, 2, 4, 8]; % 窗宽的取值
h = length(h1); % 窗函数的数量

% 生成一元标准正态分布样本
for i = 1:length(N)
    X{i} = randn(1, N(i));
end

% 计算Parzen窗估计
for i = 1:length(N)
    for j = 1:h
        hn = h1(j) / sqrt(N(i));
        p{i,j} = zeros(size(x));
        for k = 1:N(i)
            p{i,j} = p{i,j} + exp(-(x-X{i}(k)).^2/(2*hn^2)) / (sqrt(2*pi)*hn);
        end
        p{i,j} = p{i,j} / N(i);
    end
end

% 画图
figure;
count = 1;
for i = 1:length(N)
    for j = 1:h
        subplot(length(N), h, count);
        plot(x, p{i,j});
        title(['N=', num2str(N(i)), ', h1=', num2str(h1(j))]);
        count = count + 1;
    end
end
