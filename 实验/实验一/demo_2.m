close all;
clear all;
clc;

% 加载鸢尾花数据集
[attrib1, attrib2, attrib3, attrib4, class] = textread('/Users/wallanceleon/Desktop/模式识别/实验/实验一/Iris.data', '%f%f%f%f%s', 'delimiter', ',');
attrib = [attrib1, attrib2, attrib3, attrib4];

label_set = char('Iris-setosa', 'Iris-versicolor', 'Iris-virginica');
label = zeros(150, 1);
label(strcmp(class, 'Iris-setosa')) = 1;
label(strcmp(class, 'Iris-versicolor')) = 2;
label(strcmp(class, 'Iris-virginica')) = 3;

% 将鸢尾花数据集按类别分组
setosa = attrib(label == 1, :);
versicolor = attrib(label == 2, :);
virginica = attrib(label == 3, :);

% 求取各类的均值，协方差矩阵及其逆矩阵
mean_setosa = mean(setosa);
mean_versicolor = mean(versicolor);
mean_virginica = mean(virginica);
cov_setosa = cov(setosa);
cov_versicolor = cov(versicolor);
cov_virginica = cov(virginica);
inv_cov_setosa = inv(cov_setosa);
inv_cov_versicolor = inv(cov_versicolor);
inv_cov_virginica = inv(cov_virginica);
det_cov_setosa = det(cov_setosa);
det_cov_versicolor = det(cov_versicolor);
det_cov_virginica = det(cov_virginica);

% 给定一个测试样本x_test,根据公式(2-39)判断x_test的类别归属
x_test = [6, 3.5, 4.5, 2.5];
p1 = 1/3 * exp(-1/2 * (x_test - mean_setosa) * inv_cov_setosa * (x_test - mean_setosa)') / sqrt(det_cov_setosa);
p2 = 1/3 * exp(-1/2 * (x_test - mean_versicolor) * inv_cov_versicolor * (x_test - mean_versicolor)') / sqrt(det_cov_versicolor);
p3 = 1/3 * exp(-1/2 * (x_test - mean_virginica) * inv_cov_virginica * (x_test - mean_virginica)') / sqrt(det_cov_virginica);
p = [p1, p2, p3];
[max_p, index] = max(p); 

disp(['The class of x_test is ', label_set(index, :)]);
