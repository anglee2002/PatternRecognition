%----------------------- ������Ϊ�ڶ��»�����С������׼����Ʊ�Ҷ˹��������ʾ������ -----------------------%
close all;
clear all;
clc;

% x1,x2,x3�ֱ�Ϊw1,w2,w3����������ѵ��������������ÿ��������3��ѵ��������ÿ��������ά��(��������������)����2
x1 = [0, 0; 2, 1; 1, 0]; % w1�������ѵ��������ÿ��Ϊһ������
x2 = [-1, 1; -2, 0; -2, -1]; % w2�������ѵ��������ÿ��Ϊһ������
x3 = [0, -2; 0, -1; 1, -2]; % w3�������ѵ��������ÿ��Ϊһ������
% ��ȡ����ľ�ֵ��Э��������������
u1 = mean(x1); u2 = mean(x2); u3 = mean(x3); % ����ѵ�������ľ�ֵ
c1 = cov(x1); c2 = cov(x2); c3 = cov(x3); % ����ѵ��������Э�������
t1 = diag(c1); t2 = diag(c2); t3 = diag(c3);
c1 = diag(t1); c2 = diag(t2); c3 = diag(t3); % ������ѵ��������Э��������Ϊ�ԽǾ�����������Ϊ����������ԣ�����������������
inv_c1 = inv(c1); inv_c2 = inv(c2); inv_c3 = inv(c3); % ����ѵ��������Э�������������
d1 = det(c1); d2 = det(c2); d3 = det(c3); % ����ѵ��������Э������������ʽ

% ����һ����������x_test,���ݹ�ʽ(2-39)�ж�x_test��������
x_test = [-2, 2];
p1 = -0.5 * (x_test - u1) * inv_c1 * (x_test - u1)' - 0.5 * log(d1);
p2 = -0.5 * (x_test - u2) * inv_c2 * (x_test - u2)' - 0.5 * log(d2);
p3 = -0.5 * (x_test - u3) * inv_c3 * (x_test - u3)' - 0.5 * log(d3);
[~, max_id] = max([p1, p2, p3]);
fprintf('x_test���ڵ�%d��\n', max_id);

g = str2sym('[x,y]');
g1 = simplify(-0.5 * (g - u1) * inv_c1 * (g - u1)' - 0.5 * log(d1));
g2 = simplify(-0.5 * (g - u2) * inv_c2 * (g - u2)' - 0.5 * log(d2));
g3 = simplify(-0.5 * (g - u3) * inv_c3 * (g - u3)' - 0.5 * log(d3));
g12 = simplify(g1 - g2); % w1,w2��ķֽ���
g23 = simplify(g2 - g3); % w2,w2��ķֽ���
g31 = simplify(g3 - g1); % w3,w1��ķֽ���

% �ò�ͬ��ɫ��������ֽ���
h12 = ezplot(g12); hold on;
set(h12, 'LineWidth', 2, 'color', 'red');
h23 = ezplot(g23); hold on;
set(h23, 'LineWidth', 2, 'color', 'blue');
h31 = ezplot(g31); hold on;
set(h31, 'LineWidth', 2, 'color', 'black');
legend('g12', 'g23', 'g31')

% �ò�ͬ��ɫ����״��������ѵ������
plot(x1(1, 1), x1(1, 2), 'or'); hold on;
plot(x1(2, 1), x1(2, 2), 'or'); hold on;
plot(x1(3, 1), x1(3, 2), 'or'); hold on;
plot(x2(1, 1), x2(1, 2), '>b'); hold on;
plot(x2(2, 1), x2(2, 2), '>b'); hold on;
plot(x2(3, 1), x2(3, 2), '>b'); hold on;
plot(x3(1, 1), x3(1, 2), 'vk'); hold on;
plot(x3(2, 1), x3(2, 2), 'vk'); hold on;
plot(x3(3, 1), x3(3, 2), 'vk'); hold on;
title('��Ҷ˹����');
xlabel('����1'); ylabel('����2');
hold off;
