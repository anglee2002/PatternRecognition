function predicted_labels = myCNN(train_data, train_label, test_data, trainnum)
    % 使用卷积神经网络进行训练和预测

    % 构建卷积神经网络模型
    layers = [
        imageInputLayer([32 32 1]) % 输入层，指定输入图像的尺寸
        convolution2dLayer(5, 32) % 卷积层，使用5x5的卷积核，输出32个特征图
        reluLayer() % ReLU激活函数层
        maxPooling2dLayer(2, 'Stride', 2) % 最大池化层，使用2x2的窗口进行池化
        fullyConnectedLayer(40) % 全连接层，输出40个类别
        softmaxLayer() % Softmax层，进行分类
        classificationLayer() % 分类层
    ];

    % 设置训练参数
    options = trainingOptions('adam', 'MaxEpochs', 600, 'MiniBatchSize', 32, 'Verbose', true, ...
        'Plots', 'training-progress', 'OutputFcn', @myTrainingProgressFcn);

    % 将训练数据转换为图像数据格式
    XTrain = reshape(train_data, [32 32 1 trainnum]);

    % 将训练标签转换为分类标签数据格式
    YTrain = categorical(train_label);

    % 创建用于保存训练过程数据的结构体
    myCNN_log = struct('Epoch', [], 'Accuracy', []);

    % 自定义训练过程回调函数
    function stop = myTrainingProgressFcn(info)
        % 在每个轮次结束时记录准确率数据
        myCNN_log.Epoch = [myCNN_log.Epoch, info.Epoch];
        myCNN_log.Accuracy = [myCNN_log.Accuracy, info.TrainingAccuracy];
        stop = false; % 继续训练
    end

    % 训练卷积神经网络模型
    net = trainNetwork(XTrain, YTrain, layers, options);

    % 将测试数据转换为图像数据格式
    XTest = reshape(test_data, [32 32 1 size(test_data, 2)]);

    % 使用训练好的模型进行预测
    YPred = classify(net, XTest);

    % 将预测结果转换为数值标签格式
    predicted_labels = double(YPred);

    % 绘制正确率随轮次的图像
    %figure;
    %plot(myCNN_log.Epoch, myCNN_log.Accuracy);
    %xlabel('Epoch');
    %ylabel('Accuracy');
    %title('Accuracy vs. Epoch');
end
