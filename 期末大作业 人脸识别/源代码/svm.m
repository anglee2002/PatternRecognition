function predicted_labels = svm(train_data, train_label, test_data)
    svm_model = fitcecoc(train_data', train_label); % 使用fitcecoc训练多类别分类器
    predicted_labels = predict(svm_model, test_data');
end

