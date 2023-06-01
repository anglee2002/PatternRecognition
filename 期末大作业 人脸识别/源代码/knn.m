function predicted_labels = knn(train_data, train_label, test_data, k)
    knn_model = fitcknn(train_data', train_label, 'NumNeighbors', k);
    predicted_labels = predict(knn_model, test_data');
end

