function accuracy=calculate_accuracy(predict_label,label_truth)

predict_label=predict_label(:);
label_truth=label_truth(:);

if length(predict_label)~=length(label_truth)
    error('分类器预测的测试集输出与真实输出的样本数不一致!');
end

accuracy=1-length(find(predict_label~=label_truth))/length(label_truth);