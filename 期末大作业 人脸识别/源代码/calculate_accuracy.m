function accuracy=calculate_accuracy(predict_label,label_truth)

predict_label=predict_label(:);
label_truth=label_truth(:);

if length(predict_label)~=length(label_truth)
    error('������Ԥ��Ĳ��Լ��������ʵ�������������һ��!');
end

accuracy=1-length(find(predict_label~=label_truth))/length(label_truth);