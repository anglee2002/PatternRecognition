# PatternRecognition
YNU 2023 Spring Course

## 平时实验
略
## 期末综合实验 人脸识别

### 项目结构
以下是期末综合实验的项目结构
```
├── 实验要求
│   └── 期末综合实验:基于人脸识别的分类器设计.doc
│   └── 模式识别期末综合实验:基于人脸识别的分类器设计.pptx
├── 源代码
│   └── ORL_trainset.m  %ORL训练集
│   └── ORL_testset.m  %ORL测试集 
│   └── ORL_testlabel.m  %%ORL测试集类别标签
│   └── FR_holistic.m  %该部分是使用整体性方法进行人脸识别的主程序，调用了svm.m和knn.m两个模块
│   └── FR_featurebased.m  %该部分是使用特征提取方法（通过网格搜索寻找最优参数）进行人脸识别的主程序，调用了svm.m、knn.m、pca.m、lda.m四个模块
│   └── FR_deeplearning.m  %该部分是使用深度学习进行人脸识别的主程序，调用了myCNN.m模块
│   └── calculate_accurary.m  %该部分是评估分类结果的子模块
│   └── svm.m  %支持向量机子模块
│   └── knn.m  %K最近临子模块
│   └── pca.m  %主成分分析子模块
│   └── lda.m  %线性判别分析子模块
│   └── myCNN.m  %卷机神经网络子模块
│   └── Grid_search_log  %该部分为网格搜索文件夹，用于存储网格搜索日志并生成训练图像
│        └── svm_pca_log.mat  %记录使用PCA降维的SVM算法的参数和准确率日志
│        └── svm_lda_log.mat  %记录使用LDA降维的SVM算法的参数和准确率日志
│        └── knn_pca_log.mat  %记录使用PCA降维的KNN算法的参数和准确率日志
│        └── knn_lda_log.mat  %记录使用LDA降维的KNN算法的参数和准确率日志
│        └── grid_search_plot.m  %将搜索日志绘制成图像
├── 参考文献 
│   └── Face Recognition:From Traditional to Deep Learning Methods.pdf
│   └── 基于深度学习的人脸识别方法综述 余璀璨.pdf
└── 实验报告 
    └── 人脸识别综述及MATLAB实现.pdf
    └── 人脸识别综述及MATLAB实现.md
```
### 实验结果

| 方法    | 准确率 |
| ------- | ------ |
| SVM     | 84.38% |
| KNN     | 88.12% |
| SVM+PCA | 85.00% |
| SVM+LDA | 90.00% |
| KNN+PCA | 88.75% |
| KNN+LDA | 98.12% |
| CNN     | 95.62% |

　　<center><strong>表 1  分类结果 </strong></center>
  
<img src="https://raw.githubusercontent.com/anglee2002/Picbed/main/untitled.png" alt="untitled" style="zoom:33%;" />

<center><strong>图 1 特征提取识别结果  （1）SVM-PCA （2）SVM-LDA （3）KNN-PCA （4）KNN-LDA </strong></center>

<img src="https://raw.githubusercontent.com/anglee2002/Picbed/main/screenshot2023-06-01%2019.25.18.gif" style="zoom:50%;" />
<center><strong>图 2 卷机神经网络训练轮次图  </strong></center>
<img src="https://raw.githubusercontent.com/anglee2002/Picbed/main/screenshot2023-06-01%2019.25.18.gif" style="zoom:50%;" />
<img src="https://raw.githubusercontent.com/anglee2002/Picbed/main/screenshot2023-06-01%2019.25.18.gif" style="zoom:50%;" />
