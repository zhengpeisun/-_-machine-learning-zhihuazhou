import pandas as pd#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  (accuracy_score,precision_score,f1_score,recall_score,roc_auc_score,roc_curve,precision_recall_curve,average_precision_score,log_loss,confusion_matrix)
data = pd.read_csv('Synthetic_Watermelon_Dataset__10000_rows_.csv', header=0, encoding='utf-8')#通过pd.read_csv方式读取数据，UTF-8，header=0指的是第一行就是列明
data = data.rename(columns={'好瓜(1=好;0=坏)': '好瓜'})#重命名最后一列名字
#data = data[:8]
x = data[['密度', '含糖率']].values#从dataframe中选取密度、含糖率两列作为输入特征
y = data["好瓜"].values#最后一列作为标签

print(y)
np.random.seed(0)#设置种子
indices = np.random.permutation(len(x))#生成一个0-19的随机排列-意义在与打乱顺序，将内部分布均匀化
train_idx = indices[:8000]#取前16个索引为训练集合
test_idx = indices[8000:]#其余的为测试集合
x_train,y_train = x[train_idx],y[train_idx]#按照选定的索引切片，得到训练集特征 X_train（形状 (16,2)）和训练标签 y_train（长度 16）
x_test,y_test = x[test_idx],y[test_idx]#按照选定的索引切片，得到训练集特征 X_train（形状 (4,2)）和训练标签 y_train（长度 4）。
model = LogisticRegression( )#实例化一个 scikit-learn 的逻辑回归模型，使用默认参数（L2 正则化、lbfgs 优化器等）。scikit-learn（也常写作
#“sklearn”）是一个非常流行的、用纯 Python 写成的机器学习库，全称是 “Scikit-Learn”。它以简单易用、接口统一、文档完善著称，几乎涵盖了常见的监督学习和非监督学习算


#法：线性/逻辑回归、决策树、随机森林、支持向量机、聚类、降维、模型选择、评估指标……等等;
# LogisticRegression(
  #  penalty='l2',          # 使用 L2 正则化：'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'大多数优化器都支持 L2；l1'	'liblinear', 'saga'	只有这两种 solver 支持 L1
  
  #  dual=False,            # 不使用对偶（dual）实现，适合 n_samples > n_features（样本数>特征维数）当 n_samples > n_features（样本比特征多）时，原始（
  # primal）方法通常更快，就 dual=False。当 n_features > n_samples（特征比样本多）时，对偶（dual）方法可能更高效，就可以考虑 dual=True（前提是你用的 solver 
  # 支持对偶，也就是 solver='liblinear' 并且 penalty='l2'）。

   # tol=1e-4,              # 收敛阈值在每一次迭代结束后，算法会计算当前这一步的“目标函数值”（也就是带正则化的对数似然）或相邻两次参数变化的大小，检查它们与上次相比
   # 变化有多大。当“变化量”小到低于我们设定的容忍度（tol）时，就认为“进一步迭代已经不会带来明显改善”，于是停止迭代，输出当前参数作为最终解。

   # C=1.0,                 # 正则化强度的倒数，C 越大正则化越弱

   # fit_intercept=True,    # 是否拟合截距项,true时这时模型会在训练时一起学习 𝑤 和 𝑏。fit_intercept=True 是默认值，一般不用显式指定。False时只学习W

   # solver='lbfgs',        # 优化算法，这里默认用 L-BFGS，优化器，L1，L2选择时记得注意；
#| Solver        | 支持正则化                 | 支持多分类模式          | 适合数据类型 | 适合规模                    | 通常场景                        |
#| ------------- | ------------------------ | ------------------    | -----      | ------------------        | ------------------------       |
#| **liblinear** | L1, L2                   | OvR（二分类 & 多分类）   | 稠密 / 稀疏 | 中小规模（几万样本、几千特征） | 需要 L1/L2、样本 & 特征都中等时    |
#| **lbfgs**     | L2, none                 | 支持 Multinomial 多分类 | 稠密       | 小到中规模（几千样本到几万特征）| 只需 L2，多分类精度较好           |
#| **newton-cg** | L2, none                 | 支持 Multinomial 多分类 | 稠密       | 小到中规模                  | 只需 L2，多分类，二阶收敛快        |
#| **sag**       | L2, none                 | 支持 Multinomial 多分类 | 稠密       | 大规模（$\ge 10^5$ 样本）    | 只需 L2，海量稠密数据             |
#| **saga**      | L1, L2, ElasticNet, none | 支持 Multinomial 多分类 | 稠密 / 稀疏 | 大规模（$\ge 10^5$ 样本）    | 需要 L1/ElasticNet，海量或稀疏数据|
#大量时优先sag或saga

   # max_iter=100,          # 最大迭代次数 max_iter=100 只是告诉优化器“如果进行了 100 次参数更新还没满足收敛条件，就停止并警告我”。默认值100

   # multi_class='auto',    # 多分类模式，auto 会根据标签自动选择 'ovr'=》2分类liblinear不支持 或 'multinomial' ：多分类
   #| 特性   | OvR                                                      | Multinomial                            |
   #| ---- | -------------------------------------------------------- | -------------------------------------- |
   #| 训练方式 | 对每个类别单独训练一个二分类模型，共 $K$ 个模型                               | 一次联合训练一个 $K$-分类模型                      |
   #| 计算量  | $K$ 次二分类（每次优化维度为 R^d）                         | 一次多项式 Softmax 优化（参数矩阵大小为 dxk） |
   #| 预测阶段 | 先并行（或逐一）计算每个二分类器的概率，再选最大者                                | 直接一次性输出所有类别的 Softmax 概率                |
   #| 优点   | - 实现简单<br>- 对小样本、特征极高维时更稳定                               | - 联合优化，能更好地协调各类别间竞争<br>- 对多分类精度更好      |
   #| 缺点   | - 各个二分类器不共享信息、相互独立<br>- 类别失衡时可能表现不稳定                     | - 对特征维度和类别数都较敏感，计算和内存开销更大              |
   #| 适用场景 | - 需要 L1 正则化（因为 `liblinear` 只支持 OvR）<br>- 样本量小、特征极稀疏（如文本） | - 样本量中等或大，特征维度不极端时<br>- 需要最优的多分类性能时    |

   # verbose=0,             # 是否输出训练过程日志 0否 1是
   
   # warm_start=False,      # 是否使用上一次训练的权重作为初始点这会导致两个不同的 fit 过程产生“依赖关系”——第二次的结果会深受第一次训练结果的影响，
   # 破坏了“每次调用 fit 都是从同一起点开始”的一致性，降低了结果的可复现性。“增量式训练（warm start）”常见于非常大的数据流、在线学习或需要分批次多次微调同一模型的场景。

   # n_jobs=None,           # 并行线程数，None 表示 1 个；-1 表示使用所有 CPU
   
   #l1_ratio=None          # 当 penalty='elasticnet' 时，l1_ratio 用来指定 L1/L2 比例；否则为 None
#)





model.fit(x_train,y_train)#在训练集上进行参数估计，自动根据 X_train 和 y_train 求出最优的权重向量。
y_pred = model.predict(x_test)#在测试集上预测瓜的好坏
y_proba = model.predict_proba(x_test)[:,1]#y_proba代表是返回概率分布；

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))
total = len(y_test)

accuracy_manual = (TP+TN)/total
precision_manual = TP / (TP+FP) if (TP+FP)>0 else 0
recall_manual = TP / (TP+FN) if (TP+FN)>0 else 0
specificity_manual =  TN / (TN + FP) if (TN + FP) > 0 else 0
fpr_manual = FP / (FP + TN) if (FP + TN) > 0 else 0#假正例率
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
logloss_manual = log_loss(y_test,y_proba)
 
accuracy_sklearn = accuracy_score(y_test,y_pred)
precision_sklearn = precision_score(y_test,y_pred)
recall_sklearn = recall_score(y_test,y_pred)
f1_sklearn = f1_score(y_test,y_pred)
tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()
specificity_sklearn = tn / (tn+fp)
fpr_sklearn = fp/(fp+tn)
logloss_sklearn = log_loss(y_test,y_proba)

print("指标            手动计算        sklearn")
print(f"Accuracy      {accuracy_manual:.3f}         {accuracy_sklearn:.3f}")
print(f"Precision     {precision_manual:.3f}         {precision_sklearn:.3f}")
print(f"Recall        {recall_manual:.3f}         {recall_sklearn:.3f}")
print(f"Specificity   {specificity_manual:.3f}         {specificity_sklearn:.3f}")
print(f"FPR           {fpr_manual:.3f}         {fpr_sklearn:.3f}")
print(f"F1 Score      {f1_manual:.3f}         {f1_sklearn:.3f}")
print(f"Log-loss      {logloss_manual:.3f}         {logloss_sklearn:.3f}")

precisions ,recalls ,pr_thresholds = precision_recall_curve(y_test,y_proba)
ap_manual = average_precision_score(y_test,y_proba)
plt.figure(figsize = (6,4))
plt.plot(recalls , precisions ,marker = "o",label = f'AP={ap_manual: .3f}')
plt.xlabel("Recall")
plt.ylabel('Precision')
plt.title('Precision-Recall 曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fprs, tprs, roc_thresholds = roc_curve(y_test, y_proba)
auc_manual = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(fprs, tprs, marker='o', label=f'AUC={auc_manual:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机猜测')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()