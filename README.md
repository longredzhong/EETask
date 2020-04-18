TODO:
- [x] 完成训练、验证一条龙服务
- [x] ~~使用其他数据集验证我的代码是否有问题~~ 不需要了问题解决了
- [ ] 使用bert或者其他预训练模型~~，因为感觉Albert有点问题~~
- [ ] 完善训练代码，加入对test1.json的预测，加入实验数据记录
- [ ] linear_schedule_with_warmup有问题或者我用错了，需要完善
- [ ] 改善模型
  - [ ] 将分类与序列标注相结合
  - [ ] 简化序列标注类别
- [ ] 如果以后精度提不上去了，就搞一波超参数搜索
- [ ] 训练时加入学习率衰减

## 踩坑记录

- torch的bucketiterate要慎用，因为如果使batch中的text长度趋于一致，可能能会改变数据的分布，使模型难以学习
- 预训练模型的学习率要小 1e-5，crf层的学习率要大，大概是一百倍吧，苏剑林大佬讲的，因为我没有做完整的实验，所以也不知道情况

## 模型改善想法
1. 将序列分类的类别数降低:现在类别是(event_type,role) 改为(role)
2. 模型加入分类任务，对event_type进行预测，为多分类
3. 根据预测的role和event_type预测完整的类别(event_type,role)


``` mermaid
graph TD
input --> label_role_ids
input --> text_to_ids
input --> event_type_labels
text_to_ids --> Model
Model --> output_0 --> classifition_role
Model --> output_1 --> classifition_event_labels
classifition_role --> CRF
CRF --> loss_role
label_role_ids --> CRF
classifition_event_labels --> BCELoss
event_type_labels --> BCELoss
BCELoss --> loss_event
```
