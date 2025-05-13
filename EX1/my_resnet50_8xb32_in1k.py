'''
Author: foreverstyle
Date: 2025-05-08 15:58:39
LastEditTime: 2025-05-08 17:43:38
Description: 创建微调配置文件
FilePath: \ECE371\mmpretrain\configs\EX1\my_resnet50_8xb32_in1k.py
'''
_base_ = [
    '../_base_/models/resnet50.py', 
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime.py', 
]

load_from = 'pre_trained/resnet50_8xb32_in1k_20210831-ea4938fc.pth' # 预训练模型路径
model = dict(
    # backbone=dict(
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='pre_trained/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
    #     )),
    head=dict(
        num_classes=5, # 将类别数改为5
        topk=(1,), # 只保留top1准确率 
    )
)



data_preprocessor = dict(
    num_classes=5, # 将类别数改为5
)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='data/flower_dataset',
        data_prefix='',
        split='',
        ann_file='train.txt',
        classes='data/flower_dataset/classes.txt',),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='data/flower_dataset',
        data_prefix='',
        split='',
        ann_file='val.txt',
        classes='data/flower_dataset/classes.txt',),
)

val_evaluator = dict(type='Accuracy', topk=(1,)) # 只保留top1准确率

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)) #学习率设置为0.005

# # learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[5, 10, 13], gamma=0.1) 

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1) #max_epochs=50

