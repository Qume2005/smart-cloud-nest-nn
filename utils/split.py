import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(dataset: torch.utils.data.Dataset, test_size=0.2, random_state=None):
    """
    使用sklearn的StratifiedShuffleSplit进行分层分割

    Args:
        dataset (Dataset): PyTorch数据集
        test_size (float): 测试集比例 (默认0.2)
        random_state (int): 随机种子 (默认None)

    Returns:
        train_dataset (Subset): 训练集
        test_dataset (Subset): 测试集
    """
    # 获取所有标签
    labels = [label for _, label in dataset]

    # 使用sklearn进行分层分割
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    # 获取训练和测试索引
    for train_index, test_index in sss.split(np.zeros(len(dataset)), labels):
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)

    return train_dataset, test_dataset