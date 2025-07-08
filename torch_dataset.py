from typing import final
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import ast

@final
class JQDataset(Dataset):
    """
    PyTorch数据集类，从CSV文件加载带标签的10×16数组

    CSV格式要求:
    - tag列: 字符串标签
    - data列: 格式为10×16嵌套数组的字符串

    修正点:
    1. 使用分类索引而不是独热向量索引
    2. 自动构建标签到索引的映射
    3. 存储原始标签和索引映射供外部访问
    """

    def __init__(self, csv_path):
        """
        初始化数据集
        Args:
            csv_path: CSV文件路径
        """
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)

        # 获取唯一标签并创建映射
        unique_tags = sorted(self.df['tag'].unique())
        self.class_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        self.idx_to_class = {idx: tag for tag, idx in self.class_to_idx.items()}
        self.classes = unique_tags  # 存储原始标签名称

        # 转换数据格式
        self.data_arrays = []
        self.label_indices = []  # 存储标签索引而不是原始标签

        # 处理每一行数据
        for _, row in self.df.iterrows():
            # 转换字符串格式的数组
            arr = ast.literal_eval(row['data'])
            self.data_arrays.append(np.array(arr, dtype=np.float32))

            # 将标签转换为索引
            self.label_indices.append(self.class_to_idx[row['tag']])

    def __len__(self):
        """返回数据集长度"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 数据索引
        Returns:
            (tensor_data, tensor_label)
        """
        # 获取数据和标签索引
        data = self.data_arrays[idx]
        label_idx = self.label_indices[idx]

        # 转换为PyTorch张量
        tensor_data = torch.tensor(data, dtype=torch.float32).view(-1, 10, 16) / 255
        tensor_label = torch.tensor(label_idx, dtype=torch.long)  # 使用long类型作为分类索引

        return tensor_data, tensor_label

    def get_label_mapping(self):
        """获取标签到索引的映射关系"""
        return self.class_to_idx

    def get_index_mapping(self):
        """获取索引到标签的映射关系"""
        return self.idx_to_class
