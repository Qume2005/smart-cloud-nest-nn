import unittest
from typing import final

import torch
from torch.utils.data import DataLoader

from main_nn_module import MainNnModule
from torch_dataset import JQDataset
from utils.split import stratified_split


@final
class DatasetTestCase(unittest.TestCase):

    def test_shape(self) -> None:
        # 1. 创建示例数据集 (实际使用时应替换为真实CSV路径)
        csv_path = "./datas/dataset.csv"

        # 2. 创建PyTorch数据集实例
        dataset = JQDataset(csv_path)

        # 3. 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 4. 获取一个batch的数据
        sample_data, sample_tags = next(iter(dataloader))

        # 打印数据形状验证
        print(f"数据形状: {sample_data.shape}")  # 应为 torch.Size([4, 1, 10, 16])
        print(f"标签: {sample_tags}")
        self.assertEqual(sample_data.shape, torch.Size([4, 1, 10, 16]))


class TestStratifiedSplit(unittest.TestCase):
    """测试分层分割功能"""

    def setUp(self):
        self.dataset = JQDataset("./datas/dataset.csv")

    def test_split_ratio(self):
        """测试分割比例"""
        train_set, test_set = stratified_split(self.dataset, test_size=0.2)
        total = len(self.dataset)
        self.assertAlmostEqual(len(test_set) / total, 0.2, delta=0.02)

    def test_stratification(self):
        """测试分层是否保持标签分布"""
        train_set, test_set = stratified_split(self.dataset, test_size=0.2)

        # 获取原始标签分布
        orig_label_counts = {}
        for _, label in self.dataset:
            orig_label_counts[label] = orig_label_counts.get(label, 0) + 1

        # 检查测试集标签分布
        test_label_counts = {}
        for _, label in test_set:
            test_label_counts[label] = test_label_counts.get(label, 0) + 1

        # 验证每个标签在测试集中的比例
        for label, orig_count in orig_label_counts.items():
            exp_count = max(1, int(0.2 * orig_count))  # 期望测试样本数
            self.assertAlmostEquals(test_label_counts.get(label, 0), exp_count, delta=1)

    def test_dataloader(self):
        """测试DataLoader批次形状"""
        train_set, _ = stratified_split(self.dataset, test_size=0.2)
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

        # 验证批次形状
        for batch_data, batch_labels in train_loader:
            self.assertEqual(batch_data.shape, (8, 1, 10, 16))
            self.assertEqual(len(batch_labels), 8)
            break  # 只检查第一个批次

@final
class NNModuleTestCase(unittest.TestCase):
    @staticmethod
    def test_input_shape() -> torch.Tensor:
        main_nn_module = MainNnModule()
        dummy_input = torch.randn(4, 1, 10, 16)
        return main_nn_module(dummy_input)

    def test_output_shape(self):
        logist = self.test_input_shape()
        self.assertEqual(logist.shape, torch.Size([4, 3]))

if __name__ == "__main__":
    unittest.main()