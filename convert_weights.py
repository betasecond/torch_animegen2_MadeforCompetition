import argparse
import numpy as np
import os
import tensorflow as tf
from AnimeGANv2.net import generator as tf_generator
import torch
from transformers.modeling_tf_utils import load_tf_weights
from model import Generator


class ModelConverter:
    def __init__(self, tf_checkpoint_path, save_name):
        self.tf_checkpoint_path = tf_checkpoint_path
        self.save_name = save_name

    def load_tf_weights(tf_path):
        # 定义输入占位符
        test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
        # 定义生成器模型
        with tf.variable_scope("generator", reuse=False):
            test_generated = tf_generator.G_net(test_real).fake
        # 定义模型权重保存器
        saver = tf.train.Saver()
        # 创建TensorFlow会话
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})) as sess:
            # 加载模型检查点
            ckpt = tf.train.get_checkpoint_state(tf_path)
            # 确保检查点存在
            assert ckpt is not None and ckpt.model_checkpoint_path is not None, f"Failed to load checkpoint {tf_path}"
            # 恢复会话中的权重
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(f"Tensorflow model checkpoint {ckpt.model_checkpoint_path} loaded")
            # 存储权重
            tf_weights = {}
            for v in tf.trainable_variables():
                tf_weights[v.name] = v.eval()
        return tf_weights

    @staticmethod
    def convert_keys(k):

        # 1. divide tf weight name in three parts [block_idx, layer_idx, weight/bias]
        # 2. handle each part & merge into a pytorch model keys

        k = k.replace("Conv/", "Conv_0/").replace("LayerNorm/", "LayerNorm_0/")
        keys = k.split("/")[2:]

        is_dconv = False

        # handle C block..
        if keys[0] == "C":
            if keys[1] in ["Conv_1", "LayerNorm_1"]:
                keys[1] = keys[1].replace("1", "5")

            if len(keys) == 4:
                assert "r" in keys[1]

                if keys[1] == keys[2]:
                    is_dconv = True
                    keys[2] = "1.1"

                block_c_maps = {
                    "1":  "1.2",
                    "Conv_1":  "2",
                    "2":  "3",
                }
                if keys[2] in block_c_maps:
                    keys[2] = block_c_maps[keys[2]]

                keys[1] = keys[1].replace("r", "") + ".layers." + keys[2]
                keys[2] = keys[3]
                keys.pop(-1)
        assert len(keys) == 3

        # handle output block
        if "out" in keys[0]:
            keys[1] = "0"

        # first part
        if keys[0] in ["A", "B", "C", "D", "E"]:
            keys[0] = "block_" + keys[0].lower()

        # second part
        if "LayerNorm_" in keys[1]:
            keys[1] = keys[1].replace("LayerNorm_", "") + ".2"
        if "Conv_" in keys[1]:
            keys[1] = keys[1].replace("Conv_", "") + ".1"

        # third part
        keys[2] = {
            "weights:0": "weight",
            "w:0": "weight",
            "bias:0": "bias",
            "gamma:0": "weight",
            "beta:0": "bias",
        }[keys[2]]

        return ".".join(keys), is_dconv

    def convert_and_save(self):
        tf_weights = self.load_tf_weights()
        torch_net = Generator()
        torch_weights = torch_net.state_dict()
        torch_converted_weights = {}
        for k, v in tf_weights.items():
            torch_k, is_dconv = convert_keys(k)
            assert torch_k in torch_weights, f"weight name mismatch: {k}"
            converted_weight = torch.from_numpy(v)
            if len(converted_weight.shape) == 4:
                if is_dconv:
                    converted_weight = converted_weight.permute(2, 3, 0, 1)
                else:
                    converted_weight = converted_weight.permute(3, 2, 0, 1)
                    assert torch_weights[torch_k].shape == converted_weight.shape, f"shape mismatch: {k}"
            torch_converted_weights[torch_k] = converted_weight
        torch_net.load_state_dict(torch_converted_weights)
        torch.save(torch_net.state_dict(), self.save_name)
        print(f"PyTorch model saved at {self.save_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tf_checkpoint_path',
        type=str,
        default='AnimeGANv2/checkpoint/generator_Paprika_weight',
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default='pytorch_generator_Paprika.pt',
    )
    args = parser.parse_args()

    model_converter = ModelConverter(args.tf_checkpoint_path, args.save_name)
    model_converter.convert_and_save()

# 加载TensorFlow模型权重
# def load_tf_weights(tf_path):
#     # 定义输入占位符
#     test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
#     # 定义生成器模型
#     with tf.variable_scope("generator", reuse=False):
#         test_generated = tf_generator.G_net(test_real).fake
#     # 定义模型权重保存器
#     saver = tf.train.Saver()
#     # 创建TensorFlow会话
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})) as sess:
#         # 加载模型检查点
#         ckpt = tf.train.get_checkpoint_state(tf_path)
#         # 确保检查点存在
#         assert ckpt is not None and ckpt.model_checkpoint_path is not None, f"Failed to load checkpoint {tf_path}"
#         # 恢复会话中的权重
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print(f"Tensorflow model checkpoint {ckpt.model_checkpoint_path} loaded")
#         # 存储权重
#         tf_weights = {}
#         for v in tf.trainable_variables():
#             tf_weights[v.name] = v.eval()
#     return tf_weights
#
# # 转换权重键名
# def convert_keys(k):
#     # 省略部分代码
#     return ".".join(keys), is_dconv
#
# # 转换并保存权重
# def convert_and_save(tf_checkpoint_path, save_name):
#     # 加载TensorFlow权重
#     tf_weights = load_tf_weights(tf_checkpoint_path)
#     # 创建PyTorch生成器模型
#     torch_net = Generator()
#     torch_weights = torch_net.state_dict()
#     # 转换权重
#     torch_converted_weights = {}
#     for k, v in tf_weights.items():
#         torch_k, is_dconv = convert_keys(k)
#         assert torch_k in torch_weights, f"weight name mismatch: {k}"
#         converted_weight = torch.from_numpy(v)
#         if len(converted_weight.shape) == 4:
#             if is_dconv:
#                 converted_weight = converted_weight.permute(2, 3, 0, 1)
#             else:
#                 converted_weight = converted_weight.permute(3, 2, 0, 1)
#         assert torch_weights[torch_k].shape == converted_weight.shape, f"shape mismatch: {k}"
#         torch_converted_weights[torch_k] = converted_weight
#     # 确保所有权重都已转换
#     assert sorted(list(torch_converted_weights)) == sorted(list(torch_weights)), f"some weights are missing"
#     # 加载转换后的权重
#     torch_net.load_state_dict(torch_converted_weights)
#     # 保存PyTorch模型
#     torch.save(torch_net.state_dict(), save_name)
#     print(f"PyTorch model saved at {save_name}")
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # 添加参数
#     parser.add_argument(
#         '--tf_checkpoint_path',
#         type=str,
#         default='AnimeGANv2/checkpoint/generator_Paprika_weight',
#     )
#     parser.add_argument(
#         '--save_name',
#         type=str,
#         default='pytorch_generator_Paprika.pt',
#     )
#     args = parser.parse_args()
#     convert_and_save(args.tf_checkpoint_path, args.save_name)



#
#
# def load_tf_weights(tf_path):
#     test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
#     with tf.variable_scope("generator", reuse=False):
#         test_generated = tf_generator.G_net(test_real).fake
#
#     saver = tf.train.Saver()
#
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})) as sess:
#         ckpt = tf.train.get_checkpoint_state(tf_path)
#
#         assert ckpt is not None and ckpt.model_checkpoint_path is not None, f"Failed to load checkpoint {tf_path}"
#
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print(f"Tensorflow model checkpoint {ckpt.model_checkpoint_path} loaded")
#
#         tf_weights = {}
#         for v in tf.trainable_variables():
#             tf_weights[v.name] = v.eval()
#
#     return tf_weights
#
#
# def convert_keys(k):
#
#     # 1. divide tf weight name in three parts [block_idx, layer_idx, weight/bias]
#     # 2. handle each part & merge into a pytorch model keys
#
#     k = k.replace("Conv/", "Conv_0/").replace("LayerNorm/", "LayerNorm_0/")
#     keys = k.split("/")[2:]
#
#     is_dconv = False
#
#     # handle C block..
#     if keys[0] == "C":
#         if keys[1] in ["Conv_1", "LayerNorm_1"]:
#             keys[1] = keys[1].replace("1", "5")
#
#         if len(keys) == 4:
#             assert "r" in keys[1]
#
#             if keys[1] == keys[2]:
#                 is_dconv = True
#                 keys[2] = "1.1"
#
#             block_c_maps = {
#                 "1":  "1.2",
#                 "Conv_1":  "2",
#                 "2":  "3",
#             }
#             if keys[2] in block_c_maps:
#                 keys[2] = block_c_maps[keys[2]]
#
#             keys[1] = keys[1].replace("r", "") + ".layers." + keys[2]
#             keys[2] = keys[3]
#             keys.pop(-1)
#     assert len(keys) == 3
#
#     # handle output block
#     if "out" in keys[0]:
#         keys[1] = "0"
#
#     # first part
#     if keys[0] in ["A", "B", "C", "D", "E"]:
#         keys[0] = "block_" + keys[0].lower()
#
#     # second part
#     if "LayerNorm_" in keys[1]:
#         keys[1] = keys[1].replace("LayerNorm_", "") + ".2"
#     if "Conv_" in keys[1]:
#         keys[1] = keys[1].replace("Conv_", "") + ".1"
#
#     # third part
#     keys[2] = {
#         "weights:0": "weight",
#         "w:0": "weight",
#         "bias:0": "bias",
#         "gamma:0": "weight",
#         "beta:0": "bias",
#     }[keys[2]]
#
#     return ".".join(keys), is_dconv
#
#
# def convert_and_save(tf_checkpoint_path, save_name):
#
#     tf_weights = load_tf_weights(tf_checkpoint_path)
#
#     torch_net = Generator()
#     torch_weights = torch_net.state_dict()
#
#     torch_converted_weights = {}
#     for k, v in tf_weights.items():
#         torch_k, is_dconv = convert_keys(k)
#         assert torch_k in torch_weights, f"weight name mismatch: {k}"
#
#         converted_weight = torch.from_numpy(v)
#         if len(converted_weight.shape) == 4:
#             if is_dconv:
#                 converted_weight = converted_weight.permute(2, 3, 0, 1)
#             else:
#                 converted_weight = converted_weight.permute(3, 2, 0, 1)
#
#         assert torch_weights[torch_k].shape == converted_weight.shape, f"shape mismatch: {k}"
#
#         torch_converted_weights[torch_k] = converted_weight
#
#     assert sorted(list(torch_converted_weights)) == sorted(list(torch_weights)), f"some weights are missing"
#     torch_net.load_state_dict(torch_converted_weights)
#     torch.save(torch_net.state_dict(), save_name)
#     print(f"PyTorch model saved at {save_name}")
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--tf_checkpoint_path',
#         type=str,
#         default='AnimeGANv2/checkpoint/generator_Paprika_weight',
#     )
#     parser.add_argument(
#         '--save_name',
#         type=str,
#         default='pytorch_generator_Paprika.pt',
#     )
#     args = parser.parse_args()
#
#     convert_and_save(args.tf_checkpoint_path, args.save_name)