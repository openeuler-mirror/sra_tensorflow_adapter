# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op

np.random.seed(140)


def ori_fused_sparse_dynamic_stitch_graph(x, emb_tables):
    x_1 = tf.reshape(x, shape=[-1])  # 将输入 x 展平成一维向量 x_1
    group_ids = tf.math.floormod(x_1, 12)
    group_ids = tf.cast(group_ids, dtype=np.int32)
    chunk_indices = tf.math.floordiv(x_1, 12)
    original_indices = tf.range(0, tf.size(x_1), 1)
    a = tf.dynamic_partition(original_indices, group_ids, num_partitions=12)
    b = tf.dynamic_partition(chunk_indices, group_ids, num_partitions=12)
    c = [tf.gather(emb_tables[i], b[i]) for i in range(12)]
    d = tf.dynamic_stitch(a, c)
    return d


def opt_fused_sparse_dynamic_stitch_graph(x, emb_tables):
    output = gen_embedding_fused_ops.KPFusedSparseDynamicStitch(
        x = x,
        variables = emb_tables
    )
    return output


class TestSparseDynamicStitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize config"""
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 1
        
        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()
        
        # Create tf session
        cls.sess = tf.compat.v1.Session()
        cls.sess.run(tf.compat.v1.global_variables_initializer())

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_base(self):
        variables = []
        max_val = float('inf')
        for i in range(12):
            N_i = np.random.randint(100000, 4473924)
            max_val = min(N_i, max_val)
            var = tf.Variable(
                tf.random.normal([N_i, 10], dtype=tf.float32),  # shape: (N_i, 10)
                name=f"embedding_{i}"
            ) 
            variables.append(var)
            # print(f"Created variable {i}: shape={var.shape}")
        
        x_np = np.random.randint(0, 12*max_val, size=(10000, 12))
        x = tf.constant(x_np, dtype=tf.int64)

        self.sess.run(tf.compat.v1.variables_initializer(variables))

        x_first = self.sess.run(x)
        var_first = self.sess.run(variables[0])

        x_second = self.sess.run(x)
        var_second = self.sess.run(variables[0])
        
        np.testing.assert_allclose(
            x_first,
            x_second,
            rtol=1e-6,
            err_msg="Input values mismatch"
        )

        np.testing.assert_allclose(
            var_first,
            var_second,
            rtol=1e-6,
            err_msg="Input values mismatch"
        )

    def test_kp_sparse_dynamic_stitch(self):
        # Create Graph
        with tf.Graph().as_default():
            num_tables = 12
            emb_dim = 10
            max_val = float('inf')
            # 每张表的 placeholder，行数随机生成
            tables = []
            table_sizes = []
            for i in range(num_tables):
                N_i = np.random.randint(1000000, 44739244)
                table_sizes.append(N_i)
                max_val = min(N_i, max_val)
                table_ph = tf.compat.v1.placeholder(
                    tf.float32, shape=(N_i, emb_dim), name=f"embedding_table_{i}"
                )
                tables.append(table_ph)
            # 生成全局索引 placeholder
            x_shape = (1000, num_tables)
            input_x = tf.compat.v1.placeholder(tf.int64, shape=x_shape, name="input_x")
            # 初始化 feed 数据
            feed = {}
            rng = np.random.default_rng(12345)
            # 为每张表生成随机 embedding 数据
            for i in range(num_tables):
                feed[tables[i]] = rng.standard_normal(size=(table_sizes[i], emb_dim)).astype(np.float32)
            # 生成索引数据（保持原逻辑：范围是 0 ~ num_tables * max_val - 1）
            feed[input_x] = rng.integers(
                low=0, high=num_tables * max_val, size=x_shape, dtype=np.int64
            )
            with tf.name_scope("ori"):
                out_ori = ori_fused_sparse_dynamic_stitch_graph(input_x, tables)
            with tf.name_scope("opt"):
                out_opt = opt_fused_sparse_dynamic_stitch_graph(input_x, tables)
            
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                out_ori_val = sess.run(
                    [out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori
                )
                out_opt_val = sess.run(
                    [out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt
                )
                
                np.testing.assert_array_equal(
                    out_ori_val,
                    out_opt_val,
                    err_msg="result mismatch"
                )
                
                benchmark_op(
                    sess,
                    feed,
                    [out_ori],
                    [out_opt],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedSparseDynamicStitch",
                    start_op="ori/Reshape",
                    end_op="ori/DynamicStitch",
                    num_runs=100,
                    tag="--------TF_origin---------"
                )


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=1)