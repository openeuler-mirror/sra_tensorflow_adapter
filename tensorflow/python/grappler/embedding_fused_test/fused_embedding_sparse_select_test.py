# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op

np.random.seed(140)


def ori_sparse_select_graph(input_a, input_b, input_c):
    a = tf.reshape(input_a, [-1, 1])
    b = tf.reshape(input_b, [-1, 1])
    c = tf.reshape(input_c, [-1, 1])

    greater_a = tf.greater(a, 0)
    shape_reshape_a1 = tf.shape(a)
    shape_reshape_a2 = tf.shape(a)
    fill_a1 = tf.fill(shape_reshape_a1, tf.constant(1, dtype=tf.float32))
    realdiv = tf.realdiv(fill_a1, tf.constant(1, dtype=tf.float32))
    output_x = tf.fill(shape_reshape_a2, tf.constant(0, dtype=tf.float32))  # 全部填充为0，第一个输出
    cast_a = tf.cast(greater_a, tf.float32)
    shape_a = tf.shape(cast_a)
    fill_a = tf.fill(shape_a, tf.constant(1, dtype=tf.float32))
    equal_4563 = tf.equal(b, 4563)
    equal_10831 = tf.equal(b, 10831)
    equal_3 = tf.equal(c, 3)
    select_1 = tf.where(equal_4563, fill_a, cast_a)
    select_2 = tf.where(equal_10831, fill_a, select_1)
    output_y = tf.subtract(tf.constant(1, dtype=tf.float32), select_2)  # 1 - select_2 ？
    mul = tf.multiply(tf.constant(1, dtype=tf.float32), select_2)  # Select.2415 --> Mul.2419
    select_3 = tf.where(equal_3, realdiv, fill_a1)  # realdiv和fill_a1 一样的？
    output_z = tf.concat([mul, select_3], axis=-1)
    return output_x, output_y, output_z


def opt_sparse_select_graph(input_a, input_b, input_c):
    output_x, output_y, output_z = gen_embedding_fused_ops.KPFusedSparseSelect(
        input_a=input_a, input_b=input_b, input_c=input_c
    )
    return output_x, output_y, output_z


class TestKPFusedSparseSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize config"""
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 1

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        return

    def _run_kp_select_test(self, a_shape, b_shape, c_shape, num_runs):
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int32, name="input_a")
            input1 = tf.compat.v1.placeholder(tf.int32, name="input_b")
            input2 = tf.compat.v1.placeholder(tf.int32, name="input_c")
            """Initialize test data"""
            feed = {
                input0: np.random.randint(0, 100, size=a_shape).astype(np.int32),
                input1: np.random.randint(0, 100, size=b_shape).astype(np.int32),
                input2: np.random.randint(0, 100, size=c_shape).astype(np.int32),
            }

            with tf.name_scope("ori"):
                out_ori1, out_ori2, out_ori3 = ori_sparse_select_graph(input0, input1, input2)
            with tf.name_scope("opt"):
                out_opt1, out_opt2, out_opt3 = opt_sparse_select_graph(input0, input1, input2)
            
            with tf.compat.v1.Session(config=self.config) as sess:
                out_ori_val1, out_ori_val2, out_ori_val3 = sess.run(
                    [out_ori1, out_ori2, out_ori3], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_ori
                )
                out_opt_val1, out_opt_val2, out_opt_val3 = sess.run(
                    [out_opt1,out_opt2, out_opt3], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_opt
                )
                
                # 功能测试
                np.testing.assert_allclose(
                    out_ori_val1,
                    out_opt_val1,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )

                np.testing.assert_allclose(
                    out_ori_val2,
                    out_opt_val2,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )

                np.testing.assert_allclose(
                    out_ori_val3,
                    out_opt_val3,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )
                
                benchmark_op(
                    sess,
                    feed,
                    [out_ori1, out_ori2, out_ori3],
                    [out_opt1, out_opt2, out_opt3],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedSparseSelect",
                    start_op="ori/Reshape",
                    end_op="ori/Sub",
                    num_runs=num_runs,
                    tag="-----TF_origin-----"
                )


    def test_fused_embedding_sparse_select(self):
        shapes = [
            [(i,), (i,), (i,)] for i in range(1, 101)
        ]  # 新添加的测试案例，shape组中abc的shape都一样，而且大小在1~100之间
        shapes.append([(100, 10), (10, 100), (20, 50)])
        shapes.extend([[(i, i,), (i, i,), (i, i,)] for i in range(1, 101)])
        shapes.extend([[(i, i, i,), (i, i, i,), (i, i, i,)] for i in range(1, 101)])
        for shape in shapes:
            self._run_kp_select_test(*shape, num_runs=10)
            print(f"tested shape_a {shape[0]}")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)