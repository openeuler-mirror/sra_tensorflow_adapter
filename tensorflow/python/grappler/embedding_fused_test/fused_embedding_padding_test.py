# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op

np.random.seed(140)


def opt_padding_fast_graph(input0, input1, input2, input3):
    # execute custom op
    _, custom_out = gen_embedding_fused_ops.kp_fused_embedding_padding_fast(input0, input1, input2, input3)
    return custom_out


def opt_padding_graph(input0, input1, input2, input3):
    # execute custom op
    _, custom_out = gen_embedding_fused_ops.kp_fused_embedding_padding(input0, input1, input2, input3)
    return custom_out


def ori_padding_fast_graph(input0, input1, input2, input3):
    cast = tf.cast(input0, tf.int32)
    begin = tf.constant([0], dtype=tf.int32)
    end = tf.constant([1], dtype=tf.int32)
    strides = tf.constant([1], dtype=tf.int32)
    hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    sub_out = hash_rows - input2
    const = tf.constant(input1.shape[1], dtype=tf.int32)
    pack = tf.stack([sub_out, const], axis=0)
    fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
    concat = tf.concat([input1, fill], 0)
    reshape = tf.reshape(concat, input3)
    shape_tensor = tf.shape(reshape)
    output = tf.strided_slice(shape_tensor, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    return output


def ori_padding_graph(input0, input1, input2, input3):
    cast = tf.cast(input0, tf.int32)
    begin = tf.constant([0], dtype=tf.int32)
    end = tf.constant([1], dtype=tf.int32)
    strides = tf.constant([1], dtype=tf.int32)
    hash_rows = tf.strided_slice(cast, begin=begin, end=end, strides=strides, shrink_axis_mask=1)
    sub_out = hash_rows - input2
    const = tf.constant(input1.shape[1], dtype=tf.int32)
    pack = tf.stack([sub_out, const], axis=0)
    fill = tf.fill(pack, tf.constant(0, dtype=tf.float32))
    concat = tf.concat([input1, fill], 0)
    output = tf.reshape(concat, input3)
    return output


class TestFusedEmbeddingPadding(unittest.TestCase):
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
    
    def _run_kp_padding_test(self, input1_shape, input3_shape, num_runs=500):
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int64, shape=(2,), name="input0")
            input1 = tf.compat.v1.placeholder(tf.float32, shape=input1_shape, name="input1")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=(), name="input2")
            input3 = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="input3")
            """Initialize test data"""
            feed = {
                input0: np.array([6, input1_shape[1]]).astype(np.int64),
                input1: np.random.rand(*input1_shape).astype(np.float),
                input2: input1_shape[0],
                input3: np.array(input3_shape).astype(np.int32),
            }
            with tf.name_scope("ori"):
                out_ori = ori_padding_graph(input0, input1, input2, input3)
            with tf.name_scope("opt"):
                out_opt = opt_padding_graph(input0, input1, input2, input3)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                ori_result = sess.run(
                    [out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori
                )
                opt_result = sess.run(
                    [out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt
                )

                np.testing.assert_array_equal(
                    ori_result,
                    opt_result,
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
                    op_name="KPFusedEmbeddingPadding",
                    start_op="ori/Cast",
                    end_op="ori/Reshape",
                    num_runs=num_runs,
                    tag="-------TF_origin-------"
                )


    def _run_kp_padding_fast_test(self, input1_shape, input3_shape, num_runs=500):
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int64, shape=(2,), name="input0")
            input1 = tf.compat.v1.placeholder(tf.float32, shape=input1_shape, name="input1")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=(), name="input2")
            input3 = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="input3")
            """Initialize test data"""
            feed = {
                input0: np.array([6, input1_shape[1]]).astype(np.int64),
                input1: np.random.rand(*input1_shape).astype(np.float),
                input2: input1_shape[0],
                input3: np.array(input3_shape).astype(np.int32),
            }
            with tf.name_scope("ori"):
                out_ori = ori_padding_fast_graph(input0, input1, input2, input3)
            with tf.name_scope("opt"):
                out_opt = opt_padding_fast_graph(input0, input1, input2, input3)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                ori_result = sess.run(
                    [out_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori
                )
                opt_result = sess.run(
                    [out_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt
                )

                np.testing.assert_array_equal(
                    ori_result,
                    opt_result,
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
                    op_name="KPFusedEmbeddingPaddingFast",
                    start_op="ori/Cast",
                    end_op="ori/StridedSlice_1",
                    num_runs=num_runs,
                    tag="---------TF_origin---------"
                )

    
    def test_kp_padding_shape10(self):
        input1_shape = (4, 10)
        input3_shape = (-1, 20)
        self._run_kp_padding_test(input1_shape, input3_shape, num_runs=100)

    def test_kp_padding_shape12(self):
        input1_shape = (1, 12)
        input3_shape = (-1, 36)
        self._run_kp_padding_test(input1_shape, input3_shape, num_runs=100)
    
    def test_kp_padding_fast_shape10(self):
        input1_shape = (4, 10)
        input3_shape = (-1, 20)
        self._run_kp_padding_fast_test(input1_shape, input3_shape, num_runs=100)

    def test_kp_padding_fast_shape12(self):
        input1_shape = (1, 12)
        input3_shape = (-1, 36)
        self._run_kp_padding_fast_test(input1_shape, input3_shape, num_runs=100)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)