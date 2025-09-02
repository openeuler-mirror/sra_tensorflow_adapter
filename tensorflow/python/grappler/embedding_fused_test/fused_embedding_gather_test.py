# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op


def ori_fused_embedding_gather_graph(data, slice_input, begin):
    slice_out = tf.strided_slice(
        slice_input,
        begin=begin,
        end=[tf.shape(slice_input)[0], begin[1] + 2],
        strides=[1, 1],
        begin_mask=1,
        end_mask=1,
        shrink_axis_mask=2
    )
    
    value, indices = tf.unique(slice_out)
    value = tf.reshape(value, [-1])
    value_1, indices_1 = tf.unique(value)
    gather1 = tf.gather(data, value_1)
    gather2 = tf.gather(gather1, indices_1)
    return value, indices, gather2


def opt_fused_embedding_gather_graph(data, slice_input, begin):
    custom_out1, custom_out2, custom_out3 = gen_embedding_fused_ops.KPFusedGather(
        data=data,
        slice_input=slice_input,
        begin=begin
    )
    return custom_out1, custom_out2, custom_out3


class TestFusedGather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize"""        
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 1

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        return 

    def _run_kp_gather_test(self, data_shape, slice_shape, base_data, base_slice_input, base_begin, num_runs):
        with tf.Graph().as_default():
            data = tf.compat.v1.placeholder(tf.float32, shape=data_shape, name="data")
            slice_input = tf.compat.v1.placeholder(tf.int64, shape=slice_shape, name="slice_input")
            begin = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="begin")

            feed = {
                data: base_data,
                slice_input: base_slice_input,
                begin: base_begin
            }

            # original graph
            with tf.name_scope("ori"):
                out_ori1, out_ori2, out_ori3 = ori_fused_embedding_gather_graph(data, slice_input, begin)

            # optimized graph
            with tf.name_scope("opt"):
                out_opt1, out_opt2, out_opt3 = opt_fused_embedding_gather_graph(data, slice_input, begin)

            with tf.compat.v1.Session(config=self.config) as sess:
                # run ori
                out_ori_val1, out_ori_val2, out_ori_val3 = sess.run(
                    [out_ori1, out_ori2, out_ori3],
                    feed_dict=feed,
                    options=self.run_options,
                    run_metadata=self.run_metadata_ori
                )
                # run opt
                out_opt_val1, out_opt_val2, out_opt_val3 = sess.run(
                    [out_opt1, out_opt2, out_opt3],
                    feed_dict=feed,
                    options=self.run_options,
                    run_metadata=self.run_metadata_opt
                )

                # 功能测试
                np.testing.assert_array_equal(
                    out_ori_val1,
                    out_opt_val1,
                    err_msg="Segment count mismatch"
                )
                np.testing.assert_array_equal(
                    out_ori_val2,
                    out_opt_val2,
                    err_msg="Segment count mismatch"
                )
                np.testing.assert_allclose(
                    out_opt_val3,
                    out_ori_val3,
                    rtol=1e-6,
                    err_msg="Output values mismatch"
                )

                # benchmark
                benchmark_op(
                    sess,
                    feed,
                    [out_ori1, out_ori2, out_ori3],
                    [out_opt1, out_opt2, out_opt3],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedGather",
                    start_op="ori/strided_slice_1",
                    end_op="ori/GatherV2_1",
                    num_runs=num_runs,
                    tag="--TF_origin--"
                )


    def test_kp_embedding_gather(self):
        base_data = np.linspace(0, 11, num=240, endpoint=False, dtype=np.float32).reshape(20, 12)
        base_slice_input = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
        base_begin = np.array([0, 1], dtype=np.int32)
        self._run_kp_gather_test((20, 12), (3, 2), base_data, base_slice_input, base_begin, num_runs=100)

    def test_kp_gather_with_duplicates(self):
        base_data = np.random.rand(100, 12).astype(np.float32)
        base_slice_input = np.array([[5, 3], [7, 3], [9, 4], [5, 3]], dtype=np.int64)
        base_begin = np.array([0, 1], dtype=np.int32)
        self._run_kp_gather_test((100, 12), (4, 2), base_data, base_slice_input, base_begin, num_runs=100)
        
    def test_kp_gather_single_unique(self):
        base_data = np.random.rand(50, 12).astype(np.float32)
        base_slice_input = np.array([[10, 7], [20, 7], [30, 7]], dtype=np.int64)
        base_begin = np.array([0, 1], dtype=np.int32)
        self._run_kp_gather_test((50, 12), (3, 2), base_data, base_slice_input, base_begin, num_runs=100)

    def test_kp_gather_262145(self):
        base_data = np.linspace(0, 11111, num=262145*12, dtype=np.float32).reshape(262145, 12)
        base_slice_input = np.random.randint(0, 262146, size=(46, 2), dtype=np.int64)
        base_begin = np.array([0, 0], dtype=np.int32)
        self._run_kp_gather_test((262145, 12), (46, 2), base_data, base_slice_input, base_begin, num_runs=100)
    

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)