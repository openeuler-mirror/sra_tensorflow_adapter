# Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import benchmark_op


def ori_fused_embedding_sparse_segment_reduce_graph(data, indices, slice_input, begin, end, strides, is_mean):
    slice_out = tf.strided_slice(
            slice_input,
            begin=begin,
            end=end,
            strides=strides,
            begin_mask=1,
            end_mask=1,
            shrink_axis_mask=2
        )
        
    segment_ids = tf.cast(slice_out, dtype=tf.int32)
    if is_mean:
        output = tf.sparse.segment_mean(
            data=data,
            indices=indices,
            segment_ids=segment_ids
        )
    else:
        output = tf.sparse.segment_sum(
            data=data,
            indices=indices,
            segment_ids=segment_ids
        )
    
    output_shape = tf.shape(output)
    slice_out = tf.strided_slice(output_shape, begin=[0], end=[1], strides=[1])
    
    return output, slice_out


def opt_fused_embedding_sparse_segment_reduce_graph(data, indices, slice_input, begin, end, strides, is_mean):
    if is_mean:
        custom_out, custom_slice_out = gen_embedding_fused_ops.KPFusedSparseSegmentReduce(
                data=data,
                indices=indices,
                slice_input=slice_input,
                begin=begin,
                end=end,
                strides=strides
            )
    else:
        custom_out, custom_slice_out = gen_embedding_fused_ops.KPFusedSparseSegmentReduce(
                data=data,
                indices=indices,
                slice_input=slice_input,
                begin=begin,
                end = end,
                strides=strides,
                combiner=0
            )
    return custom_out, custom_slice_out


class TestSparseSegmentMeanSlice(unittest.TestCase):
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

    def test_mean(self):
        with tf.Graph().as_default():
            data = tf.compat.v1.placeholder(tf.float32, shape=(4,3), name="data")
            indices = tf.compat.v1.placeholder(tf.int32, shape=(3,), name="indices")
            slice_input = tf.compat.v1.placeholder(tf.int64, shape=(3,2), name="slice_input")
            begin = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="begin")
            end = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="end")
            strides = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="strides")
            
            base_data = np.array(
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]], 
                dtype=np.float32
            ) # shape {4， 3}
            base_indices = np.array([0, 1, 2], dtype=np.int64) # shape {3}
            base_slice_input = np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int64) # shape {3, 2}
            base_begin = [0, 1]
            base_end = [0, 2]
            base_strides = [1, 2]
            
            feed = {
                data: base_data,
                indices: base_indices,
                slice_input: base_slice_input,
                begin: base_begin,
                end: base_end,
                strides: base_strides
            }
            
            with tf.name_scope("ori"):
                out_ori1, out_ori2 = ori_fused_embedding_sparse_segment_reduce_graph(
                    data, indices, slice_input, begin, end, strides, True
                )
            with tf.name_scope("opt"):
                out_opt1, out_opt2 = opt_fused_embedding_sparse_segment_reduce_graph(
                    data, indices, slice_input, begin, end, strides, True
                )
            
            with tf.compat.v1.Session(config=self.config) as sess:
                out_ori_val1, out_ori_val2 = sess.run(
                    [out_ori1, out_ori2], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_ori
                )
                out_opt_val1, out_opt_val2 = sess.run(
                    [out_opt1, out_opt2], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_opt
                )

                np.testing.assert_allclose(
                    out_opt_val1,
                    out_ori_val1,
                    rtol=1e-6,
                    err_msg="Output values mismatch"
                )
                np.testing.assert_array_equal(
                    out_opt_val2,
                    out_ori_val2,
                    err_msg="Segment count mismatch"
                )
                
                benchmark_op(
                    sess,
                    feed,
                    [out_ori1, out_ori2],
                    [out_opt1, out_opt2],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedSparseSegmentReduce",
                    start_op="ori/StridedSlice",
                    end_op="ori/StridedSlice_1",
                    num_runs=500,
                    tag="--------TF_origin---------"
                )

    
    def test_sum(self):
        with tf.Graph().as_default():
            data = tf.compat.v1.placeholder(tf.float32, shape=(4,3), name="data")
            indices = tf.compat.v1.placeholder(tf.int32, shape=(3,), name="indices")
            slice_input = tf.compat.v1.placeholder(tf.int64, shape=(3,2), name="slice_input")
            begin = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="begin")
            end = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="end")
            strides = tf.compat.v1.placeholder(tf.int32, shape=(2,), name="strides")
            
            base_data = np.array(
                [[1.0, 2.0, 3.0], [3.0, 4.0,5.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]], 
                dtype=np.float32
            ) # shape {4， 3}
            base_indices = np.array([0, 1, 2], dtype=np.int64)
            base_slice_input = np.array([[0, 0], [0, 2], [1, 2]], dtype=np.int64) 
            base_begin = [0, 1]
            base_end = [0, 2]
            base_strides = [1, 2]
            
            feed = {
                data: base_data,
                indices: base_indices,
                slice_input: base_slice_input,
                begin: base_begin,
                end: base_end,
                strides: base_strides
            }
            with tf.name_scope("ori"):
                out_ori1, out_ori2 = ori_fused_embedding_sparse_segment_reduce_graph(
                    data, indices, slice_input, begin, end, strides, False
                )
            with tf.name_scope("opt"):
                out_opt1, out_opt2 = opt_fused_embedding_sparse_segment_reduce_graph(
                    data,indices, slice_input, begin, end, strides, False
                )
            
            with tf.compat.v1.Session(config=self.config) as sess:
                out_ori_val1, out_ori_val2 = sess.run(
                    [out_ori1, out_ori2], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_ori
                )
                out_opt_val1, out_opt_val2 = sess.run(
                    [out_opt1, out_opt2], 
                    feed_dict=feed, 
                    options=self.run_options, 
                    run_metadata=self.run_metadata_opt
                )
                np.testing.assert_allclose(
                    out_opt_val1,
                    out_ori_val1,
                    rtol=1e-6,
                    err_msg="Output values mismatch"
                )
                np.testing.assert_array_equal(
                    out_opt_val2,
                    out_ori_val2,
                    err_msg="Segment count mismatch"
                )
                
                benchmark_op(
                    sess,
                    feed,
                    [out_ori1, out_ori2],
                    [out_opt1, out_opt2],
                    self.run_options,
                    self.run_metadata_ori,
                    self.run_metadata_opt,
                    op_name="KPFusedSparseSegmentReduce",
                    start_op="ori/StridedSlice",
                    end_op="ori/StridedSlice_1",
                    num_runs=1000,
                    tag="---------TF_origin--------"
                )


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)