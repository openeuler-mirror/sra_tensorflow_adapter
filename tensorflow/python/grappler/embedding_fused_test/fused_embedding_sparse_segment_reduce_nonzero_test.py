import tensorflow as tf
import numpy as np
import unittest

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import perf_run, generate_timeline, wrapper_sess


def ori_fused_embedding_sparse_segment_reduce_nonzero(data, indices, slice_input, begin, end, strides, combiner):
    slice_out = tf.strided_slice(
                slice_input,
                begin= begin,
                end= end,
                strides= strides,
                begin_mask=1,
                end_mask=1,
                shrink_axis_mask=2
            )
            
    segment_ids = tf.cast(slice_out, dtype=tf.int32)

    if combiner ==1 :
        sparseseg_out = tf.sparse.segment_mean(
                data = data,
                indices = indices,
                segment_ids= segment_ids
            )
    else:
        sparseseg_out = tf.sparse.segment_sum(
                data = data,
                indices = indices,
                segment_ids= segment_ids
            )
    zero = tf.zeros_like(sparseseg_out)
    notequal = tf.not_equal(x=sparseseg_out, y = zero)
    where_out = tf.where(notequal)
    output_shape = tf.cast(where_out, dtype=tf.int64)
    output_data = tf.gather_nd(params=sparseseg_out, indices=where_out)
    shape = tf.shape(sparseseg_out)
    output_ids = tf.cast(shape, dtype=tf.int64)

    return output_shape, output_ids, output_data
    

def opt_fused_embedding_sparse_segment_reduce_nonzero(data, indices, slice_input, begin, end, strides, combiner):
    output_shape, output_ids, output_data = gen_embedding_fused_ops.KPFusedSparseSegmentReduceNonzero(
        data=data,
        indices=indices,
        slice_input=slice_input,
        begin=begin,
        end = end,
        strides = strides,
        combiner=combiner
        )
    return output_shape, output_ids, output_data

class TestKPFusedSparseSegmentReduceNonzero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize config"""
        cls.config = tf.compat.v1.ConfigProto()
        cls.config.intra_op_parallelism_threads = 16
        cls.config.inter_op_parallelism_threads = 16

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        return

    def test_fused_embedding_sparse_select(self):
        combiners = [0, 1]
        
        for combiner in combiners:
            # Create Graph
            with tf.Graph().as_default():
                input0 = tf.compat.v1.placeholder(tf.float32, shape=[1449], name="input_a")
                input1 = tf.compat.v1.placeholder(tf.int64, shape=[5742], name="input_b")
                input2 = tf.compat.v1.placeholder(tf.int64, shape=[5742, 2], name="input_c")
                input3 = tf.compat.v1.placeholder(tf.int32, shape=[2], name="input_d")
                input4 = tf.compat.v1.placeholder(tf.int32, shape=[2], name="input_e")
                input5 = tf.compat.v1.placeholder(tf.int32, shape=[2], name="input_f")
                
                """Initialize test data"""
                data = np.random.rand(1449).astype(np.float32) * 10
                zero_prob = 0.3
                mask = np.random.rand(1449) > zero_prob
                data[~mask] = 0

                indices = np.random.randint(0, 1449, size=5742, dtype=np.int64)
                
                start_points = np.sort(np.random.choice(np.arange(0, 15660), size=5742, replace=False))
                end_points = start_points + np.random.randint(1, 100, size=5742)
                end_points = np.minimum(end_points, 15661)
                slice_input = np.column_stack((start_points, end_points))
                slice_input[:, 1] = slice_input[:, 0]
                slice_input[-2, 1] = 15659

                begin = np.array([0, 1], dtype=np.int32)
                end = np.array([0, 2], dtype=np.int32)
                strides = np.array([1, 2], dtype=np.int32)

                feed = {
                    input0: data,
                    input1: indices,
                    input2: slice_input,
                    input3: begin,
                    input4: end,
                    input5: strides
                }
                with tf.name_scope("ori"):
                    out0_ori, out1_ori, out2_ori = ori_fused_embedding_sparse_segment_reduce_nonzero(
                        input0, input1, input2, input3, input4, input5, combiner)
                with tf.name_scope("opt"):
                    out0_opt, out1_opt, out2_opt = opt_fused_embedding_sparse_segment_reduce_nonzero(
                        input0, input1, input2, input3, input4, input5, combiner)
 
                with tf.compat.v1.Session(config=self.config) as sess:
                    out0_ori_val, out1_ori_val, out2_ori_val = sess.run(
                        [out0_ori, out1_ori, out2_ori], 
                        feed_dict=feed, 
                        options=self.run_options, 
                        run_metadata=self.run_metadata_ori
                    )
                    out0_opt_val, out1_opt_val, out2_opt_val = sess.run(
                        [out0_opt, out1_opt, out2_opt], 
                        feed_dict=feed, 
                        options=self.run_options, 
                        run_metadata=self.run_metadata_opt
                    )

                    np.testing.assert_allclose(
                        out0_ori_val,
                        out0_opt_val,
                        rtol=1e-5,
                        err_msg=f"Output values mismatch when combiner={combiner}"
                    )

                    np.testing.assert_allclose(
                        out1_ori_val,
                        out1_opt_val,
                        rtol=1e-5,
                        err_msg=f"Output values mismatch when combiner={combiner}"
                    )

                    np.testing.assert_allclose(
                        out2_ori_val,
                        out2_opt_val,
                        rtol=1e-5,
                        err_msg=f"Output values mismatch when combiner={combiner}"
                    )

                    generate_timeline(
                        self.run_metadata_ori.step_stats, 
                        f"{self._testMethodName}_ori_combiner_{combiner}"
                    )
                    generate_timeline(
                        self.run_metadata_opt.step_stats, 
                        f"{self._testMethodName}_opt_combiner_{combiner}"
                    )

                    perf_run(
                        wrapper_sess(sess, [out0_ori, out1_ori, out2_ori], feed_dict=feed), 
                        wrapper_sess(sess, [out0_opt, out1_opt, out2_opt], feed_dict=feed), 
                        f"KPFusedSparseSegmentReduceNonzero_combiner_{combiner}"
                    )

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)