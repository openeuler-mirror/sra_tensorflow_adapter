import tensorflow as tf
import numpy as np
import unittest

from tensorflow.python.ops import gen_embedding_fused_ops
from utils.utils import perf_run, generate_timeline, wrapper_sess


def ori_fused_embedding_sparse_select_graph(input_a, input_b, input_c):
    a = tf.reshape(input_a, [-1, 1])
    b = tf.reshape(input_b, [-1, 1])
    c = tf.reshape(input_c, [-1, 1])

    greater_a = tf.greater(a, 0)
    shape_reshape_a1 = tf.shape(a)
    shape_reshape_a2 = tf.shape(a)
    fill_a1 = tf.fill(shape_reshape_a1, tf.constant(1, dtype=tf.float32))
    realdiv = tf.realdiv(fill_a1, tf.constant(1, dtype=tf.float32))
    output_x = tf.fill(shape_reshape_a2, tf.constant(0, dtype=tf.float32))
    cast_a = tf.cast(greater_a, tf.float32)
    shape_a = tf.shape(cast_a)
    fill_a = tf.fill(shape_a, tf.constant(1, dtype=tf.float32))
    equal_4563 = tf.equal(b, 4563)
    equal_10831 = tf.equal(b, 10831)
    equal_3 = tf.equal(c, 3)
    select_1 = tf.where(equal_4563, fill_a, cast_a)
    select_2 = tf.where(equal_10831, fill_a, select_1)
    output_y = tf.subtract(tf.constant(1, dtype=tf.float32), select_2)
    mul = tf.multiply(tf.constant(1, dtype=tf.float32), select_2)
    select_3 = tf.where(equal_3, realdiv, fill_a1)
    output_z = tf.concat([mul, select_3], axis=-1)
    return output_x, output_y, output_z


def opt_fused_embedding_sparse_select_graph(input_a, input_b, input_c):
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
        cls.config.inter_op_parallelism_threads = 16

        cls.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        cls.run_metadata_ori = tf.compat.v1.RunMetadata()
        cls.run_metadata_opt = tf.compat.v1.RunMetadata()

    @classmethod
    def tearDownClass(cls):
        return

    def test_fused_embedding_sparse_select(self):
        # Create Graph
        with tf.Graph().as_default():
            input0 = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input_a")
            input1 = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input_b")
            input2 = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input_c")
            """Initialize test data"""
            feed = {
                input0: np.random.randint(0, 100, size=(100, 10)).astype(np.int32),
                input1: np.random.randint(0, 100, size=(10, 100)).astype(np.int32),
                input2: np.random.randint(0, 100, size=(20, 50)).astype(np.int32),
            }
            with tf.name_scope("ori"):
                out0_ori, out1_ori, out2_ori = ori_fused_embedding_sparse_select_graph(input0, input1, input2)
            with tf.name_scope("opt"):
                out0_opt, out1_opt, out2_opt = opt_fused_embedding_sparse_select_graph(input0, input1, input2)
        
            # Create tf session
            with tf.compat.v1.Session(config=self.config) as sess:
                # functest
                out0_ori_val, out1_ori_val, out2_ori_val = sess.run([out0_ori, out1_ori, out2_ori], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_ori)
                out0_opt_val, out1_opt_val, out2_opt_val = sess.run([out0_opt, out1_opt, out2_opt], feed_dict=feed, options=self.run_options, run_metadata=self.run_metadata_opt)

                np.testing.assert_allclose(
                    out0_ori_val,
                    out0_opt_val,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )

                np.testing.assert_allclose(
                    out1_ori_val,
                    out1_opt_val,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )

                np.testing.assert_allclose(
                    out2_ori_val,
                    out2_opt_val,
                    rtol=1e-5,
                    err_msg="Output values mismatch"
                )

                generate_timeline(self.run_metadata_ori.step_stats, f"{self._testMethodName}_ori")
                generate_timeline(self.run_metadata_opt.step_stats, f"{self._testMethodName}_opt")

                # perftest
                perf_run(wrapper_sess(sess, [out0_ori, out1_ori, out2_ori], feed_dict=feed), 
                         wrapper_sess(sess, [out0_opt, out1_opt, out2_opt], feed_dict=feed), 
                         "KPFusedEmbeddingSparseSelect")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    unittest.main(argv=[''], verbosity=2)