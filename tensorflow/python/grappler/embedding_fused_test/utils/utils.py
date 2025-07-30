import timeit

from tensorflow.python.client import timeline


def perf_run(ori_func, opt_func, name, warmup=5, iters=50):
    
    print(f"\nWarmup ori: {warmup} iters")
    for _ in range(warmup):
        ori_func()
    
    print(f"Running performance test: ori {iters} iters")
    total_time = timeit.timeit(ori_func, number=iters)
    ori_avg_time = total_time / iters * 1000
    print(f"{name}: {ori_avg_time:.2f} ms per run")

    print(f"\nWarmup opt: {warmup} iters")
    for _ in range(warmup):
        opt_func()

    print(f"Running performance test: opt {iters} iters")
    total_time = timeit.timeit(opt_func, number=iters)
    opt_avg_time = total_time / iters * 1000
    print(f"{name}: {opt_avg_time:.2f} ms per run")

    improvement = (ori_avg_time - opt_avg_time) / ori_avg_time * 100
    print(f"improve: {improvement:.2f}%")


def generate_timeline(step_stats, filename):
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format()

    with open(f"timeline/{filename}.json", "w") as f:
        f.write(ctf)


def wrapper_sess(sess, fetches, feed_dict=None, options=None, run_metadata=None):
                    return lambda: sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)