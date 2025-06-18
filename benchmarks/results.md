# Throughput results

No throughput comparison has been run in this repo.

`src/bench/throughput.py` contains a harness sketch but the two paths
measure different things: the JAX path runs warmup then times an
in-process step loop (excluding compile), while the torch-xla path
subprocess-times a whole interpreter launch (including compile). The two
sides are not comparable and were never used to produce a like-for-like
number.

If you want to run a fair comparison you would need to (a) time in-process
on both sides, (b) match the warmup/compile handling, (c) match the
optimizer, schedule, precision and gradient-accumulation config on both
sides, and (d) run on the same TPU slice.
