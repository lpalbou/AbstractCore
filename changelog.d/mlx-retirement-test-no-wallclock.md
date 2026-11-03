### Fixed

- **The MLX owner-retirement tests prove "the GC callback did not block" with events, not a stopwatch**
  (`tests/providers/test_mlx_native_owner_retirement_unit.py`). Three tests asserted that a whole `gc.collect()` took
  under 0.5 s. In the Local-model CI job, a full collection alone takes about 0.55 s (torch and transformers loaded,
  thousands of tests' objects alive), so `main` went red at 9d46642 and de1f9c85 with no MLX change. The tests now
  check that the blocked CPU backend, or the blocked close hook, has not returned when `gc.collect()` returns: a
  callback that waited for it could only return after it. There is no wall-clock bound, and the intent is unchanged.
  A mutant whose GC callback joins the worker turns all three red. Test-only.
