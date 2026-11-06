//! Fork-safe scoped parallelism — NO global thread pool.
//!
//! The executor forks children (sandbox zygote, audit ProcessPoolExecutor);
//! a lazily-initialized global pool (rayon) would be inherited in a broken
//! state by the child. `std::thread::scope` spawns and joins fresh threads
//! per call, so a forked child can never observe a half-built pool.

/// Run `f(i)` for i in 0..n across `threads` scoped threads.
/// Threads are partitioned contiguously — deterministic per input.
#[allow(dead_code)]
pub fn par_for<F>(n: usize, threads: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n == 0 {
        return;
    }
    let t = threads.max(1).min(n);
    if t == 1 {
        for i in 0..n {
            f(i);
        }
        return;
    }
    let chunk = n.div_ceil(t);
    std::thread::scope(|s| {
        let f = &f;
        for c in 0..t {
            let lo = c * chunk;
            let hi = (lo + chunk).min(n);
            if lo >= hi {
                break;
            }
            s.spawn(move || {
                for i in lo..hi {
                    f(i);
                }
            });
        }
    });
}

/// Chunk a mutable slice and run `f(piece, global_offset)` per chunk across
/// scoped threads — safe indexed parallel writes, deterministic.
pub fn par_chunks<T, F>(items: &mut [T], threads: usize, f: F)
where
    T: Send,
    F: Fn(&mut [T], usize) + Sync,
{
    let n = items.len();
    if n == 0 {
        return;
    }
    let t = threads.max(1).min(n);
    if t == 1 {
        f(items, 0);
        return;
    }
    let chunk = n.div_ceil(t);
    std::thread::scope(|s| {
        let f = &f;
        let mut off = 0usize;
        for piece in items.chunks_mut(chunk) {
            let o = off;
            off += piece.len();
            s.spawn(move || f(piece, o));
        }
    });
}
