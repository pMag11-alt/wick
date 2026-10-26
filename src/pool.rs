//! A persistent thread pool for fork-join parallel loops.
//!
//! Decoding one token runs a few hundred small matrix-vector products, each
//! taking tens of microseconds, so workers must already be awake when a
//! job arrives. They spin briefly on an epoch counter, then park on a
//! condition variable so an idle pool costs no CPU.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Task = dyn Fn(usize) + Sync;

struct Job {
    task: *const Task,
    n: usize,
    next: AtomicUsize,
}

struct Shared {
    epoch: AtomicUsize,
    job: Mutex<Option<JobPtr>>,
    finished: AtomicUsize,
    quit: AtomicBool,
    sleeping: Mutex<usize>,
    wake: Condvar,
}

#[derive(Clone, Copy)]
struct JobPtr(*const Job);
// SAFETY: the job outlives every worker's use of it (run() waits).
unsafe impl Send for JobPtr {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    lock: Mutex<()>,
}

const SPIN: u32 = 1 << 14;

impl Pool {
    /// A pool with `threads` threads in total (the caller counts as one).
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            job: Mutex::new(None),
            finished: AtomicUsize::new(0),
            quit: AtomicBool::new(false),
            sleeping: Mutex::new(0),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new().name(format!("wick-{i}")).spawn(move || worker(&s)).unwrap()
            })
            .collect();
        Pool { shared, workers, lock: Mutex::new(()) }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(i)` for every i in 0..n across the pool and wait for all.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if self.workers.is_empty() || n == 1 {
            (0..n).for_each(f);
            return;
        }
        let _one_job_at_a_time = self.lock.lock().unwrap();
        // SAFETY: we erase the closure's lifetime; it stays valid because we
        // do not return until every worker has finished with the job.
        let task: *const Task = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static Task>(f) };
        let job = Job { task, n, next: AtomicUsize::new(0) };
        let s = &*self.shared;
        s.finished.store(0, Ordering::Relaxed);
        *s.job.lock().unwrap() = Some(JobPtr(&job));
        s.epoch.fetch_add(1, Ordering::Release);
        if *s.sleeping.lock().unwrap() > 0 {
            s.wake.notify_all();
        }
        work_on(&job);
        let mut spins = 0u32;
        while s.finished.load(Ordering::Acquire) < self.workers.len() {
            spins += 1;
            if spins < SPIN { std::hint::spin_loop() } else { std::thread::yield_now() }
        }
        *s.job.lock().unwrap() = None;
    }

    /// Split 0..len into chunks of at least `min_chunk` and run f(start, end).
    pub fn chunks(&self, len: usize, min_chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let parts = (self.threads() * 4).min(len.div_ceil(min_chunk.max(1))).max(1);
        let size = len.div_ceil(parts);
        self.run(parts, &|i| {
            let a = i * size;
            let b = ((i + 1) * size).min(len);
            if a < b {
                f(a, b)
            }
        });
    }
}

fn work_on(job: &Job) {
    // SAFETY: see Pool::run.
    let task = unsafe { &*job.task };
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.n {
            break;
        }
        task(i);
    }
}

fn worker(s: &Shared) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            if s.quit.load(Ordering::Acquire) {
                return;
            }
            if s.epoch.load(Ordering::Acquire) != seen {
                break;
            }
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
            } else {
                let mut sleeping = s.sleeping.lock().unwrap();
                *sleeping += 1;
                while s.epoch.load(Ordering::Acquire) == seen && !s.quit.load(Ordering::Acquire) {
                    sleeping = s.wake.wait(sleeping).unwrap();
                }
                *sleeping -= 1;
                spins = 0;
            }
        }
        seen = s.epoch.load(Ordering::Acquire);
        let job = *s.job.lock().unwrap();
        if let Some(JobPtr(job)) = job {
            // SAFETY: run() keeps the job alive until `finished` counts us.
            work_on(unsafe { &*job });
        }
        s.finished.fetch_add(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Release);
        {
            let _g = self.shared.sleeping.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A raw pointer that may be shared between threads writing disjoint ranges.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// # Safety
    /// The caller guarantees no two threads touch overlapping ranges.
    pub unsafe fn slice<'a>(self, start: usize, len: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_index_once() {
        let pool = Pool::new(4);
        for n in [1, 3, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
        // Many tiny jobs in a row, as in decoding.
        let total = AtomicUsize::new(0);
        for _ in 0..2000 {
            pool.run(8, &|_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 16000);
    }
}
