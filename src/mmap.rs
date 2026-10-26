//! Read-only memory maps and the byte buffers that borrow from them.
//!
//! Weights stay in the page cache and are paged in on first use, so even a
//! multi-gigabyte model "loads" in milliseconds and is shared between
//! processes. On platforms without mmap the file is read into memory.

use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

pub struct Mmap {
    ptr: *const u8,
    len: usize,
    #[cfg(not(unix))]
    owned: Vec<u8>,
}

// The mapping is read-only and never moves.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    unsafe extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
}

impl Mmap {
    #[cfg(unix)]
    pub fn open(path: &Path) -> std::io::Result<Mmap> {
        use std::os::fd::AsRawFd;
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mmap { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0 });
        }
        // SAFETY: a fresh private read-only mapping of an open file.
        let ptr =
            unsafe { sys::mmap(std::ptr::null_mut(), len, sys::PROT_READ, sys::MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    #[cfg(not(unix))]
    pub fn open(path: &Path) -> std::io::Result<Mmap> {
        let owned = std::fs::read(path)?;
        Ok(Mmap { ptr: owned.as_ptr(), len: owned.len(), owned })
    }
}

impl Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: ptr/len describe a live mapping for as long as self exists.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.len > 0 {
            // SAFETY: unmapping exactly what open() mapped.
            unsafe { sys::munmap(self.ptr as *mut _, self.len) };
        }
    }
}

/// A cheaply clonable byte slice that keeps its backing storage alive:
/// either part of a memory map or an owned buffer.
#[derive(Clone)]
pub struct Bytes {
    owner: Arc<dyn Deref<Target = [u8]> + Send + Sync>,
    start: usize,
    len: usize,
}

/// Owned storage with 8-byte alignment (a plain Vec<u8> only promises 1).
struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Deref for Aligned {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: words holds at least len initialized bytes.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }
}

impl Bytes {
    /// Copy bytes into a new 8-byte aligned buffer.
    pub fn copy_from(data: &[u8]) -> Bytes {
        let mut words = vec![0u64; data.len().div_ceil(8)];
        // SAFETY: words has room for data.len() bytes.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr() as *mut u8, data.len()) };
        let len = data.len();
        Bytes { owner: Arc::new(Aligned { words, len }), start: 0, len }
    }

    pub fn slice_of(map: &Arc<Mmap>, start: usize, len: usize) -> Bytes {
        assert!(start + len <= map.len(), "slice out of bounds");
        Bytes { owner: map.clone(), start, len }
    }

    /// View as elements of T, copying into an aligned buffer if needed.
    pub fn aligned_to<T: Copy>(self) -> Bytes {
        if (self.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return self;
        }
        Bytes::copy_from(&self)
    }

    pub fn as_f32(&self) -> &[f32] {
        cast(self)
    }

    pub fn as_u16(&self) -> &[u16] {
        cast(self)
    }
}

fn cast<T: Copy>(b: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(b.len() % size, 0, "buffer length is not a multiple of the element size");
    assert_eq!(b.as_ptr() as usize % std::mem::align_of::<T>(), 0, "misaligned buffer");
    // SAFETY: length and alignment checked; T is plain old data.
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, b.len() / size) }
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.owner[self.start..self.start + self.len]
    }
}
