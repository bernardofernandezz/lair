//! What this machine has to run models on: CPU, memory, GPUs and the backend lair was built for.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

#[derive(Clone, Default, Serialize)]
pub struct Gpu {
    pub name: String,
    pub total: u64,
    /// As the backend reports it; some (Vulkan without the memory-budget extension) report the total.
    pub free: u64,
}

#[derive(Clone, Default, Serialize)]
pub struct Hardware {
    pub cpu: String,
    /// Logical cores.
    pub cores: usize,
    pub ram: u64,
    /// 0 when unknown (macOS).
    pub ram_free: u64,
    pub gpus: Vec<Gpu>,
    /// GPU memory, summed; llama.cpp splits layers across GPUs.
    pub vram: u64,
    pub vram_free: u64,
    /// `cuda`, `vulkan`, `metal` or `cpu`: what generation actually runs on.
    pub backend: &'static str,
    pub wsl: bool,
}

/// Free memory changes as models load; everything else is cheap to read again with it.
const FRESH: Duration = Duration::from_secs(2);

pub fn hardware() -> Hardware {
    static LAST: Mutex<Option<(Instant, Hardware)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap();
    match &*last {
        Some((at, hw)) if at.elapsed() < FRESH => hw.clone(),
        _ => {
            let hw = read();
            *last = Some((Instant::now(), hw.clone()));
            hw
        }
    }
}

/// The hardware read right now, for decisions that free memory changes (loading a model).
pub fn now() -> Hardware {
    read()
}

fn read() -> Hardware {
    use llama_cpp_2::LlamaBackendDeviceType as T;
    let gpus: Vec<Gpu> = llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| matches!(d.device_type, T::Gpu | T::IntegratedGpu | T::Accelerator))
        .map(|d| Gpu {
            name: if d.description.is_empty() { d.name } else { d.description },
            total: d.memory_total as u64,
            free: d.memory_free as u64,
        })
        .collect();
    let (ram, ram_free) = ram();
    Hardware {
        cpu: cpu(),
        cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        ram,
        ram_free,
        vram: gpus.iter().map(|g| g.total).sum(),
        vram_free: gpus.iter().map(|g| g.free).sum(),
        backend: if gpus.is_empty() { "cpu" } else { BUILT_FOR },
        gpus,
        wsl: wsl(),
    }
}

const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else {
    "cpu"
};

/// A `/proc/meminfo` value in bytes.
#[cfg(any(target_os = "linux", test))]
fn meminfo(text: &str, key: &str) -> u64 {
    text.lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(':')?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// The first `model name` of `/proc/cpuinfo`.
#[cfg(any(target_os = "linux", test))]
fn cpu_name(text: &str) -> String {
    text.lines()
        .find_map(|l| l.strip_prefix("model name")?.trim_start().strip_prefix(':'))
        .map_or(String::new(), |n| n.trim().to_owned())
}

#[cfg(target_os = "linux")]
fn ram() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    (meminfo(&text, "MemTotal"), meminfo(&text, "MemAvailable"))
}

#[cfg(target_os = "linux")]
fn cpu() -> String {
    cpu_name(&std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default())
}

#[cfg(target_os = "linux")]
fn wsl() -> bool {
    // The same test install.sh uses.
    std::fs::read_to_string("/proc/sys/kernel/osrelease").is_ok_and(|s| s.to_ascii_lowercase().contains("microsoft"))
}

#[cfg(not(target_os = "linux"))]
fn wsl() -> bool {
    false
}

#[cfg(windows)]
fn ram() -> (u64, u64) {
    /// Win32's `MEMORYSTATUSEX`; only the physical total and available are read.
    #[repr(C)]
    struct MemoryStatus {
        length: u32,
        _load: u32,
        total_phys: u64,
        avail_phys: u64,
        _rest: [u64; 5],
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
    }
    let length = size_of::<MemoryStatus>() as u32;
    let mut status = MemoryStatus { length, _load: 0, total_phys: 0, avail_phys: 0, _rest: [0; 5] };
    // SAFETY: `status` is a `MEMORYSTATUSEX` with its length set, which is all the call asks for.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { (status.total_phys, status.avail_phys) } else { (0, 0) }
}

// ponytail: the env var names the CPU family, not its marketing name; read the registry's
// ProcessorNameString if people want "Ryzen 7 7800X3D" here.
#[cfg(windows)]
fn cpu() -> String {
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default()
}

/// Reads a sysctl value into `buf`, returning how many bytes it wrote (0 on failure).
#[cfg(target_os = "macos")]
fn sysctl(name: &std::ffi::CStr, buf: &mut [u8]) -> usize {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut len = buf.len();
    // SAFETY: `name` is NUL-terminated and `len` is `buf`'s size, which the call never writes past.
    let ok = unsafe { sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } == 0;
    if ok { len.min(buf.len()) } else { 0 }
}

// ponytail: free RAM is unknown on macOS; Metal reports unified memory as the GPU's, which is what fit uses.
#[cfg(target_os = "macos")]
fn ram() -> (u64, u64) {
    let mut total = [0u8; 8];
    let n = sysctl(c"hw.memsize", &mut total);
    (if n == 8 { u64::from_ne_bytes(total) } else { 0 }, 0)
}

#[cfg(target_os = "macos")]
fn cpu() -> String {
    let mut buf = [0u8; 128];
    let n = sysctl(c"machdep.cpu.brand_string", &mut buf);
    String::from_utf8_lossy(&buf[..n]).trim_end_matches('\0').trim().to_owned()
}

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
fn ram() -> (u64, u64) {
    (0, 0)
}

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
fn cpu() -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_files_are_parsed() {
        let mem = "MemTotal:       16314160 kB\nMemFree:          210000 kB\nMemAvailable:    8000000 kB\n";
        assert_eq!(meminfo(mem, "MemTotal"), 16314160 * 1024);
        assert_eq!(meminfo(mem, "MemAvailable"), 8000000 * 1024);
        assert_eq!(meminfo(mem, "Mem"), 0, "a key is matched whole");
        assert_eq!(meminfo("", "MemTotal"), 0);
        let cpu = "processor\t: 0\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\nmodel name\t: other\n";
        assert_eq!(cpu_name(cpu), "AMD Ryzen 7 5800X 8-Core Processor");
        assert_eq!(cpu_name("processor: 0\n"), "");
    }

    #[test]
    fn this_machine_is_read() {
        let hw = hardware();
        assert!(hw.cores >= 1);
        assert!(hw.ram > 0 && hw.ram_free <= hw.ram, "{} / {}", hw.ram_free, hw.ram);
    }
}
