#![allow(dead_code)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::core::ModelInfo;

#[derive(Debug, Clone)]
pub struct SanityIssue {
    pub path: String,
    pub name: String,
    pub reason: String,
}

pub struct SanityReport {
    pub valid: Vec<ModelInfo>,
    pub broken: Vec<SanityIssue>,
    pub warnings: Vec<SanityIssue>,
    pub checked: usize,
}

// (block_size, block_bytes) per ggml tensor dtype; entries not listed here
// (IQ variants, T1_*, etc.) are excluded from the size check to avoid
// false positives.
const GGML_TYPE_SIZES: [(u32, u32, u32); 16] = [
    (0, 1, 4),   // F32
    (1, 1, 2),   // F16
    (2, 1, 2),   // BF16
    (3, 32, 16), // Q4_0
    (4, 32, 18), // Q4_1
    (5, 32, 22), // Q5_0
    (6, 32, 24), // Q5_1
    (7, 32, 34), // Q8_0
    (8, 32, 40), // Q8_1
    (9, 256, 84),  // Q2_K
    (10, 256, 110), // Q3_K
    (11, 256, 144), // Q4_K
    (12, 256, 176), // Q5_K
    (13, 256, 210), // Q6_K
    (16, 256, 84),  // Q2_K_S
    (20, 32, 18),   // IQ4_NL
];

struct TrackedReader {
    file: File,
    pos: u64,
}

impl TrackedReader {
    fn new(file: File) -> Self {
        Self { file, pos: 0 }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.file
            .read_exact(buf)
            .map_err(|_| format!("unexpected EOF at offset {}", self.pos))?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u32le(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64le(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn skip(&mut self, n: u64) -> Result<(), String> {
        self.file
            .seek(SeekFrom::Current(n as i64))
            .map_err(|e| format!("seek failed at {}: {}", self.pos, e))?;
        self.pos += n;
        Ok(())
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64le()?;
        if len > 1024 * 1024 {
            return Err(format!("corrupt header: implausible string length {}", len));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

fn skip_metadata_value(
    r: &mut TrackedReader,
    val_type: u8,
    depth: u32,
    type_size: usize,
) -> Result<(), String> {
    if depth > 8 {
        return Err("corrupt header: array nesting too deep".into());
    }
    match val_type {
        0 | 1 | 7 => r.skip(1),                    // u8, i8, bool
        2 | 3 => r.skip(2),                        // u16, i16
        4..=6 => r.skip(4),                       // u32, i32, f32
        10..=12 => r.skip(8),                    // u64, i64, f64
        8 => {
            let len = r.u64le()?;
            if len > 1024 * 1024 {
                return Err(format!("corrupt header: implausible string length {}", len));
            }
            r.skip(len)
        }
        9 => {
            let elem_type = if type_size == 4 {
                r.u32le()? as u8
            } else {
                r.u8()?
            };
            let count = r.u64le()?;
            if count > 50_000_000 {
                return Err(format!("corrupt header: implausible array length {}", count));
            }
            let elem_size = match elem_type {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            };
            if let Some(size) = elem_size {
                r.skip(count * size)?;
            } else {
                for _ in 0..count {
                    skip_metadata_value(r, elem_type, depth + 1, type_size)?;
                }
            }
            Ok(())
        }
        t => Err(format!("corrupt header: unknown metadata type {}", t)),
    }
}

fn ggml_tensor_size(dtype: u32, elem_count: u64) -> Option<u64> {
    for (t, block_size, block_bytes) in GGML_TYPE_SIZES.iter() {
        if *t == dtype {
            if *block_size == 1 {
                return Some(elem_count * *block_bytes as u64);
            }
            if !elem_count.is_multiple_of(*block_size as u64) {
                return None;
            }
            return Some((elem_count / *block_size as u64) * *block_bytes as u64);
        }
    }
    None
}

fn parse_gguf_header(path: &Path, type_size: usize) -> Result<(), String> {
    let file_size = p_size(path)?;
    let mut r = TrackedReader::new(File::open(path).map_err(|e| format!("cannot open: {}", e))?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err("missing GGUF magic bytes (not a GGUF file?)".to_string());
    }
    let version = r.u32le()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {}", version));
    }
    let n_tensors = r.u64le()?;
    let n_kv = r.u64le()?;
    if n_tensors > 200_000 || n_kv > 50_000 {
        return Err("corrupt header: implausible tensor/KV counts".to_string());
    }

    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = r.string()?;
        let val_type = if type_size == 4 {
            r.u32le()? as u8
        } else {
            r.u8()?
        };
        if key == "general.alignment" {
            if val_type == 4 {
                alignment = r.u32le()? as u64;
                if alignment == 0 || alignment > 1024 * 1024 {
                    alignment = 32;
                }
            } else {
                return Err("corrupt header: general.alignment has wrong type".into());
            }
        } else {
            skip_metadata_value(&mut r, val_type, 0, type_size)?;
        }
    }

    let mut max_end: u64 = 0;
    let mut known_tensors = 0u64;
    for _ in 0..n_tensors {
        let _name = r.string()?;
        let n_dims = r.u32le()?;
        if n_dims == 0 || n_dims > 64 {
            return Err("corrupt header: implausible tensor rank".to_string());
        }
        let mut elem_count: u64 = 1;
        for _ in 0..n_dims {
            let dim = r.u64le()?;
            elem_count = elem_count.saturating_mul(dim);
        }
        let dtype = r.u32le()?;
        let offset = r.u64le()?;
        if offset > file_size {
            return Err(format!(
                "corrupt header: tensor offset {} beyond file size {}",
                offset, file_size
            ));
        }
        if let Some(size) = ggml_tensor_size(dtype, elem_count) {
            max_end = max_end.max(offset + size);
            known_tensors += 1;
        }
    }

    let infos_end = r.pos;
    let data_start = infos_end + ((alignment - infos_end % alignment) % alignment);

    if known_tensors > 0 && file_size < data_start + max_end {
        return Err(format!(
            "incomplete file: {} bytes on disk, but model data requires at least {}",
            file_size,
            data_start + max_end
        ));
    }
    Ok(())
}

fn p_size(path: &Path) -> Result<u64, String> {
    if !path.exists() {
        return Err("model file missing".to_string());
    }
    if !path.is_file() {
        return Err("model path is not a regular file".to_string());
    }
    let size = path
        .metadata()
        .map_err(|e| format!("cannot read file metadata: {}", e))?
        .len();
    if size == 0 {
        return Err("model file is empty".to_string());
    }
    Ok(size)
}

/// Validate a single GGUF file: magic bytes, header structure, and (when all
/// tensor dtypes are known) whether the file is large enough to contain the
/// full tensor data, which detects truncated / incomplete downloads.
///
/// Some community files store metadata value types as u32 instead of the
/// spec's u8; both layouts are tried before reporting a file as broken.
pub fn check_gguf_file(path: &str) -> Result<(), String> {
    p_size(Path::new(path))?;
    let p = Path::new(path);
    match parse_gguf_header(p, 1) {
        Ok(()) => Ok(()),
        Err(std_err) => match parse_gguf_header(p, 4) {
            Ok(()) => Ok(()),
            Err(alt_err) => {
                if alt_err.starts_with("incomplete") {
                    Err(alt_err)
                } else {
                    Err(std_err)
                }
            }
        },
    }
}

fn extract_part_pattern(filename: &str) -> Option<(usize, usize, String, String)> {
    // "-00001-of-00004" inside the stem -> (index, total, prefix, suffix)
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    let of_pos = stem.rfind("-of-")?;
    let total_str = &stem[of_pos + 4..];
    if total_str.len() != 5 || !total_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let idx_start = of_pos.checked_sub(5)?;
    let index_str = &stem[idx_start + 1..of_pos];
    if index_str.len() != 5 || !index_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((
        index_str.parse().unwrap_or(0),
        total_str.parse().unwrap_or(0),
        stem[..idx_start + 1].to_string(),
        ".gguf".to_string(),
    ))
}

fn check_multipart_parts(path: &str, index: usize, total: usize, prefix: &str) -> Vec<String> {
    let mut missing = Vec::new();
    if index != 1 {
        return missing;
    }
    let p = Path::new(path);
    let dir = match p.parent() {
        Some(d) => d,
        None => return missing,
    };
    for i in 2..=total {
        let part = dir.join(format!("{}{:05}-of-{:05}.gguf", prefix, i, total));
        if !part.exists() {
            missing.push(part.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        }
    }
    missing
}

fn check_non_gguf_model(model: &ModelInfo) -> Result<(), String> {
    let path = &model.path;
    if Path::new(path).exists() {
        return Ok(());
    }
    if !path.starts_with('/') && !path.starts_with('.') {
        // HuggingFace repo ID (with or without an org namespace, e.g.
        // "Qwen/Qwen3-4B" or "gpt2"): verify its cache directory exists
        let found = crate::providers::sglang::find_huggingface_model_path(path)
            .or_else(|| crate::providers::vllm::find_huggingface_model_path(path));
        if found.is_some() {
            return Ok(());
        }
        let hub_missing = dirs::cache_dir()
            .map(|c| {
                c.join("huggingface")
                    .join("hub")
                    .join(format!("models--{}", path.replace('/', "--")))
            })
            .map(|d| !d.exists())
            .unwrap_or(true);
        if hub_missing {
            return Err(format!(
                "model '{}' not found in local HuggingFace cache (not downloaded?)",
                path
            ));
        }
        return Ok(());
    }
    Err(format!("model file missing: {}", path))
}

pub fn check_models(models: Vec<ModelInfo>) -> SanityReport {
    let mut report = SanityReport {
        valid: Vec::new(),
        broken: Vec::new(),
        warnings: Vec::new(),
        checked: models.len(),
    };

    for model in models {
        let is_gguf = model.path.to_lowercase().ends_with(".gguf");
        let issue = if is_gguf {
            let filename = Path::new(&model.path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            if let Some((index, total, prefix, _)) = extract_part_pattern(&filename) {
                // Split GGUF: metadata lives in part 1; other parts are raw data
                if index != 1 {
                    if Path::new(&model.path).exists() {
                        Ok(())
                    } else {
                        Err("model file part missing".to_string())
                    }
                } else {
                    check_gguf_file(&model.path).and_then(|_| {
                        let missing = check_multipart_parts(&model.path, index, total, &prefix);
                        if missing.is_empty() {
                            Ok(())
                        } else {
                            Err(format!("split model is missing parts: {}", missing.join(", ")))
                        }
                    })
                }
            } else {
                check_gguf_file(&model.path)
            }
        } else {
            check_non_gguf_model(&model)
        };

        match issue {
            Ok(()) => report.valid.push(model),
            Err(reason) => report.broken.push(SanityIssue {
                path: model.path.clone(),
                name: model.name.clone(),
                reason,
            }),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_gguf(type_size: usize, tensors_extra_bytes_removed: u64) -> Vec<u8> {
        fn s(x: &str) -> Vec<u8> {
            let mut v = (x.len() as u64).to_le_bytes().to_vec();
            v.extend_from_slice(x.as_bytes());
            v
        }
        fn u64le(x: u64) -> Vec<u8> {
            x.to_le_bytes().to_vec()
        }
        fn u32le(x: u32) -> Vec<u8> {
            x.to_le_bytes().to_vec()
        }

        let mut kv = Vec::new();
        kv.extend(s("general.architecture"));
        if type_size == 4 {
            kv.extend(u32le(8));
        } else {
            kv.push(8);
        }
        kv.extend(s("llama"));
        let mut head = b"GGUF".to_vec();
        head.extend(u32le(3));
        head.extend(u64le(1)); // n_tensors
        head.extend(u64le(1)); // n_kv
        head.extend(kv);
        // tensor "w": f32, dims [2,3] => 24 bytes, offset 0
        head.extend(s("w"));
        head.extend(u32le(2));
        head.extend(u64le(2));
        head.extend(u64le(3));
        head.extend(u32le(0));
        head.extend(u64le(0));
        let pad = (32 - (head.len() as u64 % 32)) % 32;
        head.extend(vec![0u8; pad as usize]);
        let mut full = head.clone();
        full.extend(vec![65u8; (24 - tensors_extra_bytes_removed) as usize]);
        full
    }

    fn write_tmp(name: &str, bytes: &[u8]) -> String {
        let dir = std::env::temp_dir().join("lllmman_validator_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn valid_spec_gguf_passes() {
        let path = write_tmp("good_u8.gguf", &synth_gguf(1, 0));
        assert!(check_gguf_file(&path).is_ok());
    }

    #[test]
    fn valid_u32_type_layout_gguf_passes() {
        let path = write_tmp("good_u32.gguf", &synth_gguf(4, 0));
        assert!(check_gguf_file(&path).is_ok());
    }

    #[test]
    fn truncated_gguf_detected() {
        let path = write_tmp("trunc.gguf", &synth_gguf(1, 14));
        let err = check_gguf_file(&path).unwrap_err();
        assert!(err.contains("incomplete"), "got: {}", err);
    }

    #[test]
    fn garbage_detected() {
        let path = write_tmp("garbage.gguf", b"NOTGGUF and some bytes");
        assert!(check_gguf_file(&path).unwrap_err().contains("magic"));
    }

    #[test]
    fn empty_detected() {
        let path = write_tmp("empty.gguf", b"");
        assert!(check_gguf_file(&path).unwrap_err().contains("empty"));
    }

    #[test]
    fn missing_file_detected() {
        let err = check_gguf_file("/nonexistent/path/model.gguf").unwrap_err();
        assert!(err.contains("missing"));
    }
}
