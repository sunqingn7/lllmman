use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::time::SystemTime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub n_layer: Option<u32>,
    pub n_ctx_train: Option<u32>,
    pub n_embd: Option<u32>,
    pub n_head: Option<u32>,
    pub n_kv_head: Option<u32>,
    pub n_vocab: Option<u32>,
    pub file_size_bytes: u64,
    pub file_modified_secs: u64,
    pub extracted_at_secs: u64,
}

static CACHE_DIR: &str = "model_metadata";

fn get_cache_path() -> std::path::PathBuf {
    let dir = if let Some(config) = dirs::config_dir() {
        config.join("lllmman").join(CACHE_DIR)
    } else {
        std::env::temp_dir().join("lllmman").join(CACHE_DIR)
    };
    fs::create_dir_all(&dir).ok();
    dir.join("cache.json")
}

fn load_cache() -> HashMap<String, ModelMetadata> {
    let path = get_cache_path();
    if path.exists() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(cache) = serde_json::from_str(&content) {
                return cache;
            }
        }
    }
    HashMap::new()
}

fn save_cache(cache: &HashMap<String, ModelMetadata>) {
    let path = get_cache_path();
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        fs::write(&path, json).ok();
    }
}

pub fn get_model_metadata(path: &str) -> Option<ModelMetadata> {
    let meta = fs::metadata(path).ok()?;
    let file_size = meta.len();
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let cache = load_cache();
    let key = path.to_string();

    if let Some(entry) = cache.get(&key) {
        if entry.file_size_bytes == file_size && entry.file_modified_secs == modified {
            return Some(entry.clone());
        }
    }

    let metadata = extract_gguf_metadata(path)?;

    let mut cache = load_cache();
    cache.insert(key, metadata.clone());
    save_cache(&cache);

    Some(metadata)
}

fn extract_gguf_metadata(path: &str) -> Option<ModelMetadata> {
    let meta = fs::metadata(path).ok()?;
    let file_size = meta.len();
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let file = fs::File::open(path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };

    if mmap.len() < 24 || &mmap[0..4] != b"GGUF" {
        return None;
    }

    let metadata_count = u64::from_le_bytes([
        mmap[16], mmap[17], mmap[18], mmap[19], mmap[20], mmap[21], mmap[22], mmap[23],
    ]) as usize;

    if metadata_count > 10000 {
        return None;
    }

    let mut offset = 24usize;

    let mut n_layer = None;
    let mut n_ctx_train = None;
    let mut n_embd = None;
    let mut n_head = None;
    let mut n_kv_head = None;
    let mut n_vocab = None;

    for _ in 0..metadata_count {
        if offset + 12 > mmap.len() {
            break;
        }

        let key_len = u64::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
            mmap[offset + 4],
            mmap[offset + 5],
            mmap[offset + 6],
            mmap[offset + 7],
        ]) as usize;

        if key_len > 200 || offset + 12 + key_len > mmap.len() {
            break;
        }

        let key = match String::from_utf8(mmap[offset + 8..offset + 8 + key_len].to_vec()) {
            Ok(k) => k,
            Err(_) => {
                offset += 8 + key_len + 4;
                continue;
            }
        };
        offset += 8 + key_len;

        if offset + 4 > mmap.len() {
            break;
        }
        let val_type = u32::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
        ]);
        offset += 4;

        if val_type == 4 && offset + 4 <= mmap.len() {
            let val = u32::from_le_bytes([
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ]);

            if key.ends_with(".block_count") {
                n_layer = Some(val);
            } else if key.ends_with(".context_length") {
                n_ctx_train = Some(val);
            } else if key.ends_with(".embedding_length") {
                n_embd = Some(val);
            } else if key.ends_with(".attention.head_count") {
                n_head = Some(val);
            } else if key.ends_with(".attention.head_count_kv") {
                n_kv_head = Some(val);
            } else if key.ends_with(".vocab_size") {
                n_vocab = Some(val);
            }
        }

        skip_value(&mmap, val_type as u32, &mut offset);
    }

    Some(ModelMetadata {
        n_layer,
        n_ctx_train,
        n_embd,
        n_head,
        n_kv_head,
        n_vocab,
        file_size_bytes: file_size,
        file_modified_secs: modified,
        extracted_at_secs: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs())
            .unwrap_or(0),
    })
}

fn skip_value(data: &[u8], val_type: u32, offset: &mut usize) {
    match val_type {
        0 | 1 => {
            *offset += 1;
        }
        2 | 3 => {
            *offset += 2;
        }
        4 | 5 | 6 => {
            *offset += 4;
        }
        7 => {
            *offset += 8;
        }
        8 => {
            if *offset + 8 <= data.len() {
                let str_len = u64::from_le_bytes([
                    data[*offset],
                    data[*offset + 1],
                    data[*offset + 2],
                    data[*offset + 3],
                    data[*offset + 4],
                    data[*offset + 5],
                    data[*offset + 6],
                    data[*offset + 7],
                ]) as usize;
                *offset += 8 + str_len.min(10000);
            }
        }
        9 => {
            if *offset + 12 <= data.len() {
                let arr_type = u32::from_le_bytes([
                    data[*offset],
                    data[*offset + 1],
                    data[*offset + 2],
                    data[*offset + 3],
                ]);
                *offset += 4;
                let arr_count = u64::from_le_bytes([
                    data[*offset],
                    data[*offset + 1],
                    data[*offset + 2],
                    data[*offset + 3],
                    data[*offset + 4],
                    data[*offset + 5],
                    data[*offset + 6],
                    data[*offset + 7],
                ]) as usize;
                *offset += 8;

                match arr_type {
                    0 | 1 => {
                        *offset += arr_count;
                    }
                    2 | 3 => {
                        *offset += arr_count * 2;
                    }
                    4 | 5 | 6 => {
                        *offset += arr_count * 4;
                    }
                    7 => {
                        *offset += arr_count * 8;
                    }
                    8 => {
                        for _ in 0..arr_count.min(1000) {
                            if *offset + 8 > data.len() {
                                break;
                            }
                            let el_len = u64::from_le_bytes([
                                data[*offset],
                                data[*offset + 1],
                                data[*offset + 2],
                                data[*offset + 3],
                                data[*offset + 4],
                                data[*offset + 5],
                                data[*offset + 6],
                                data[*offset + 7],
                            ]) as usize;
                            *offset += 8 + el_len.min(10000);
                        }
                    }
                    _ => {}
                }
            }
        }
        10 | 11 | 12 => {
            *offset += 8;
        }
        _ => {}
    }
}

pub fn clear_cache() {
    let path = get_cache_path();
    if path.exists() {
        fs::remove_file(&path).ok();
    }
}
