use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub timestamp: std::time::SystemTime,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}

const MAX_LOG_ENTRIES: usize = 5000;

#[derive(Clone)]
pub struct LogBuffer {
    entries: Arc<Mutex<Vec<LogEntry>>>,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogBuffer {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn push(&self, level: LogLevel, message: String) {
        let entry = LogEntry {
            timestamp: std::time::SystemTime::now(),
            level,
            message,
        };
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);
        if entries.len() > MAX_LOG_ENTRIES {
            entries.remove(0);
        }
    }

    pub fn push_info(&self, message: String) {
        self.push(LogLevel::Info, message);
    }

    pub fn push_warn(&self, message: String) {
        self.push(LogLevel::Warn, message);
    }

    pub fn push_error(&self, message: String) {
        self.push(LogLevel::Error, message);
    }

    pub fn get_entries(&self) -> Vec<LogEntry> {
        self.entries.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}
