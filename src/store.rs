//! Trace store: every routed request is logged to an embedded SQLite database,
//! off the hot path. The miner reads it overnight; self-healing writes to it.
//!
//! Writes go through a bounded channel to one background thread that batches
//! them into transactions. The request path never waits on disk: when the
//! channel is full, the trace is dropped and counted.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::time::Duration;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

/// One routed request, as recorded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub ts_ms: i64,
    /// chat | responses | stream
    pub surface: String,
    /// Hash of tenant key + prompt + tool set: never the raw key.
    pub scope: i64,
    /// Hash of the user id: never the raw id.
    pub user_hash: Option<String>,
    /// The question, unless `store.redact_questions` is set.
    pub question: Option<String>,
    /// Abstracted shape of the question ("calls for <code> in <date>").
    pub shape: String,
    /// What answered: a model name, learned-route, speculative-template, planner:<op>, rule:<name>, cache:<kind>.
    pub served_by: String,
    pub intent: String,
    pub tier: String,
    pub phase: String,
    pub tool: Option<String>,
    pub tool_args: Option<String>,
    pub call_id: Option<String>,
    /// For a result hop: which call the result answered and whether it was an error.
    pub result_for: Option<String>,
    pub result_error: Option<bool>,
    pub outcome: String,
    pub latency_ms: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cost_usd: f64,
    pub confidence: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealEvent {
    pub ts_ms: i64,
    pub kind: String,
    pub subject: String,
    pub detail: String,
}

enum Msg {
    Trace(Box<Trace>),
    Heal(HealEvent),
    Kv(String, Vec<u8>),
    Flush(std::sync::mpsc::Sender<()>),
}

pub struct Store {
    path: PathBuf,
    tx: SyncSender<Msg>,
    pub dropped: AtomicU64,
    pub written: std::sync::Arc<AtomicU64>,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

const SCHEMA: &str = "
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, ts_ms INTEGER NOT NULL, surface TEXT, scope INTEGER, user_hash TEXT,
  question TEXT, shape TEXT, served_by TEXT, intent TEXT, tier TEXT, phase TEXT,
  tool TEXT, tool_args TEXT, call_id TEXT, result_for TEXT, result_error INTEGER,
  outcome TEXT, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, confidence REAL, error TEXT);
CREATE INDEX IF NOT EXISTS traces_ts ON traces(ts_ms);
CREATE INDEX IF NOT EXISTS traces_shape ON traces(shape);
CREATE INDEX IF NOT EXISTS traces_call ON traces(call_id);
CREATE INDEX IF NOT EXISTS traces_result_for ON traces(result_for);
CREATE TABLE IF NOT EXISTS heal_log (ts_ms INTEGER, kind TEXT, subject TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB, updated_ms INTEGER);
";

impl Store {
    /// Open (or create) the database and start the writer thread.
    pub fn open(path: impl AsRef<Path>, queue: usize) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch(SCHEMA)?;
        let (tx, rx) = sync_channel::<Msg>(queue.max(64));
        let written = std::sync::Arc::new(AtomicU64::new(0));
        let w = written.clone();
        std::thread::Builder::new().name("pankh-store".into()).spawn(move || writer(conn, rx, w))?;
        Ok(Self { path, tx, dropped: AtomicU64::new(0), written })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Never blocks. A full queue drops the trace and counts it.
    pub fn record(&self, t: Trace) {
        if let Err(TrySendError::Full(_)) = self.tx.try_send(Msg::Trace(Box::new(t))) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn heal(&self, kind: &str, subject: &str, detail: &str) {
        let _ = self.tx.try_send(Msg::Heal(HealEvent { ts_ms: now_ms(), kind: kind.into(), subject: subject.into(), detail: detail.into() }));
    }

    pub fn put_kv(&self, k: &str, v: Vec<u8>) {
        let _ = self.tx.send(Msg::Kv(k.to_string(), v));
    }

    /// Wait until everything queued so far is on disk.
    pub fn flush(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        if self.tx.send(Msg::Flush(tx)).is_ok() {
            let _ = rx.recv_timeout(Duration::from_secs(10));
        }
    }

    /// A read-only connection for queries (miner, API). Separate from the writer.
    pub fn reader(&self) -> rusqlite::Result<Connection> {
        let c = Connection::open_with_flags(&self.path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
        c.busy_timeout(Duration::from_secs(5))?;
        Ok(c)
    }

    pub fn get_kv(&self, k: &str) -> Option<Vec<u8>> {
        let c = self.reader().ok()?;
        c.query_row("SELECT v FROM kv WHERE k = ?1", params![k], |r| r.get::<_, Vec<u8>>(0)).ok()
    }
}

fn writer(mut conn: Connection, rx: Receiver<Msg>, written: std::sync::Arc<AtomicU64>) {
    loop {
        // Block for the first message, then drain whatever else is waiting into one transaction.
        let first = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let mut batch = vec![first];
        while batch.len() < 1000 {
            match rx.try_recv() {
                Ok(m) => batch.push(m),
                Err(_) => break,
            }
        }
        let mut flushes = Vec::new();
        let n = batch.iter().filter(|m| matches!(m, Msg::Trace(_))).count() as u64;
        if let Ok(tx) = conn.transaction() {
            for m in batch {
                match m {
                    Msg::Trace(t) => {
                        let _ = tx.execute(
                            "INSERT OR REPLACE INTO traces (id, ts_ms, surface, scope, user_hash, question, shape, served_by, intent, tier, phase, tool, tool_args, call_id, result_for, result_error, outcome, latency_ms, input_tokens, output_tokens, cost_usd, confidence, error)
                             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23)",
                            params![t.id, t.ts_ms, t.surface, t.scope, t.user_hash, t.question, t.shape, t.served_by, t.intent, t.tier, t.phase, t.tool, t.tool_args, t.call_id, t.result_for, t.result_error.map(|b| b as i64), t.outcome, t.latency_ms, t.input_tokens, t.output_tokens, t.cost_usd, t.confidence, t.error],
                        );
                    }
                    Msg::Heal(h) => {
                        let _ = tx.execute("INSERT INTO heal_log (ts_ms, kind, subject, detail) VALUES (?1,?2,?3,?4)", params![h.ts_ms, h.kind, h.subject, h.detail]);
                    }
                    Msg::Kv(k, v) => {
                        let _ = tx.execute("INSERT OR REPLACE INTO kv (k, v, updated_ms) VALUES (?1,?2,?3)", params![k, v, now_ms()]);
                    }
                    Msg::Flush(done) => flushes.push(done),
                }
            }
            let _ = tx.commit();
            written.fetch_add(n, Ordering::Relaxed);
        }
        for f in flushes {
            let _ = f.send(());
        }
    }
}

/// Read traces newer than `since_ms`, oldest first.
pub fn read_traces(conn: &Connection, since_ms: i64, limit: usize) -> rusqlite::Result<Vec<Trace>> {
    let mut st = conn.prepare(
        "SELECT id, ts_ms, surface, scope, user_hash, question, shape, served_by, intent, tier, phase, tool, tool_args, call_id, result_for, result_error, outcome, latency_ms, input_tokens, output_tokens, cost_usd, confidence, error
         FROM traces WHERE ts_ms >= ?1 ORDER BY ts_ms ASC LIMIT ?2",
    )?;
    let rows = st.query_map(params![since_ms, limit as i64], |r| {
        Ok(Trace {
            id: r.get(0)?,
            ts_ms: r.get(1)?,
            surface: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
            scope: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
            user_hash: r.get(4)?,
            question: r.get(5)?,
            shape: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
            served_by: r.get::<_, Option<String>>(7)?.unwrap_or_default(),
            intent: r.get::<_, Option<String>>(8)?.unwrap_or_default(),
            tier: r.get::<_, Option<String>>(9)?.unwrap_or_default(),
            phase: r.get::<_, Option<String>>(10)?.unwrap_or_default(),
            tool: r.get(11)?,
            tool_args: r.get(12)?,
            call_id: r.get(13)?,
            result_for: r.get(14)?,
            result_error: r.get::<_, Option<i64>>(15)?.map(|v| v != 0),
            outcome: r.get::<_, Option<String>>(16)?.unwrap_or_default(),
            latency_ms: r.get::<_, Option<i64>>(17)?.unwrap_or(0),
            input_tokens: r.get::<_, Option<i64>>(18)?.unwrap_or(0),
            output_tokens: r.get::<_, Option<i64>>(19)?.unwrap_or(0),
            cost_usd: r.get::<_, Option<f64>>(20)?.unwrap_or(0.0),
            confidence: r.get::<_, Option<f64>>(21)?.unwrap_or(0.0),
            error: r.get(22)?,
        })
    })?;
    rows.collect()
}

pub fn read_heal(conn: &Connection, since_ms: i64) -> rusqlite::Result<Vec<HealEvent>> {
    let mut st = conn.prepare("SELECT ts_ms, kind, subject, detail FROM heal_log WHERE ts_ms >= ?1 ORDER BY ts_ms ASC")?;
    let rows = st.query_map(params![since_ms], |r| Ok(HealEvent { ts_ms: r.get(0)?, kind: r.get(1)?, subject: r.get(2)?, detail: r.get(3)? }))?;
    rows.collect()
}

/// Delete traces older than `retention_days`.
pub fn prune(conn: &Connection, retention_days: u64) -> rusqlite::Result<usize> {
    let cutoff = now_ms() - (retention_days as i64) * 86_400_000;
    conn.execute("DELETE FROM traces WHERE ts_ms < ?1", params![cutoff])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_batch_read_and_kv() {
        let p = std::env::temp_dir().join(format!("pankh-store-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let s = Store::open(&p, 1024).unwrap();
        for i in 0..250 {
            s.record(Trace { id: format!("t{i}"), ts_ms: 1000 + i, shape: "calls for <code>".into(), served_by: if i % 2 == 0 { "learned-route".into() } else { "mini".into() }, ..Default::default() });
        }
        s.heal("quarantine", "route:1", "2 error results");
        s.put_kv("tuning", b"{\"hedge_after_ms\":900}".to_vec());
        s.flush();
        let c = s.reader().unwrap();
        let t = read_traces(&c, 0, 10_000).unwrap();
        assert_eq!(t.len(), 250);
        assert_eq!(t[0].id, "t0");
        assert_eq!(read_heal(&c, 0).unwrap().len(), 1);
        assert_eq!(s.get_kv("tuning").unwrap(), b"{\"hedge_after_ms\":900}".to_vec());
        assert_eq!(s.written.load(Ordering::Relaxed), 250);
        drop(c);
        let _ = std::fs::remove_file(&p);
    }
}
