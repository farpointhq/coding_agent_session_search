//! Indexing Performance Benchmarks
//!
//! This module benchmarks indexing performance, including streaming vs batch mode
//! comparisons added in Opt 8.4 (coding_agent_session_search-nkc9).
//!
//! ## Memory Profiling
//!
//! For memory profiling (Peak RSS, memory timeline), use external tools:
//!
//! ### Peak RSS Comparison
//! ```bash
//! # Batch mode
//! CASS_STREAMING_INDEX=0 /usr/bin/time -v cargo run --release -- index --full 2>&1 | grep "Maximum resident"
//!
//! # Streaming mode (default)
//! /usr/bin/time -v cargo run --release -- index --full 2>&1 | grep "Maximum resident"
//! ```
//!
//! ### Memory Timeline (heaptrack)
//! ```bash
//! # Install heaptrack: apt install heaptrack heaptrack-gui
//! CASS_STREAMING_INDEX=0 heaptrack cargo run --release -- index --full
//! heaptrack_gui heaptrack.*.zst
//!
//! CASS_STREAMING_INDEX=1 heaptrack cargo run --release -- index --full
//! heaptrack_gui heaptrack.*.zst
//! ```
//!
//! ### Memory Timeline (valgrind massif)
//! ```bash
//! CASS_STREAMING_INDEX=0 valgrind --tool=massif cargo run --release -- index --full
//! ms_print massif.out.* > batch_memory.txt
//!
//! CASS_STREAMING_INDEX=1 valgrind --tool=massif cargo run --release -- index --full
//! ms_print massif.out.* > streaming_memory.txt
//! ```
//!
//! ## Expected Results
//! - Peak RSS: 295 MB (batch) → ~150 MB (streaming), ~50% reduction
//! - Throughput: No more than 10% regression
//! - Memory timeline: Streaming should show flat profile vs batch's spike

use coding_agent_search::connectors::{ScanContext, ScanRoot, preflight_codex_explicit_file_roots};
use coding_agent_search::indexer::redact_secrets::redact_text;
use coding_agent_search::indexer::semantic::{
    EmbeddingInput, SemanticIndexer, SemanticShardBuildPlan,
};
use coding_agent_search::indexer::{IndexOptions, get_connector_factories, run_index};
use coding_agent_search::search::semantic_manifest::{SemanticShardManifest, TierKind};
use coding_agent_search::search::tantivy::index_dir;
use coding_agent_search::search::vector_index::{VectorIndex as FsVectorIndex, vector_index_path};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::fs;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

/// Create a test corpus with the specified number of conversations.
///
/// Each conversation has 2 messages (user + assistant).
fn create_corpus(tmp: &TempDir, count: usize) -> (std::path::PathBuf, std::path::PathBuf) {
    let data_dir = tmp.path().join("data");
    let db_path = data_dir.join("agent_search.db");

    // Create Codex-format sessions
    let codex_home = data_dir.clone();
    for i in 0..count {
        let date_path = format!("sessions/2024/11/{:02}", (i % 30) + 1);
        let sessions = codex_home.join(&date_path);
        fs::create_dir_all(&sessions).unwrap();

        let filename = format!("rollout-{i}.jsonl");
        let file = sessions.join(&filename);
        let ts = 1732118400000 + (i as u64 * 1000);
        let content = format!(
            r#"{{"type": "event_msg", "timestamp": {ts}, "payload": {{"type": "user_message", "message": "test message {i} with unique content"}}}}
{{"type": "response_item", "timestamp": {}, "payload": {{"role": "assistant", "content": "response to message {i}"}}}}
"#,
            ts + 1000
        );
        fs::write(file, content).unwrap();
    }

    (data_dir, db_path)
}

fn bench_index_full(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let db_path = data_dir.join("agent_search.db");
    let sample_dir = data_dir.join("sample_logs");
    fs::create_dir_all(&sample_dir).unwrap();
    let mut f = fs::File::create(sample_dir.join("rollout-1.jsonl")).unwrap();
    writeln!(f, "{{\"role\":\"user\",\"content\":\"hello\"}}").unwrap();
    writeln!(f, "{{\"role\":\"assistant\",\"content\":\"world\"}}").unwrap();

    let opts = IndexOptions {
        full: true,
        force_rebuild: true,
        watch: false,
        watch_once_paths: None,
        db_path,
        data_dir: data_dir.clone(),
        semantic: false,
        build_hnsw: false,
        embedder: "fastembed".to_string(),
        only_agents: None,
        rescan_agents: Vec::new(),
        progress: None,
        watch_interval_secs: 30,
    };

    // create empty index dir so Tantivy opens cleanly
    let _ = index_dir(&data_dir);

    c.bench_function("index_full_empty", |b| {
        b.iter(|| run_index(opts.clone(), None))
    });
}

/// Benchmark ingestion-time secret redaction. The harmless case is the hot path
/// for normal message content and should stay at one RegexSet scan with no
/// owned output allocation.
fn bench_redact_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("redact_text");
    let harmless = "ordinary tool output with code review notes and no credentials";
    let key_label = ["api", "_", "key", "="].concat();
    let key_value = ["abcdefgh", "12345678"].concat();
    let pat_prefix: String = ['g', 'h', 'p'].into_iter().collect();
    let pat_body = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghij"].concat();
    let credential_sample = format!("{key_label}{key_value} and token {pat_prefix}_{pat_body}");

    group.bench_function("harmless", |b| {
        b.iter(|| {
            let output = redact_text(std::hint::black_box(harmless));
            std::hint::black_box(output);
        });
    });
    group.bench_function("with_secrets", |b| {
        b.iter(|| {
            let output = redact_text(std::hint::black_box(credential_sample.as_str()));
            std::hint::black_box(output);
        });
    });
    group.finish();
}

/// Benchmark streaming vs batch indexing throughput.
///
/// This compares the performance of the streaming indexing mode (Opt 8.2)
/// against the original batch mode. Streaming uses bounded channels with
/// backpressure to reduce peak memory usage.
fn bench_streaming_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_vs_batch");

    // Test with multiple corpus sizes to see scaling behavior
    for &corpus_size in &[50, 100, 250] {
        // Create fresh corpus for each size
        let tmp = TempDir::new().unwrap();
        let (data_dir, db_path) = create_corpus(&tmp, corpus_size);

        // Ensure index directory exists
        let _ = index_dir(&data_dir);

        let base_opts = IndexOptions {
            full: true,
            force_rebuild: true,
            watch: false,
            watch_once_paths: None,
            db_path: db_path.clone(),
            data_dir: data_dir.clone(),
            semantic: false,
            build_hnsw: false,
            embedder: "fastembed".to_string(),
            only_agents: None,
            rescan_agents: Vec::new(),
            progress: None,
            watch_interval_secs: 30,
        };

        // Benchmark batch mode
        group.bench_with_input(
            BenchmarkId::new("batch", corpus_size),
            &corpus_size,
            |b, _| {
                // Disable streaming for batch mode
                // SAFETY: Benchmarks run single-threaded per test, no concurrent env access
                unsafe { std::env::set_var("CASS_STREAMING_INDEX", "0") };
                let opts = base_opts.clone();
                b.iter(|| {
                    // Clear any existing data for clean measurement
                    let _ = fs::remove_file(&opts.db_path);
                    let _ = fs::remove_dir_all(opts.data_dir.join("index"));
                    run_index(opts.clone(), None)
                });
            },
        );

        // Benchmark streaming mode
        group.bench_with_input(
            BenchmarkId::new("streaming", corpus_size),
            &corpus_size,
            |b, _| {
                // Enable streaming (default)
                // SAFETY: Benchmarks run single-threaded per test, no concurrent env access
                unsafe { std::env::set_var("CASS_STREAMING_INDEX", "1") };
                let opts = base_opts.clone();
                b.iter(|| {
                    // Clear any existing data for clean measurement
                    let _ = fs::remove_file(&opts.db_path);
                    let _ = fs::remove_dir_all(opts.data_dir.join("index"));
                    run_index(opts.clone(), None)
                });
            },
        );
    }

    // Reset to default
    // SAFETY: Benchmarks run single-threaded per test, no concurrent env access
    unsafe { std::env::remove_var("CASS_STREAMING_INDEX") };
    group.finish();
}

/// Benchmark channel overhead in streaming mode.
///
/// Measures the impact of different channel buffer sizes on throughput.
/// The STREAMING_CHANNEL_SIZE constant (32) balances memory vs throughput.
fn bench_channel_overhead(c: &mut Criterion) {
    let corpus_size = 100;
    let tmp = TempDir::new().unwrap();
    let (data_dir, db_path) = create_corpus(&tmp, corpus_size);
    let _ = index_dir(&data_dir);

    let opts = IndexOptions {
        full: true,
        force_rebuild: true,
        watch: false,
        watch_once_paths: None,
        db_path,
        data_dir: data_dir.clone(),
        semantic: false,
        build_hnsw: false,
        embedder: "fastembed".to_string(),
        only_agents: None,
        rescan_agents: Vec::new(),
        progress: None,
        watch_interval_secs: 30,
    };

    // Enable streaming mode for this benchmark
    // SAFETY: Benchmarks run single-threaded per test, no concurrent env access
    unsafe { std::env::set_var("CASS_STREAMING_INDEX", "1") };

    c.bench_function("streaming_channel_default", |b| {
        b.iter(|| {
            let opts = opts.clone();
            let _ = fs::remove_file(&opts.db_path);
            let _ = fs::remove_dir_all(opts.data_dir.join("index"));
            run_index(opts, None)
        });
    });

    // SAFETY: Benchmarks run single-threaded per test, no concurrent env access
    unsafe { std::env::remove_var("CASS_STREAMING_INDEX") };
}

fn scan_codex_conversation_count(data_dir: &Path, scan_roots: &[ScanRoot]) -> usize {
    let factories = get_connector_factories();
    let (_slug, build_codex) = factories
        .iter()
        .find(|(slug, _)| *slug == "codex")
        .expect("codex factory registered");
    let connector = build_codex();
    let ctx = ScanContext::with_roots(data_dir.to_path_buf(), scan_roots.to_vec(), None);
    let mut count = 0usize;
    connector
        .scan_with_callback(&ctx, &mut |_conversation| {
            count = count.saturating_add(1);
            Ok(())
        })
        .expect("codex scan_with_callback");
    count
}

/// Benchmark the fallback-safe Codex scan preflight for explicit scan roots.
/// The `preflight_then_explicit_files` row includes deterministic directory
/// enumeration plus the connector scan over explicit file roots; the
/// `explicit_files_scan_only` row isolates the connector-side savings available
/// once a faster enumerator produces the same explicit-file set.
fn bench_codex_scan_preflight(c: &mut Criterion) {
    let corpus_size = 1_000usize;
    let tmp = TempDir::new().unwrap();
    let (data_dir, _db_path) = create_corpus(&tmp, corpus_size);
    let directory_roots = vec![ScanRoot::local(data_dir.clone())];
    let preflight = preflight_codex_explicit_file_roots(&directory_roots, None);
    assert_eq!(preflight.fallback_roots, 0);
    assert_eq!(preflight.explicit_file_roots, corpus_size);
    assert_eq!(
        scan_codex_conversation_count(&data_dir, &directory_roots),
        scan_codex_conversation_count(&data_dir, &preflight.scan_roots)
    );

    let mut group = c.benchmark_group("codex_scan_preflight");
    group.sample_size(10);

    group.bench_function("directory_root_1000", |b| {
        b.iter(|| {
            let count = scan_codex_conversation_count(&data_dir, &directory_roots);
            std::hint::black_box(count);
        });
    });

    group.bench_function("preflight_then_explicit_files_1000", |b| {
        b.iter(|| {
            let preflight = preflight_codex_explicit_file_roots(&directory_roots, None);
            let count = scan_codex_conversation_count(&data_dir, &preflight.scan_roots);
            std::hint::black_box(count);
        });
    });

    group.bench_function("explicit_files_scan_only_1000", |b| {
        b.iter(|| {
            let count = scan_codex_conversation_count(&data_dir, &preflight.scan_roots);
            std::hint::black_box(count);
        });
    });

    group.finish();
}

/// Build a representative semantic-embedding input corpus. Mixes short,
/// long, markdown, and code-block inputs so the canonicalizer is exercised
/// across all of its branches.
fn build_semantic_corpus(count: usize) -> Vec<EmbeddingInput> {
    (0..count)
        .map(|i| {
            let content = match i % 7 {
                0 => format!("Plain message number {i} with some ordinary words to embed."),
                1 => format!("**Bold** and _italic_ markdown line {i}"),
                2 => format!(
                    "```rust\nfn example_{i}() {{\n    println!(\"code block {i}\");\n}}\n```\nfollow-up text"
                ),
                3 => format!("short line {i}"),
                4 => format!("   whitespace   galore   {i}   "),
                5 => format!(
                    "Unicode \u{00E9}\u{0301} (combining accent) and emoji \u{1F600} line {i}"
                ),
                _ => format!(
                    "Mixed line {i}: `inline_code`, [link](http://x), {{braces}}, \u{201C}curly quotes\u{201D}, \
                     and a moderately long tail so the canonicalizer has something to truncate when it hits \
                     its default cap of 2000 characters, well not really at this length but on longer inputs."
                ),
            };
            EmbeddingInput::new(i as u64, content)
        })
        .collect()
}

/// Benchmark the semantic embedding prep + embed loop.
///
/// Sweeps `CASS_SEMANTIC_BATCH_SIZE` across 32/128/256 and
/// `CASS_SEMANTIC_PREP_PARALLEL` on/off so we can measure the impact of the
/// parallel canonicalize+hash prep (perf refactor introduced with the
/// responsiveness governor work) against the legacy serial prep. Uses the
/// hash embedder so the benchmark stays fast, deterministic, and dependency-
/// free (no ONNX model download).
fn bench_semantic_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_embedding");
    group.sample_size(20);
    let messages = build_semantic_corpus(2_000);

    for &batch_size in &[32usize, 128, 256] {
        for &serial_label in &["parallel", "serial"] {
            let id = format!("batch{batch_size}_{serial_label}");
            // SAFETY: Criterion runs benchmarks single-threaded per fn so
            // env mutation is safe within this closure. We reset at the end.
            unsafe {
                std::env::set_var("CASS_SEMANTIC_BATCH_SIZE", batch_size.to_string());
                std::env::set_var(
                    "CASS_SEMANTIC_PREP_PARALLEL",
                    if serial_label == "serial" { "0" } else { "1" },
                );
            }
            // Build a fresh indexer per configuration so it picks up the env.
            let indexer = SemanticIndexer::new("hash", None).unwrap();
            group.bench_with_input(BenchmarkId::new("embed_messages", id), &(), |b, _| {
                b.iter(|| {
                    let out = indexer.embed_messages(&messages).unwrap();
                    std::hint::black_box(out);
                });
            });
        }
    }

    // Reset env so downstream benches are not polluted.
    // SAFETY: single-threaded cleanup outside any iter loop.
    unsafe {
        std::env::remove_var("CASS_SEMANTIC_BATCH_SIZE");
        std::env::remove_var("CASS_SEMANTIC_PREP_PARALLEL");
    }
    group.finish();
}

/// Benchmark the prototype sharded semantic writer against the existing
/// monolithic FSVI writer. The corpus uses the hash embedder so this isolates
/// file publishing, per-shard manifest cost, and optional shard-local ANN build
/// without depending on model downloads.
fn bench_semantic_shard_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_shard_generation");
    group.sample_size(10);
    let indexer = SemanticIndexer::new("hash", None).unwrap();
    let messages = build_semantic_corpus(128);
    let embeddings = indexer.embed_messages(&messages).unwrap();

    group.bench_function("monolithic_fsvi_build", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let index = indexer
                .build_and_save_index(embeddings.clone(), tmp.path())
                .unwrap();
            std::hint::black_box(index.record_count());
        });
    });

    group.bench_function("sharded_fsvi_build_32", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let outcome = indexer
                .build_and_save_index_shards(
                    embeddings.clone(),
                    tmp.path(),
                    SemanticShardBuildPlan {
                        tier: TierKind::Fast,
                        db_fingerprint: "bench-db-fp".to_string(),
                        model_revision: "hash".to_string(),
                        total_conversations: 128,
                        max_records_per_shard: 32,
                        build_ann: false,
                    },
                )
                .unwrap();
            std::hint::black_box(outcome.shard_count);
        });
    });

    group.bench_function("sharded_fsvi_hnsw_build_32", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let outcome = indexer
                .build_and_save_index_shards(
                    embeddings.clone(),
                    tmp.path(),
                    SemanticShardBuildPlan {
                        tier: TierKind::Fast,
                        db_fingerprint: "bench-db-fp-ann".to_string(),
                        model_revision: "hash".to_string(),
                        total_conversations: 128,
                        max_records_per_shard: 32,
                        build_ann: true,
                    },
                )
                .unwrap();
            std::hint::black_box(outcome.ann_index_paths.len());
        });
    });

    let manifest_tmp = TempDir::new().unwrap();
    indexer
        .build_and_save_index_shards(
            embeddings.clone(),
            manifest_tmp.path(),
            SemanticShardBuildPlan {
                tier: TierKind::Fast,
                db_fingerprint: "bench-db-fp-open".to_string(),
                model_revision: "hash".to_string(),
                total_conversations: 128,
                max_records_per_shard: 32,
                build_ann: true,
            },
        )
        .unwrap();
    group.bench_function("shard_manifest_load_summary", |b| {
        b.iter(|| {
            let manifest = SemanticShardManifest::load(manifest_tmp.path())
                .unwrap()
                .unwrap();
            let summary =
                manifest.summary(TierKind::Fast, indexer.embedder_id(), "bench-db-fp-open");
            std::hint::black_box((summary.ready_shards, summary.ann_ready_shards));
        });
    });

    let open_tmp = TempDir::new().unwrap();
    let monolithic_open_index = indexer
        .build_and_save_index(embeddings.clone(), open_tmp.path())
        .unwrap();
    std::hint::black_box(monolithic_open_index.record_count());
    drop(monolithic_open_index);
    let monolithic_open_path = vector_index_path(open_tmp.path(), indexer.embedder_id());

    let sharded_open_outcome = indexer
        .build_and_save_index_shards(
            embeddings.clone(),
            open_tmp.path(),
            SemanticShardBuildPlan {
                tier: TierKind::Fast,
                db_fingerprint: "bench-db-fp-vector-open".to_string(),
                model_revision: "hash".to_string(),
                total_conversations: 128,
                max_records_per_shard: 32,
                build_ann: false,
            },
        )
        .unwrap();
    let sharded_open_paths = sharded_open_outcome.index_paths;
    assert_eq!(sharded_open_paths.len(), 4);

    group.bench_function("monolithic_fsvi_mmap_open_128", |b| {
        b.iter(|| {
            let index = FsVectorIndex::open(&monolithic_open_path).unwrap();
            std::hint::black_box((index.record_count(), index.dimension()));
        });
    });

    group.bench_function("sharded_fsvi_mmap_open_all_4x32", |b| {
        b.iter(|| {
            let mut records = 0usize;
            for path in &sharded_open_paths {
                let index = FsVectorIndex::open(path).unwrap();
                records = records.saturating_add(index.record_count());
                std::hint::black_box(index.dimension());
            }
            std::hint::black_box(records);
        });
    });

    let stability_query = embeddings[42].embedding.clone();
    let monolithic_search_index = FsVectorIndex::open(&monolithic_open_path).unwrap();
    let sharded_search_indexes: Vec<_> = sharded_open_paths
        .iter()
        .map(|path| FsVectorIndex::open(path).unwrap())
        .collect();
    // Fetch every candidate for the stability proof. Per-shard top-k can drop
    // equal-score records before the global merge sees them, which is exactly
    // the sort of boundary artifact this benchmark should expose.
    let monolithic_signature = {
        let mut hits = monolithic_search_index
            .search_top_k(
                &stability_query,
                monolithic_search_index.record_count(),
                None,
            )
            .unwrap();
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        hits.truncate(10);
        hits.into_iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect::<Vec<_>>()
    };
    let sharded_signature = {
        let mut hits = Vec::new();
        for index in &sharded_search_indexes {
            hits.extend(
                index
                    .search_top_k(&stability_query, index.record_count(), None)
                    .unwrap(),
            );
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        hits.truncate(10);
        hits.into_iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(monolithic_signature, sharded_signature);

    group.bench_function("monolithic_fsvi_search_top10_128", |b| {
        b.iter(|| {
            let hits = monolithic_search_index
                .search_top_k(&stability_query, 10, None)
                .unwrap();
            std::hint::black_box(hits);
        });
    });

    group.bench_function("sharded_exact_full_merge_search_top10_4x32", |b| {
        b.iter(|| {
            let mut hits = Vec::new();
            for index in &sharded_search_indexes {
                hits.extend(
                    index
                        .search_top_k(&stability_query, index.record_count(), None)
                        .unwrap(),
                );
            }
            hits.sort_by(|left, right| {
                left.cmp_by_score(right)
                    .then_with(|| left.doc_id.cmp(&right.doc_id))
            });
            hits.truncate(10);
            std::hint::black_box(hits);
        });
    });

    group.finish();
}

/// Larger semantic shard build probe intended for external peak-RSS wrappers
/// such as `/usr/bin/time -v`. Criterion does not report memory itself, so keep
/// these rows narrow and filterable: run one row at a time when collecting RSS.
fn bench_semantic_shard_generation_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_shard_generation_large");
    group.sample_size(10);
    let indexer = SemanticIndexer::new("hash", None).unwrap();
    let messages = build_semantic_corpus(4_096);
    let embeddings = indexer.embed_messages(&messages).unwrap();

    group.bench_function("monolithic_fsvi_build_4096", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let index = indexer
                .build_and_save_index(embeddings.clone(), tmp.path())
                .unwrap();
            std::hint::black_box(index.record_count());
        });
    });

    group.bench_function("sharded_fsvi_build_4096x256", |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let outcome = indexer
                .build_and_save_index_shards(
                    embeddings.clone(),
                    tmp.path(),
                    SemanticShardBuildPlan {
                        tier: TierKind::Fast,
                        db_fingerprint: "bench-db-fp-large-rss".to_string(),
                        model_revision: "hash".to_string(),
                        total_conversations: 4_096,
                        max_records_per_shard: 256,
                        build_ann: false,
                    },
                )
                .unwrap();
            std::hint::black_box((outcome.shard_count, outcome.doc_count));
        });
    });

    group.finish();
}

/// Benchmark the full ingest pipeline with and without the parallel
/// pre-compute of `map_to_internal`. The `CASS_STREAMING_INDEX` toggle
/// doesn't affect the hoist; both modes exercise it. We compare a
/// governor-enabled run (default) against a governor-disabled run to expose
/// whether the governor is silently costing throughput on an otherwise
/// idle box.
fn bench_ingest_with_responsiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_responsiveness");
    group.sample_size(15);
    let corpus_size = 200;

    for &(label, disable_value) in &[("governor_on", "0"), ("governor_off", "1")] {
        let tmp = TempDir::new().unwrap();
        let (data_dir, db_path) = create_corpus(&tmp, corpus_size);
        let _ = index_dir(&data_dir);

        let opts = IndexOptions {
            full: true,
            force_rebuild: true,
            watch: false,
            watch_once_paths: None,
            db_path,
            data_dir: data_dir.clone(),
            semantic: false,
            build_hnsw: false,
            embedder: "fastembed".to_string(),
            only_agents: None,
            rescan_agents: Vec::new(),
            progress: None,
            watch_interval_secs: 30,
        };

        // SAFETY: Criterion benches run single-threaded.
        unsafe {
            std::env::set_var("CASS_RESPONSIVENESS_DISABLE", disable_value);
        }

        group.bench_with_input(BenchmarkId::new(label, corpus_size), &(), |b, _| {
            b.iter(|| {
                let opts = opts.clone();
                let _ = fs::remove_file(&opts.db_path);
                let _ = fs::remove_dir_all(opts.data_dir.join("index"));
                run_index(opts, None)
            });
        });
    }

    // SAFETY: single-threaded cleanup outside any iter loop.
    unsafe {
        std::env::remove_var("CASS_RESPONSIVENESS_DISABLE");
    }
    group.finish();
}

/// Measured A/B of the post-flip defaults (Cards 1/2/3 all enabled) vs
/// the pre-flip "legacy" configuration (static governor, per-message
/// consumer, shadow observer off). The goal is to answer the user's
/// question: does flipping all three defaults on actually help or hurt
/// end-to-end wall-clock on a realistic-sized ingest?
///
/// We also run the two middle corners so per-card attribution is
/// possible: toggle combine in isolation and toggle the governor in
/// isolation against the legacy baseline.
///
/// Each configuration uses `--force-rebuild` so the measured wall-clock
/// includes the full scan + persist + Tantivy index path. Corpus size
/// 200 matches the existing `ingest_responsiveness` bench so the
/// criterion baseline comparator can attribute the delta.
fn bench_card_defaults_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("card_defaults_ab");
    group.sample_size(10);
    let corpus_size = 200;

    // Four cells. Each is (label, (governor, combine, shadow)) tuple.
    // `governor`: "static" (legacy) vs "conformal" (new default)
    // `combine`:  "0" (legacy) vs "1" (new default)
    // `shadow`:   "off" (legacy) vs "shadow" (new default)
    let cells: [(&str, &str, &str, &str); 4] = [
        ("legacy_all_off", "static", "0", "off"),
        ("new_all_on", "conformal", "1", "shadow"),
        ("only_combine_on", "static", "1", "off"),
        ("only_governor_on", "conformal", "0", "off"),
    ];

    for &(label, governor, combine, shadow) in &cells {
        let tmp = TempDir::new().unwrap();
        let (data_dir, db_path) = create_corpus(&tmp, corpus_size);
        let _ = index_dir(&data_dir);

        let opts = IndexOptions {
            full: true,
            force_rebuild: true,
            watch: false,
            watch_once_paths: None,
            db_path,
            data_dir: data_dir.clone(),
            semantic: false,
            build_hnsw: false,
            embedder: "fastembed".to_string(),
            only_agents: None,
            rescan_agents: Vec::new(),
            progress: None,
            watch_interval_secs: 30,
        };

        // SAFETY: criterion benches are single-threaded per-fn.
        unsafe {
            std::env::set_var("CASS_RESPONSIVENESS_CALIBRATION", governor);
            std::env::set_var("CASS_STREAMING_CONSUMER_COMBINE", combine);
            std::env::set_var("CASS_INDEXER_PARALLEL_WAL", shadow);
        }

        group.bench_with_input(BenchmarkId::new(label, corpus_size), &(), |b, _| {
            b.iter(|| {
                let opts = opts.clone();
                let _ = fs::remove_file(&opts.db_path);
                let _ = fs::remove_dir_all(opts.data_dir.join("index"));
                run_index(opts, None)
            });
        });
    }

    // SAFETY: single-threaded cleanup outside any iter loop.
    unsafe {
        std::env::remove_var("CASS_RESPONSIVENESS_CALIBRATION");
        std::env::remove_var("CASS_STREAMING_CONSUMER_COMBINE");
        std::env::remove_var("CASS_INDEXER_PARALLEL_WAL");
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_index_full,
    bench_redact_text,
    bench_streaming_vs_batch,
    bench_channel_overhead,
    bench_codex_scan_preflight,
    bench_semantic_embedding,
    bench_semantic_shard_generation,
    bench_semantic_shard_generation_large,
    bench_ingest_with_responsiveness,
    bench_card_defaults_ab,
);
criterion_main!(benches);
