use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use regex::Regex;
use tokenizers::Tokenizer;

const CHUNK_SIZE: usize = 5000;
const WRITE_BUFFER_SIZE: usize = 1 << 20;

fn split_long_indices(text: &str, re: &Regex) -> Vec<(usize, usize)> {
    let mut parts = Vec::with_capacity(8);
    let mut start = 0;

    for mat in re.find_iter(text) {
        parts.push((start, mat.end()));
        start = mat.end();
    }

    if start < text.len() {
        parts.push((start, text.len()));
    }

    if parts.is_empty() {
        parts.push((0, text.len()));
    }

    parts
}

fn process_chunk_parallel(
    lines: &[&str],
    tokenizer: &Tokenizer,
    re: &Regex,
    bos_id: u16,
    eos_id: u16,
) -> Vec<u16> {
    let line_segments: Vec<(Vec<(usize, usize)>, &str)> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|&line| (split_long_indices(line, re), line))
        .collect();

    let mut segments: Vec<(&str, bool, bool)> = Vec::with_capacity(lines.len() * 2);

    for (indices, line) in &line_segments {
        let len = indices.len();
        for (i, &(start, end)) in indices.iter().enumerate() {
            segments.push((&line[start..end], i == 0, i == len - 1));
        }
    }

    if segments.is_empty() {
        return Vec::new();
    }

    let texts: Vec<&str> = segments.iter().map(|(t, _, _)| *t).collect();

    let encodings = tokenizer
        .encode_batch(texts, false)
        .expect("Batch encoding failed");

    let estimated_capacity: usize = encodings.iter().map(|e| e.get_ids().len() + 2).sum();
    let mut result = Vec::with_capacity(estimated_capacity);

    for (enc, (_, is_first, is_last)) in encodings.iter().zip(segments.iter()) {
        if *is_first {
            result.push(bos_id);
        }
        for &id in enc.get_ids() {
            result.push(id as u16);
        }
        if *is_last {
            result.push(eos_id);
        }
    }

    result
}

fn tokens_to_bytes(tokens: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 2);
    for &t in tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    bytes
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} <tokenizer.json> <input.txt> <output.bin> [chunk_size]", program);
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 || args.len() > 5 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let tokenizer_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];
    let chunk_size: usize = args.get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(CHUNK_SIZE);

    let num_threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("══════════════════════════════════════════════════════════");
    println!("   Token Converter (Parallel)");
    println!("══════════════════════════════════════════════════════════");
    println!(" Tokenizer:  {}", tokenizer_path);
    println!(" Input:      {}", input_path);
    println!(" Output:     {}", output_path);
    println!(" Threads:    {}", num_threads);
    println!(" Chunk Size: {}", chunk_size);
    println!("══════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .expect(&format!("Failed to load tokenizer: {}", tokenizer_path));

    let bos_id = tokenizer.token_to_id("<|bos|>").unwrap_or(2) as u16;
    let eos_id = tokenizer.token_to_id("<|eos|>").unwrap_or(1) as u16;
    let re = Regex::new(r"[。？！；.?!;][ \n]?").unwrap();

    println!("[1/4] Tokenizer loaded (BOS={}, EOS={})", bos_id, eos_id);

    let input_file = File::open(input_path)?;
    let mmap = unsafe { Mmap::map(&input_file)? };
    let content = std::str::from_utf8(&mmap).expect("Invalid UTF-8");
    let input_size_mb = mmap.len() as f64 / (1024.0 * 1024.0);

    println!("[2/4] Input mapped: {:.2} MB", input_size_mb);

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    println!("[3/4] Processing {} lines...", total_lines);

    let processed_chunks = AtomicUsize::new(0);
    let total_chunks = (total_lines + chunk_size - 1) / chunk_size;

    let results: Vec<Vec<u16>> = lines
        .par_chunks(chunk_size)
        .map(|chunk| {
            let result = process_chunk_parallel(chunk, &tokenizer, &re, bos_id, eos_id);
            let done = processed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 20 == 0 || done == total_chunks {
                eprint!("\r      Progress: {}/{} chunks ({:.1}%)", 
                    done, total_chunks, done as f64 / total_chunks as f64 * 100.0);
            }
            result
        })
        .collect();

    eprintln!();

    let total_tokens: usize = results.iter().map(|v| v.len()).sum();

    let mut all_tokens = Vec::with_capacity(total_tokens);
    for chunk_tokens in results {
        all_tokens.extend(chunk_tokens);
    }

    let bytes = tokens_to_bytes(&all_tokens);

    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, output_file);
    writer.write_all(&bytes)?;
    writer.flush()?;

    let duration = start_time.elapsed();
    let output_size_mb = (total_tokens * 2) as f64 / (1024.0 * 1024.0);

    println!("[4/4] Write complete");
    println!();
    println!("══════════════════════════════════════════════════════════");
    println!("   Conversion Complete");
    println!("══════════════════════════════════════════════════════════");
    println!(" Input:        {:.2} MB", input_size_mb);
    println!(" Output:       {:.2} MB", output_size_mb);
    println!(" Total Tokens: {}", total_tokens);
    println!(" Time:         {:.2?}", duration);
    println!(" Throughput:   {:.2} MB/s", input_size_mb / duration.as_secs_f64());
    println!("══════════════════════════════════════════════════════════");

    Ok(())
}
