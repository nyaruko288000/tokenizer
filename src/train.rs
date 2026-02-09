use std::env;
use std::time::Instant;

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::tokenizer::{AddedToken, Result, TokenizerImpl};

fn print_usage(program: &str) {
    eprintln!("Usage: {} <input.txt> <output.json> [vocab_size]", program);
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  input.txt    - Training corpus (UTF-8)");
    eprintln!("  output.json  - Output tokenizer file");
    eprintln!("  vocab_size   - Vocabulary size (default: 8192)");
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let vocab_size: usize = args.get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);

    let num_threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap_or(());

    println!("══════════════════════════════════════════════════════════");
    println!("   BPE Tokenizer Trainer (256-Byte Base)");
    println!("══════════════════════════════════════════════════════════");
    println!(" Input:       {}", input_file);
    println!(" Output:      {}", output_file);
    println!(" Vocab Size:  {}", vocab_size);
    println!(" Threads:     {}", num_threads);
    println!("══════════════════════════════════════════════════════════");
    println!();

    let start = Instant::now();

    // 256 字节字母表
    let byte_alphabet = ByteLevel::alphabet();
    println!("[1/6] Loaded 256-byte alphabet ({} chars)", byte_alphabet.len());

    // 特殊 Token
    let special_tokens = vec![
        AddedToken::from("<|pad|>", true).single_word(true),
        AddedToken::from("<|eos|>", true).single_word(true),
        AddedToken::from("<|bos|>", true).single_word(true),
        AddedToken::from("<|unk|>", true).single_word(true),
    ];

    // BPE 训练器
    let mut trainer = BpeTrainerBuilder::default()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .initial_alphabet(byte_alphabet)
        .special_tokens(special_tokens)
        .limit_alphabet(1000)
        .continuing_subword_prefix("".into())
        .end_of_word_suffix("".into())
        .build();

    println!("[2/6] BPE trainer configured");

    // Tokenizer 实例
    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(BPE::default());

    let byte_level = ByteLevel::default()
        .add_prefix_space(false)
        .trim_offsets(true)
        .use_regex(true);

    tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(byte_level.clone()));
    tokenizer.with_decoder(DecoderWrapper::ByteLevel(byte_level));

    println!("[3/6] Tokenizer initialized");

    // 训练
    println!("[4/6] Training started...");
    println!();

    let train_start = Instant::now();
    let files = vec![input_file.to_string()];
    tokenizer.train_from_files(&mut trainer, files)?;
    let train_duration = train_start.elapsed();

    println!();
    println!("      Training completed in {:?}", train_duration);

    // 后处理器
    let bos_id = tokenizer.token_to_id("<|bos|>").expect("BOS not found");
    let eos_id = tokenizer.token_to_id("<|eos|>").expect("EOS not found");

    let post_processor = TemplateProcessing::builder()
        .try_single("<|bos|> $A <|eos|>")
        .unwrap()
        .try_pair("<|bos|> $A <|eos|> <|bos|> $B <|eos|>")
        .unwrap()
        .special_tokens(vec![
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ])
        .build()?;

    tokenizer.with_post_processor(PostProcessorWrapper::Template(post_processor));
    println!("[5/6] Post-processor configured (BOS={}, EOS={})", bos_id, eos_id);

    // 保存
    tokenizer.save(output_file, true)?;
    println!("[6/6] Tokenizer saved to {}", output_file);

    let total_duration = start.elapsed();
    let final_vocab_size = tokenizer.get_vocab_size(true);

    println!();
    println!("══════════════════════════════════════════════════════════");
    println!("   Training Complete");
    println!("══════════════════════════════════════════════════════════");
    println!(" Final Vocab:    {}", final_vocab_size);
    println!(" Training Time:  {:?}", train_duration);
    println!(" Total Time:     {:?}", total_duration);
    println!("══════════════════════════════════════════════════════════");

    // 测试
    println!();
    println!("Encoding Tests:");
    let test_cases = ["你好，世界！", "Hello, World!", "萧炎冷笑一声。"];
    for text in test_cases {
        let encoding = tokenizer.encode(text, false)?;
        println!("  \"{}\" => {} tokens", text, encoding.get_ids().len());
    }

    Ok(())
}
