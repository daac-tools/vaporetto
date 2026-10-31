use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn write_model(name: &str) -> PathBuf {
    let data = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../resources/model.bin"
    ))
    .unwrap();
    let path = std::env::temp_dir().join(format!(
        "vaporetto-predict-test-{}-{name}.model.zst",
        std::process::id()
    ));
    fs::write(&path, zstd::encode_all(data.as_slice(), 0).unwrap()).unwrap();
    path
}

fn run(name: &str, args: &[&str], input: &str) -> Output {
    let model = write_model(name);
    let mut child = Command::new(env!("CARGO_BIN_EXE_predict"))
        .arg("--model")
        .arg(&model)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Err(e) = child.stdin.take().unwrap().write_all(input.as_bytes()) {
        assert_eq!(io::ErrorKind::BrokenPipe, e.kind(), "{e}");
    }
    let output = child.wait_with_output().unwrap();
    fs::remove_file(model).unwrap();
    output
}

#[test]
fn test_tag_scores_requires_predict_tags() {
    let output = run(
        "tag_scores_requires_predict_tags",
        &["--tag-scores"],
        "猫\n",
    );

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("--predict-tags"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn test_tag_scores_with_empty_lines() {
    for no_norm in [false, true] {
        let mut args = vec!["--predict-tags", "--tag-scores"];
        if no_norm {
            args.push("--no-norm");
        }
        let output = run("tag_scores_with_empty_lines", &args, "\n火星猫だ\n\n");

        assert!(output.status.success(), "no_norm={no_norm}");
        assert_eq!(
            "\n\n\
             火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ\n\
             火星\t名詞:0\tカセー:0\n\
             猫\t名詞:0\tネコ:0\n\
             だ\t助動詞:0\tダ:0\n\
             \n\
             \n\n",
            String::from_utf8(output.stdout).unwrap(),
            "no_norm={no_norm}",
        );
    }
}

#[test]
fn test_scores() {
    for no_norm in [false, true] {
        let mut args = vec!["--scores"];
        if no_norm {
            args.push("--no-norm");
        }
        let output = run("scores", &args, "火星猫だ\n");

        assert!(output.status.success(), "no_norm={no_norm}");
        assert_eq!(
            "火星 猫 だ\n0:火星 0\n1:星猫 32767\n2:猫だ 26247\n\n",
            String::from_utf8(output.stdout).unwrap(),
            "no_norm={no_norm}",
        );
    }
}

#[test]
fn test_scores_with_original_text() {
    // "a" is normalized to "ａ" before prediction, but the original text must be printed.
    let output = run(
        "scores_with_original_text",
        &["--scores", "--predict-tags", "--tag-scores"],
        "猫だa\n",
    );

    assert!(output.status.success());
    // The first line is the tokenized text.
    let stdout = String::from_utf8(output.stdout).unwrap();
    let (_, scores) = stdout.split_once('\n').unwrap();
    assert_eq!("0:猫だ -6520\n1:だa 32767\n\n猫だ\na\n\n", scores);
}
