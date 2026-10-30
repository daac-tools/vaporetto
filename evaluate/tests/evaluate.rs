use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const REF_TAGGED: &str =
    "まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ";
const REF_WRONG_TAG: &str =
    "まぁ/動詞/ダミー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ";
const REF_UNTAGGED: &str = "まぁ 社長 は 火星 猫 だ";

fn write_model(name: &str) -> PathBuf {
    let data = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../resources/model.bin"
    ))
    .unwrap();
    let path = std::env::temp_dir().join(format!(
        "vaporetto-evaluate-test-{}-{name}.model.zst",
        std::process::id()
    ));
    fs::write(&path, zstd::encode_all(data.as_slice(), 0).unwrap()).unwrap();
    path
}

fn word_f1(name: &str, args: &[&str], input: &str) -> String {
    let model = write_model(name);
    let mut child = Command::new(env!("CARGO_BIN_EXE_evaluate"))
        .arg("--model")
        .arg(&model)
        .args(["--metric", "word"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    fs::remove_file(model).unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("F1: "))
        .unwrap()
        .to_string()
}

#[test]
fn test_word_without_predict_tags() {
    // Tags are not compared if they are not predicted.
    for no_norm in [false, true] {
        let args: &[&str] = if no_norm { &["--no-norm"] } else { &[] };
        for reference in [REF_TAGGED, REF_WRONG_TAG, REF_UNTAGGED] {
            assert_eq!(
                "1",
                word_f1("without_predict_tags", args, reference),
                "no_norm={no_norm}, reference={reference}",
            );
        }
    }
}

#[test]
fn test_word_with_predict_tags() {
    for no_norm in [false, true] {
        let args: &[&str] = if no_norm {
            &["--predict-tags", "--no-norm"]
        } else {
            &["--predict-tags"]
        };
        assert_eq!("1", word_f1("with_predict_tags", args, REF_TAGGED));
        // Only tags in the reference are compared.
        assert_eq!("1", word_f1("with_predict_tags", args, REF_UNTAGGED));
        assert_eq!(
            "1",
            word_f1(
                "with_predict_tags",
                args,
                "まぁ/名詞 社長/名詞 は/助詞 火星/名詞 猫/名詞 だ/助動詞",
            ),
        );
        // 5 of 6 words are correct.
        assert_eq!(
            "0.8333333333333334",
            word_f1("with_predict_tags", args, REF_WRONG_TAG),
            "no_norm={no_norm}",
        );
    }
}
