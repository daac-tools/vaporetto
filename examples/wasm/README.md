# Wasm Example

## How to launch on your environment?

You can also launch the demo server on your machine using [trunk](https://github.com/thedodd/trunk).

Run the following commands in this directory:
```
# Installs wasm target of Rust compiler.
rustup target add wasm32-unknown-unknown

# Installs trunk
cargo install trunk
cargo install wasm-bindgen-cli

# Downloads and extracts the model file
wget https://github.com/daac-tools/vaporetto-models/releases/download/v0.5.0/bccwj-suw+unidic_pos+pron.tar.xz
tar xf ./bccwj-suw+unidic_pos+pron.tar.xz

# Converts the model file created by Vaporetto 0.6 or earlier into the current format
cargo run --release --manifest-path ../../Cargo.toml -p convert_bincode_model -- \
  --model-in ./bccwj-suw+unidic_pos+pron/bccwj-suw+unidic_pos+pron.model.zst \
  --model-out ./src/bccwj-suw+unidic_pos+pron.model.zst

# Builds and launches the server
# Note: We recommend using --release flag to reduce loading time.
trunk serve --release
```

For ARM Mac, you may need to install binaryen to build wasm-opt.
```
brew install binaryen
```
