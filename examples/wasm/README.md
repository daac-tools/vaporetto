# Wasm Example

Source code of [the demo page](https://daac-tools.github.io/vaporetto/).

## How to launch on your environment?

You can also launch the demo server on your machine using [trunk](https://github.com/thedodd/trunk).

Run the following commands in this directory:
```
# Installs wasm target of Rust compiler.
rustup target add wasm32-unknown-unknown

# Installs trunk
cargo install trunk

# Downloads and extracts the model file
wget https://github.com/daac-tools/vaporetto/releases/download/v0.5.0/bccwj-suw+unidic+tag-huge.tar.xz
tar xf ./bccwj-suw+unidic+tag-huge.tar.xz
mv ./bccwj-suw+unidic+tag-huge/bccwj-suw+unidic+tag-huge.model.zst ./src/

# Builds and launches the server
# Note: We recommend using --release flag to reduce loading time.
trunk serve --release
```
