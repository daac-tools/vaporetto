# WebAssembly example of Vaporetto

1. Build a model file:
   ```
   # jp-0.4.7-5.mod is a model file distributed by KyTea.
   cargo run --release -p convert_kytea_model -- --model-in ./jp-0.4.7-5.mod --model-out ../model/model.bin
   ```

2. Build a web assembly:
   ```
   % wasm-pack build --release --target web
   ```

3. Launch the server:
   ```
   % python3 -m http.server 8000
   ```

4. Open http://localhost:8000/www
