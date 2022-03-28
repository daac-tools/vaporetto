# WebAssembly example of Vaporetto

## How to build?

1. Build a model file following the [documentation](../README.md).

2. Build a JS file containing a web assembly using `build_portable_js.py`.
   This script requires a model file, an identifier, and an output path.

   The identifier must consist of alphanumeric characters and underscores.
   ```
   ./build_portable_js.py --model <MODEL_FILE> --identifier <IDENTIFIER> --output <OUTPUT>
   ```

3. You can use the generated JS file like the follwing code:
   ```html
   <!DOCTYPE html>
   <html>
       <head>
           <!-- Replace vaporetto.js with the script you generated. -->
           <script src="vaporetto.js"></script>
           <script>
               // Replace IDENTIFIER with a string you specified.
               vaporetto_IDENTIFIER().then((Vaporetto) => {
                   const vaporetto = Vaporetto.new("DG");
                   const tokens = vaporetto.tokenize("火星猫の生態");
                   console.log(tokens);
               });
           </script>
       </head>
       <body>
       </body>
   </html>
   ```
