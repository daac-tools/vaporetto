# Embedded device example of Vaporetto

## How to build?

This example is written for embedded devices.
Install an appropriate compiler following [the documentation](https://docs.rust-embedded.org/book/) beforehand.

We comfirmed the behaviour on [STM32F3DISCOVERY](http://www.st.com/en/evaluation-tools/stm32f3discovery.html).
This device has just 256K Flash and 40K RAM, so we recommend using a very tiny model trained with a low
`--cost` option and without a dictionary.

The model file is read and embedded on the building phase, so you need to specify the model file using `VAPORETTO_MODEL_PATH` as follows:
```sh
VAPORETTO_MODEL_PATH=$PWD/model.zst cargo +nightly build --release
```

This example automatically launches GDB on the `run` command, so you can quickly run this example using two terminals.
First, run the following command to connect to the device:
```sh
openocd
```
Then, run the following command in another terminal:
```sh
VAPORETTO_MODEL_PATH=$PWD/model.zst cargo +nightly run --release
```

If it works correctly, tokenized results will be shown on the first terminal.
