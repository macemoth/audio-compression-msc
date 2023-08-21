# audio-compression-msc

```
audio-compression-msc
├── compressors (Prototype compressors)
│   ├── ltoco.py (Lossless topological compressor)
│   ├── toco.py (Lossy topological compressor)
│   ├── utils
│   │   ├── ArithmeticBitCoder.py
│   └── veco.py (Vector quantisation compressor)
├── data
│   ├── adagio.wav
│   └── ...
├── notebooks
│   ├── fractal.ipynb
│   ├── nn.ipynb
│   ├── tensor.ipynb
│   └── vector_quantisation.ipynb
├── recompressor (MP3 recompressor)
│   ├── ArithmeticBinaryDecoder.py (Byte-level compressor)
│   ├── ArithmeticBinaryEncoder.py (Byte-level compressor)
│   ├── ArithmeticBitCoder.py (Bit-level compressor)
│   ├── Frame.py
│   ├── ...
│   ├── MP3Predictor.py (Probability model for ArithmeticBitCoder.py)
│   ├── NaiveArithmeticBitCoder.py (Probability model for ArithmeticBitCoder.py)
│   ├── Recompressor.py
│   ├── arithmetic_model.py (Probability model for ArithmeticBinary(En,De)coder.py)
│   ├── main.py (Entry point for recompressor)
│   └── ...
└── requirements.txt
```

## Installation

1. Install dependencies using `pip3 install -r requirements.txt`
2. *Only required for compressors:* Copy desired audio samples from `data/` to `compressors/`

## Topological Data Compression

1. `cd compressors`
2. Run e.g. `python3 toco.py c monoadagio.wav` to compress lossily, or `python3 ltoco.py c monoadagio.wav` to compress lossless
3. Run `python3 toco.py d monoadagio.tc` to decompress

Running instructions are printed when running `python3 toco.py` or `python3 ltoco.py`

If input file is not mono, it will be converted, indicating too high compression ratios.

## Tensor factorisation

1. Install TTHRESH and the python dependencies `pip3 install tensorly notebook`
2. Run `jupyter notebook` and open the `tensor.ipynb` notebook

For working through the complete `tensor.ipynb` notebook, TTHRESH (available at <https://github.com/rballester/tthresh>) is required.

## Vector quantisation

1. `cd compressors`
1. Run e.g. `python3 veco.py c 4 adagio.wav` to compress
2. Run `python3 veco.py d adagio.vc` to decompress

To run notebook
1. Install `pip3 install notebook`
2. Run `jupyter notebook` and open the `vector_quantisation.ipynb` notebook

## Fractal image compression

1. Run `pip3 install scipy notebook `
2. Run `jupyter notebook` and open the `fractal.ipynb` notebook

## Neural networks

1. Run `pip3 install notebook`
2. Run `jupyter notebook` and open the `nn.ipynb` notebook

## MP3 recompressor

1. `cd compressors`
2. Run e.g. `python3 main.py trance.mp3` to recompress

By default, the optimised MP3 probability model is used.
To choose the order-0 arithmetic coder or change other aspects of compression, modify the `Frame.py` file and follow the comments.

This module reuses the MP3 decoder code from <https://github.com/tomershay100/mp3-steganography-lib>.