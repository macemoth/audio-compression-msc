# audio-compression-msc

```
audio-compression-msc
| 
|───toco.py (Topological audio compressor, lossy)
|───ltoco.py (Topological audio compressor, almost-lossless)
|───requirements.txt (Python requirements)
|───monoadagio.wav (Audio test file)
|───monotrance.wav (Audio test file)
└───util (Utility files)
|  └─── ArithmeticBitCoder.py
```

## Installation

Install dependencies using `pip3 install -r requirements.txt`

## Topological Data Compression

1. Install the TSC library using `pip3 install topological-signal-compression`
2. Run e.g. `python3 toco.py c monoadagio.wav` to compress lossily, or `python3 ltoco.py c monoadagio.wav` to compress lossless
3. Run `python3 toco.py d monoadagio.tc` to decompress

## Tensor factorisation

For working through the complete `tensor.ipynb` notebook, TTHRESH (available at <https://github.com/rballester/tthresh>) is required.

1. Install TTHRESH and the tensorly library `pip3 install tensorly`

## Fractal image compression

1. Run `pip3 install scipy`