import sys
import os
import numpy as np
from tsc import compress_tsc, reconstruct_tsc
import soundfile as sf
import librosa
import librosa.core.spectrum
from utils.ArithmeticBitCoder import ArithmeticBitCoder
from tqdm import tqdm

N_FFT = 1024 # number of samples for every STFT frame

def compress(infile):
    # Read file
    fname = infile.split("/")[-1].split(".")[0]
    _, samplerate = sf.read(infile)
    original, fs = librosa.core.load(infile, sr=samplerate, mono=True)
    # channels, _ = original.shape
    
    # Apply STFT
    # spectrogram = [librosa.core.stft(original[0], n_fft=N_FFT), librosa.core.stft(original[1], n_fft=N_FFT)]
    spectrogram = np.abs(librosa.core.stft(original, n_fft=N_FFT))
    spectrogram = (10 * np.sqrt(spectrogram, out=np.zeros_like(spectrogram), where=(spectrogram != 0))).round()
    
    # Apply TSC
    print("Starting TSC compression")
    m, n = spectrogram.shape
    indexes = np.zeros([m, n], dtype=bool)
    amplitudes = bytearray()
    full_index = np.array(range(n))
    for i in tqdm(range(m)): # go frequency line by frequency line (works better than via frequency)
        c = compress_tsc(np.column_stack([full_index, spectrogram[i, :]]), num_indices_to_keep="all")
        amplitudes.extend(np.clip(c[:,1].round(), -127, 127).astype(np.int8).tobytes())
        indexes[i, c[:, 0].astype(int)] = True
    
    indexbytes = np.packbits(indexes)

    # Entropic coding
    print("Starting entropic coding")
    ae = ArithmeticBitCoder()
    cindexes = ae.encode(indexbytes)
    ae.__init__() # Reset arithmetic coder
    camplitudes = ae.encode(amplitudes)
    
    # Pack into file
    header = make_header(m, n, len(cindexes), len(camplitudes), samplerate)
    write_file(f"{fname}.tc", header + cindexes + camplitudes)
    
    print(f"Compression ratio: 1:{os.path.getsize(infile) / (len(header) + len(cindexes) + len(camplitudes))}")

def make_header(m, n, len_indexes, len_amplitudes, samplerate):
    # Header format
    # | m (4 bytes)  | n (4 bytes) | # of index bytes (4 bytes) |
    # of amplitude bytes (4 bytes) | samplerate (4 bytes) |
    # Total length: 20 bytes
    return bytes(np.array([m, n, len_indexes, len_amplitudes, samplerate], dtype=np.int32))

def decompress(infile):
    fname = infile.split("/")[-1].split(".")[0]
    # Unpack binary file
    len_header = 20
    offset = 0
    print("Unpacking compressed file")
    with open(infile, "rb") as file:
        fbytes = file.read()
        file.close()
    m, n, len_indexes, len_amplitudes, samplerate = read_header(fbytes[0:len_header])
    if (len_header + len_indexes + len_amplitudes) != len(fbytes):
        print(
            f"File {fbytes} has length {len(fbytes)}. Expected length from header: {len_header + len_indexes + len_amplitudes}")
        return
    offset += len_header

    print("Starting entropic decoding")
    # Undo entropic coding
    ae = ArithmeticBitCoder()
    indexbytes = ae.decode(bytearray(fbytes[offset:offset + len_indexes]))
    offset += len_indexes
    ae.__init__()
    amplitudebytes = ae.decode(bytearray(fbytes[offset:offset + len_amplitudes]))
    
    # Undo serialisation
    indexes = np.unpackbits(indexbytes)
    indexes = indexes[:m*n].reshape([m, n]) # remove pa
    indexes = np.array(indexes, dtype=bool)

    print("Starting TSC reconstruction")
    # Reconstruct with TSC
    amplitudes = np.zeros([m, n])
    
    offset = 0
    fullindex = np.array(range(n))
    for i in tqdm(range(m)):
        boolindex = indexes[i, :]
        len_entries = sum(boolindex) # the number of compressed amplitudes is equal to the number of indexes set to true
        tsc_amplitudes = np.frombuffer(amplitudebytes[offset:offset+len_entries], dtype=np.int8)
        index = fullindex[boolindex] # convert boolean index into number index
        amplitudes[i, :] = reconstruct_tsc(np.column_stack([index, tsc_amplitudes]))[:, 1]
        offset += len_entries

    print("Writing waveform signal")
    # Reconstruct waveform
    reconstructed_waveform = librosa.core.spectrum.griffinlim(np.square(amplitudes/10)) # Undo quantisation and STFT
    sf.write(f"{fname}.wav", reconstructed_waveform, samplerate)
    
def read_header(headerbytes):
    m, n, len_indexes, len_amplitudes, samplerate = np.frombuffer(headerbytes, dtype=np.int32)
    return m, n, len_indexes, len_amplitudes, samplerate
    
    
def write_file(fname, bytes):
    with open(fname, "wb") as file:
        file.write(bytes)
        file.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Unexpected number of arguments")
        print("Usage:")
        print("Compression: python3 toco.py c <input>.wav")
        print("Decompression: python3 toco.py d <compressed>.tc")
    mode = sys.argv[1]
    infile = sys.argv[2]
    if mode not in ["c", "d"]:
        print("Invalid mode. Use 'c' for compression and 'd' for decompression")
    if not os.path.isfile(infile):
        print(f"Could not find {infile}")
    if mode == "c":
        compress(infile)
    if mode == "d":
        decompress(infile)