import sys
import os
import numpy as np
from tsc import compress_tsc, reconstruct_tsc
import soundfile as sf
import librosa
import librosa.core.spectrum
from utils.ArithmeticBitCoder import ArithmeticBitCoder
from sklearn.cluster import KMeans

N_FFT = 1024 # number of samples for every STFT frame

def compress(infile, compression_factor=4):
    # Read file
    fname = infile.split("/")[-1].split(".")[0]
    _, samplerate = sf.read(infile)
    original, fs = librosa.core.load(infile, sr=samplerate, mono=False)
    
    n_channels = 1
    if(len(original.shape)) == 2:
        n_channels = original.shape[0]
    
    if (n_channels > 2):
        print("Unsupported number of channels. Maximum channels: 2")
        return
    
    print("Apply STFT transform")
    # Apply STFT
    if n_channels == 2:
        spectrogram = [np.abs(librosa.core.stft(original[0], n_fft=N_FFT)), np.abs(librosa.core.stft(original[1], n_fft=N_FFT))]
        spectrogram = (10 * np.sqrt(spectrogram, out=np.zeros_like(spectrogram), where=(spectrogram != 0))).round()
        con_spectrogram = np.column_stack([spectrogram[0], spectrogram[1]])
        _, m, n = spectrogram.shape
    else:
        spectrogram = np.abs(librosa.core.stft(original, n_fft=N_FFT))
        spectrogram = (10 * np.sqrt(spectrogram, out=np.zeros_like(spectrogram), where=(spectrogram != 0))).round()
        con_spectrogram = spectrogram
        m, n = spectrogram.shape

    
    # Apply TSC
    print("Starting vector quantisation")
    kmeans = KMeans(n_clusters=n//compression_factor, random_state=0, n_init="auto")
    res = kmeans.fit(con_spectrogram.T)
    n_vecs, n_frequencies = res.cluster_centers_.shape
    print(f"Obtained {n_vecs} vectors")

    # Byte conversion
    vecbytes = res.cluster_centers_.flatten().round().astype(np.int8).tobytes()
    labelbytes = res.labels_.astype(np.int32).tobytes()

    # Entropic coding
    print("Starting entropic coding")
    ae = ArithmeticBitCoder()
    cvecs = ae.encode(vecbytes)
    ae.__init__() # Reset arithmetic coder
    clabels = ae.encode(labelbytes)
    
    # Pack into file
    header = make_header(n_vecs, n_frequencies, len(cvecs), len(clabels), samplerate, 1 if n_channels == 2 else 0)
    write_file(f"{fname}.vc", header + cvecs + clabels)
    
    print(f"Compression ratio: 1:{os.path.getsize(infile) / (len(header) + len(cvecs) + len(clabels))}")

def make_header(n_vecs, n_frequencies, len_vecs, len_labels, samplerate, stereo):
    # Header format
    # | n_vecs (4 bytes)  | n_frequencies (4 bytes) | # len_vecs (4 bytes) |
    # | len_labels (4 bytes)  | samplerate (4 bytes) | stereo (4 bytes) |
    # Total length: 24 bytes
    return bytes(np.array([n_vecs, n_frequencies, len_vecs, len_labels, samplerate, stereo], dtype=np.int32))

def decompress(infile):
    fname = infile.split("/")[-1].split(".")[0]
    # Unpack binary file
    len_header = 24
    offset = 0
    print("Unpacking compressed file")
    with open(infile, "rb") as file:
        fbytes = file.read()
        file.close()
    n_vecs, n_frequencies, len_vecs, len_labels, samplerate, stereo = read_header(fbytes[0:len_header])
    if (len_header + len_vecs + len_labels) != len(fbytes):
        print(
            f"File {infile} has length {len(fbytes)}. Expected length from header: {len_header + len_vecs + len_labels}")
        return
    offset += len_header

    print("Starting entropic decoding")
    # Undo entropic coding
    ae = ArithmeticBitCoder()
    vecbytes = ae.decode(bytearray(fbytes[offset:offset + len_vecs]))
    offset += len_vecs
    ae.__init__()
    labelbytes = ae.decode(bytearray(fbytes[offset:offset + len_labels]))
    
    # Undo serialisation
    vectors = np.frombuffer(vecbytes, dtype=np.int8)
    labels = np.frombuffer(labelbytes, dtype=np.int32)
    vectors = vectors.reshape([n_vecs, n_frequencies])

    print("Starting spectrogram reconstruction")
    # Reconstruct spectrogram
    reconstructed = vectors[labels].T
    _, n_frames = reconstructed.shape
    if stereo:
        spectrogram = np.array([reconstructed[:, 0:n_frames // 2], reconstructed[:, n_frames//2:]])
    else:
        spectrogram = reconstructed
    spectrogram = np.square(spectrogram/10) # dequantise

    print("Writing waveform signal")
    # Reconstruct waveform
    if stereo:
        reconstructed_waveform = np.vstack((librosa.core.spectrum.griffinlim(spectrogram[0]), librosa.core.spectrum.griffinlim(spectrogram[1]))) # Undo STFT
    else:
        reconstructed_waveform = librosa.core.spectrum.griffinlim(spectrogram)
    sf.write(f"{fname}.wav", reconstructed_waveform.T, samplerate)
    
def read_header(headerbytes):
    n_vecs, n_frequencies, len_vecs, len_labels, samplerate, stereo = np.frombuffer(headerbytes, dtype=np.int32)
    return  n_vecs, n_frequencies, len_vecs, len_labels, samplerate, stereo
    
def write_file(fname, bytes):
    with open(fname, "wb") as file:
        file.write(bytes)
        file.close()

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Unexpected number of arguments")
        print("Usage:")
        print("Compression: python3 veco.py c <input>.wav")
        print("Decompression: python3 veco.py d <compressed>.vc")
    mode = sys.argv[1]

    if mode not in ["c", "d"]:
        print("Invalid mode. Use 'c' for compression and 'd' for decompression")
    if mode == "c":
        compression_factor = int(sys.argv[2])
        if (1 <= compression_factor <= 20):
            infile = sys.argv[3]
            if os.path.isfile(infile):
                compress(infile, compression_factor)
            else:
                print(f"Could not find {infile}")
        else:
            print("Invalid compression factor. Choose from 1 to 20")
    if mode == "d":
        infile = sys.argv[2]
        decompress(infile)