# Original code from: https://github.com/tomershay100/mp3-steganography-lib

from tqdm import tqdm
from Frame import *
from scipy.io.wavfile import write

HEADER_SIZE = 4


class Recompressor:

    def __init__(self, file_data, offset, file_path):
        # Declarations
        # self.__curr_header: FrameHeader = FrameHeader()
        self.__curr_frame: Frame = Frame()
        self.__valid: bool = False
        # List of integers that contain the file (without ID3) data
        self.__file_data: list = []
        self.__buffer: list = []
        self.__file_length: int = 0
        self.__file_path = file_path
        self.__new_file_path = self.__file_path[:-4] + '.3pm'
        self.__allsamples = []
        self.__pcm = []

        self.__bytes = bytearray()

        # cut the id3 from hex_data
        self.__buffer = file_data[offset:]

        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__valid = True
            self.__file_data = file_data
            self.__file_length = len(file_data)
            self.__offset = offset
            self.__init_curr_header()
            self.__curr_frame.set_frame_size()
        else:
            self.__valid = False

    def __init_curr_header(self):
        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__curr_frame.init_header_params(self.__buffer)
        else:
            self.__valid = False

    def __init_curr_frame(self):
        self.__curr_frame.init_frame_params(self.__buffer, self.__file_data, self.__offset)

    def parse_file(self):
        num_of_parsed_frames = 0

        pbar = tqdm(total=self.__file_length + 1 - HEADER_SIZE, desc='decoding')
        while self.__valid and self.__file_length > self.__offset + HEADER_SIZE:
            self.__init_curr_header()
            if self.__valid:
                self.__init_curr_frame()
                num_of_parsed_frames += 1
                self.__offset += self.__curr_frame.frame_size
                self.__buffer = self.__file_data[self.__offset:]
                # print(f'Parsed: {num_of_parsed_frames}')

            b = self.__curr_frame.get_3pm_bytes()
            self.__bytes.extend(b)
            self.__allsamples.append(self.__curr_frame.get_samples())
            self.__pcm.extend(list(self.__curr_frame.interleave()))
            pbar.update(self.__curr_frame.frame_size)
        
        pbar.close()

        return num_of_parsed_frames

    def export_samples(self, filename):
        print(f"Writing {len(self.__allsamples)} to {filename}")
        np.save(filename, self.__allsamples)

    def write_to_3pm(self):
        with open(self.__new_file_path, "wb") as file:
            file.write(self.__bytes)
            file.close()

    def write_to_wav(self):
        # Convert PCM to WAV (from 32-bit floating-point to 16-bit PCM by mult by 32767)
        pcm_data = np.array(self.__pcm)
        write(self.__new_file_path, self.__curr_frame.sampling_rate, (pcm_data * 32767).astype(np.int16))