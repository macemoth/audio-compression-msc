# Original code from: https://github.com/tomershay100/mp3-steganography-lib

import numpy as np

import tables
import util
from FrameHeader import *
from FrameSideInformation import FrameSideInformation
# Use below for MP3 probability model
from ArithmeticBitCoder import ArithmeticBitCoder
# Use below for order-0 coding
# from NaiveArithmeticBitCoder import ArithmeticBitCoder
from tsc import compress_tsc, reconstruct_tsc

NUM_PREV_FRAMES = 9
NUM_OF_SAMPLES = 576

SQRT2 = math.sqrt(2)
PI = math.pi


def init_synth_filterbank_block():
    n = np.zeros((64, 32))
    for i in range(64):
        for j in range(32):
            n[i][j] = math.cos((16.0 + i) * (2.0 * j + 1.0) * (PI / 64.0))

    return n


def create_sine_block():
    sine_block = np.zeros((4, 36))

    for i in range(36):
        sine_block[0][i] = math.sin(math.pi / 36.0 * (i + 0.5))
    for i in range(18):
        sine_block[1][i] = math.sin(math.pi / 36.0 * (i + 0.5))
    for i in range(18, 24):
        sine_block[1][i] = 1.0
    for i in range(24, 30):
        sine_block[1][i] = math.sin(math.pi / 12.0 * (i - 18.0 + 0.5))
    for i in range(30, 36):
        sine_block[1][i] = 1.0
    for i in range(12):
        sine_block[2][i] = math.sin(math.pi / 12.0 * (i + 0.5))
    for i in range(6):
        sine_block[3][i] = 0.0
    for i in range(6, 12):
        sine_block[3][i] = math.sin(math.pi / 12.0 * (i - 6.0 + 0.5))
    for i in range(12, 18):
        sine_block[3][i] = 1.0
    for i in range(18, 36):
        sine_block[3][i] = math.sin(math.pi / 36.0 * (i + 0.5))

    return sine_block


class Frame:
    def __init__(self):
        # Declarations
        self.__buffer: list = []
        self.__prev_frame_size: np.ndarray = np.zeros(NUM_PREV_FRAMES)
        self.__frame_size: int = 0
        self.__side_info: FrameSideInformation = FrameSideInformation()
        self.__header: FrameHeader = FrameHeader()
        self.__prev_samples: np.ndarray = np.zeros((2, 32, 18))
        self.__fifo: np.ndarray = np.zeros((2, 1024))

        self.__samples: np.ndarray = np.zeros((2, 2, NUM_OF_SAMPLES))
        self.__export_samples: np.ndarray = np.zeros((2, 2, NUM_OF_SAMPLES), dtype=np.int16)
        self.__nonzero_samples_size = np.zeros((2, 2), dtype=int) # for gr and ch
        self.__sine_block = create_sine_block()
        self.__synth_filterbank_block = init_synth_filterbank_block()
        self.__encoder = ArithmeticBitCoder()
        self.enc_sizes = []

    def init_frame_params(self, buffer, file_data, curr_offset):
        self.__buffer = buffer
        self.set_frame_size()

        starting_side_info_idx = 6 if self.__header.crc == 0 else 4
        self.__side_info.set_side_info(self.__buffer[starting_side_info_idx:], self.__header)
        self.__header_bytes = bytes(self.__buffer[:starting_side_info_idx])
        self.__sideinfo_bytes = bytes(self.__buffer[starting_side_info_idx:starting_side_info_idx+self.__side_info.offset])
        self.__set_main_data(file_data, curr_offset)
        self.__export_samples = self.__samples.copy()
        # for gr in range(2):
        #     for ch in range(self.__header.channels):
        #         # self.__requantize(gr, ch)
        #         pass
        #
        #     if self.__header.channel_mode == ChannelMode.JointStereo and self.__header.mode_extension[0]:
        #         # self.__ms_stereo(gr)
        #         pass
        #
        #     for ch in range(self.__header.channels):
        #         if self.__side_info.block_type[gr][ch] == 2 or self.__side_info.mixed_block_flag[gr][ch]:
        #             self.__reorder(gr, ch)
        #         else:
        #             #self.__alias_reduction(gr, ch)
        #             pass

                # self.__imdct(gr, ch)
                # self.__frequency_inversion(gr, ch)
                # self.__synth_filterbank(gr, ch)

    # Determine the frame size.
    def set_frame_size(self):
        samples_per_frame = 0

        if self.__header.layer == 3:
            if self.__header.mpeg_version == 1:
                samples_per_frame = 1152
            else:
                samples_per_frame = 576

        elif self.__header.layer == 2:
            samples_per_frame = 1152

        elif self.__header.layer == 1:
            samples_per_frame = 384

        # Minimum frame size = 1152 / 8 * 32000 / 48000 = 96
        # Minimum main_data size = 96 - 36 - 2 = 58
        # Maximum main_data_begin = 2^9 = 512
        # Therefore remember ceil(512 / 58) = 9 previous frames.
        for i in range(NUM_PREV_FRAMES - 1, 0, -1):
            self.prev_frame_size[i] = self.prev_frame_size[i - 1]
        self.prev_frame_size[0] = self.frame_size

        self.frame_size = int(((samples_per_frame / 8) * self.__header.bit_rate) / self.__header.sampling_rate)
        if self.__header.padding == 1:
            self.frame_size += 1

    # Due to the Huffman bits' varying length the main_data isn't aligned with the frames.
    # Unpacks the scaling factors and quantized samples.
    def __set_main_data(self, file_data: list, curr_offset: int):
        # Side information is 17 or 32 byte, plus 4 bytes from header an optional 2 bytes from CRC
        constant = 21 if self.__header.channel_mode == ChannelMode.Mono else 36
        if self.__header.crc == 0:
            constant += 2

        # We'll put the main data in its own buffer. Main data may be larger than the previous frame and doesn't
        # Include the size of side info and headers
        if self.__side_info.main_data_begin == 0:
            self.__main_data = self.__buffer[constant: self.frame_size]
        else:
            bound = 0
            for frame in range(NUM_PREV_FRAMES):
                bound += self.prev_frame_size[frame] - constant
                if self.__side_info.main_data_begin < bound:
                    ptr_offset = self.__side_info.main_data_begin + frame * constant

                    part = np.zeros(NUM_PREV_FRAMES)
                    part[frame] = self.__side_info.main_data_begin
                    for i in range(frame):
                        part[i] = self.prev_frame_size[i] - constant
                        part[frame] -= part[i]

                    loc = int(curr_offset - ptr_offset)
                    self.__main_data = file_data[loc: loc + int(part[frame])]
                    ptr_offset -= (part[frame] + constant)
                    for i in range(frame - 1, -1, -1):
                        loc = int(curr_offset - ptr_offset)
                        self.__main_data.extend(file_data[loc: loc + int(part[i])])
                        ptr_offset -= (part[i] + constant)
                    self.__main_data.extend((self.__buffer[constant: self.frame_size]))
                    break
        bit = 0
        for gr in range(2):
            for ch in range(self.__header.channels):
                max_bit = int(bit + self.__side_info.part2_3_length[gr][ch])
                bit = self.__unpack_scalefac(gr, ch, bit)
                self.__unpack_samples(gr, ch, bit, max_bit)
                bit = max_bit

    # Unpack the scale factor indices from the main data. slen1 and slen2 are the size (in bits) of each scaling factor.
    # There are 21 scaling factors for long windows and 12 for each short window.
    def __unpack_scalefac(self, gr: int, ch: int, bit: int):
        sfb = 0
        window = 0
        scalefactor_length = [slen[int(self.__side_info.scalefac_compress[gr][ch])][0],
                              slen[int(self.__side_info.scalefac_compress[gr][ch])][1]]

        # No scale factor transmission for short blocks.
        if self.__side_info.block_type[gr][ch] == 2 and self.__side_info.window_switching[gr][ch]:
            if self.__side_info.mixed_block_flag[gr][ch] == 1:  # Mixed blocks.
                for sfb in range(8):
                    self.__side_info.scalefac_l[gr][ch][sfb] = util.get_bits(self.__main_data, bit,
                                                                             scalefactor_length[0])
                    bit += scalefactor_length[0]

                for sfb in range(3, 6):
                    for window in range(3):
                        self.__side_info.scalefac_s[gr][ch][window][sfb] = util.get_bits(self.__main_data, bit,
                                                                                         scalefactor_length[0])
                        bit += scalefactor_length[0]
            else:  # Short blocks.
                for sfb in range(6):
                    for window in range(3):
                        self.__side_info.scalefac_s[gr][ch][window][sfb] = util.get_bits(self.__main_data, bit,
                                                                                         scalefactor_length[0])
                        bit += scalefactor_length[0]

            for sfb in range(6, 12):
                for window in range(3):
                    self.__side_info.scalefac_s[gr][ch][window][sfb] = util.get_bits(self.__main_data, bit,
                                                                                     scalefactor_length[1])
                    bit += scalefactor_length[1]

            for window in range(3):
                self.__side_info.scalefac_s[gr][ch][window][12] = 0

        # Scale factors for long blocks.
        else:
            if gr == 0:
                for sfb in range(11):
                    self.__side_info.scalefac_l[gr][ch][sfb] = util.get_bits(self.__main_data, bit,
                                                                             scalefactor_length[0])
                    bit += scalefactor_length[0]
                for sfb in range(11, 21):
                    self.__side_info.scalefac_l[gr][ch][sfb] = util.get_bits(self.__main_data, bit,
                                                                             scalefactor_length[1])
                    bit += scalefactor_length[1]
            else:  # Scale factors might be reused in the second granule.
                SB = [6, 11, 16, 21]
                PREV_SB = [0, 6, 11, 16]
                for i in range(2):
                    for sfb in range(PREV_SB[i], SB[i]):
                        if self.__side_info.scfsi[ch][i]:
                            self.__side_info.scalefac_l[gr][ch][sfb] = self.__side_info.scalefac_l[0][ch][sfb]
                        else:
                            self.__side_info.scalefac_l[gr][ch][sfb] = util.get_bits(self.__main_data, bit,
                                                                                     scalefactor_length[0])
                            bit += scalefactor_length[0]
                for i in range(2, 4):
                    for sfb in range(PREV_SB[i], SB[i]):
                        if self.__side_info.scfsi[ch][i]:
                            self.__side_info.scalefac_l[gr][ch][sfb] = self.__side_info.scalefac_l[0][ch][sfb]
                        else:
                            self.__side_info.scalefac_l[gr][ch][sfb] = util.get_bits(self.__main_data, bit,
                                                                                     scalefactor_length[1])
                            bit += scalefactor_length[1]

            self.__side_info.scalefac_l[gr][ch][21] = 0

        return bit

    def __unpack_samples(self, gr, ch, bit, max_bit):
        for i in range(NUM_OF_SAMPLES):
            self.__samples[gr][ch][i] = 0

        # Get big value region boundaries.
        if self.side_info.window_switching[gr][ch] and self.side_info.block_type[gr][ch] == 2:
            region0 = 36
            region1 = 576
        else:
            region0 = self.__header.band_index.long_win[int(self.side_info.region0_count[gr][ch]) + 1]
            region1 = self.__header.band_index.long_win[int(self.side_info.region0_count[gr][ch]) + 1 +
                                                        int(self.side_info.region1_count[gr][ch]) + 1]

        # Get the samples in the big value region. Each entry in the Huffman tables yields two samples.
        sample = 0
        while sample < self.side_info.big_value[gr][ch] * 2:
            if sample < region0:
                table_num = int(self.side_info.table_select[gr][ch][0])
                table = big_value_table[table_num]
            elif sample < region1:
                table_num = int(self.side_info.table_select[gr][ch][1])
                table = big_value_table[table_num]
            else:
                table_num = int(self.side_info.table_select[gr][ch][2])
                table = big_value_table[table_num]

            if table_num == 0:
                self.__samples[gr][ch][sample] = 0
                sample += 2
                continue

            repeat = True
            bit_sample = util.get_bits(self.__main_data, bit, 32)

            # Cycle through the Huffman table and find a matching bit pattern.
            row = 0
            while row < big_value_max[table_num] and repeat:
                for col in range(big_value_max[table_num]):
                    i = 2 * big_value_max[table_num] * row + 2 * col
                    value = table[i]
                    size = table[i + 1]
                    if value >> (32 - size) == bit_sample >> (32 - size):
                        bit += size
                        values = (row, col)
                        for i in range(2):

                            # linbits extend the sample's size if needed.
                            linbit = 0
                            if big_value_linbit[table_num] != 0 and values[i] == big_value_max[table_num] - 1:
                                linbit = util.get_bits(self.__main_data, bit, big_value_linbit[table_num])
                                bit += big_value_linbit[table_num]

                            # If the sample is negative or positive.
                            sign = 1
                            if values[i] > 0:
                                sign = -1 if util.get_bits(self.__main_data, bit, 1) > 0 else 1
                                bit += 1

                            self.__samples[gr][ch][sample + i] = float(sign * (values[i] + linbit))

                        repeat = False
                        break
                row += 1
            sample += 2

        # Quadruples region.
        while bit < max_bit and sample + 4 < 576:
            values = [0, 0, 0, 0]

            # Flip bits.
            if self.side_info.count1table_select[gr][ch] == 1:
                bit_sample = util.get_bits(self.__main_data, bit, 4)
                bit += 4
                values[0] = 0 if (bit_sample & 0x08) > 0 else 1
                values[1] = 0 if (bit_sample & 0x04) > 0 else 1
                values[2] = 0 if (bit_sample & 0x02) > 0 else 1
                values[3] = 0 if (bit_sample & 0x01) > 0 else 1
            else:
                bit_sample = util.get_bits(self.__main_data, bit, 32)
                for entry in range(16):
                    value = quad_table_1.hcod[entry]
                    size = quad_table_1.hlen[entry]

                    if value >> (32 - size) == bit_sample >> (32 - size):
                        bit += size
                        for i in range(4):
                            values[i] = int(quad_table_1.value[entry][i])
                        break

            # Get the sign bit.
            for i in range(4):
                if values[i] > 0:
                    if util.get_bits(self.__main_data, bit, 1) == 1:
                        values[i] = -values[i]
                    bit += 1

            for i in range(4):
                self.__samples[gr][ch][sample + i] = values[i]

            sample += 4

        # Fill remaining samples with zero.
        self.__nonzero_samples_size[gr][ch] = sample
        while sample < 576:
            self.__samples[gr][ch][sample] = 0
            sample += 1

    # The reduced samples are rescaled to their original scales and precisions.
    def __requantize(self, gr: int, ch: int):
        exp1, exp2 = 0.0, 0.0
        window = 0
        sfb = 0
        SCALEFAC_MULT = 0.5 if self.__side_info.scalefac_scale[gr][ch] == 0 else 1

        sample = 0
        i = 0
        while sample < NUM_OF_SAMPLES:
            if self.__side_info.block_type[gr][ch] == 2 or (self.__side_info.mixed_block_flag[gr][ch] and sfb >= 8):
                short_win_val = self.__header.band_width.short_win[sfb] if sfb < len(self.__header.band_width.short_win
                                                                                     ) else 0
                if i == short_win_val:
                    i = 0
                    if window == 2:
                        window = 0
                        sfb += 1
                    else:
                        window += 1

                exp1 = self.__side_info.global_gain[gr][ch] - 210.0 - 8.0 * self.__side_info.subblock_gain[gr][ch][
                    window]
                exp2 = SCALEFAC_MULT * self.__side_info.scalefac_s[gr][ch][window][sfb]
            else:
                if sample == self.__header.band_index.long_win[sfb + 1]:
                    # Don't increment sfb at the zeroth sample.
                    sfb += 1

                exp1 = self.__side_info.global_gain[gr][ch] - 210.0

                pretab_val = tables.pretab[sfb] if sfb < len(tables.pretab) else 0
                exp2 = SCALEFAC_MULT * (
                        self.__side_info.scalefac_l[gr][ch][sfb] + self.__side_info.preflag[gr][ch] * pretab_val)

            sign = -1.0 if self.__samples[gr][ch][sample] < 0 else 1.0
            a = pow(abs(self.__samples[gr][ch][sample]), 4.0 / 3.0)
            b = pow(2.0, exp1 / 4.0)
            c = pow(2.0, -exp2)

            self.__samples[gr][ch][sample] = sign * a * b * c

            sample += 1
            i += 1

    #  The left and right channels are added together to form the middle channel. The
    #  difference between each channel is stored in the side channel.
    def __ms_stereo(self, gr: int):
        for sample in range(NUM_OF_SAMPLES):
            middle = self.__samples[gr][0][sample]
            side = self.__samples[gr][1][sample]
            self.__samples[gr][0][sample] = (middle + side) / SQRT2
            self.__samples[gr][1][sample] = (middle - side) / SQRT2

    # Reorder short blocks, mapping from scalefactor subbands (for short windows) to 18 sample blocks.
    def __reorder(self, gr: int, ch: int):
        total = 0
        start = 0
        block = 0
        samples = np.zeros(NUM_OF_SAMPLES)

        for sb in range(12):
            sb_width = self.__header.band_width.short_win[sb]
            for ss in range(sb_width):
                samples[start + block + 0] = self.__samples[gr][ch][total + ss + sb_width * 0]
                samples[start + block + 6] = self.__samples[gr][ch][total + ss + sb_width * 1]
                samples[start + block + 12] = self.__samples[gr][ch][total + ss + sb_width * 2]

                if block != 0 and block % 5 == 0:
                    start += 18
                    block = 0
                else:
                    block += 1

            total += sb_width * 3

        for i in range(NUM_OF_SAMPLES):
            self.__samples[gr][ch][i] = samples[i]

    def __alias_reduction(self, gr: int, ch: int):
        cs = [.8574929257, .8817419973, .9496286491, .9833145925, .9955178161, .9991605582, .9998991952, .9999931551]
        ca = [-.5144957554, -.4717319686, -.3133774542, -.1819131996, -.0945741925, -.0409655829, -.0141985686,
              -.0036999747]

        sb_max = 2 if self.__side_info.mixed_block_flag[gr][ch] else 32

        for sb in range(1, sb_max):
            for sample in range(8):
                offset1 = 18 * sb - sample - 1
                offset2 = 18 * sb + sample
                s1 = self.__samples[gr][ch][offset1]
                s2 = self.__samples[gr][ch][offset2]
                self.__samples[gr][ch][offset1] = s1 * cs[sample] - s2 * ca[sample]
                self.__samples[gr][ch][offset2] = s2 * cs[sample] + s1 * ca[sample]

    # Inverted modified discrete cosine transformations (IMDCT) are applied to each sample and are afterwards windowed
    # to fit their window shape. As an addition, the samples are overlapped.
    def __imdct(self, gr: int, ch: int):
        sample_block = np.zeros(36)
        n = 12 if self.side_info.block_type[gr][ch] == 2 else 36
        half_n = int(n / 2)
        sample = 0

        for block in range(32):
            for win in range(3 if self.side_info.block_type[gr][ch] == 2 else 1):
                for i in range(n):
                    xi = 0.0
                    for k in range(half_n):
                        s = self.__samples[gr][ch][18 * block + half_n * win + k]
                        xi += s * math.cos(math.pi / (2 * n) * (2 * i + 1 + half_n) * (2 * k + 1))

                    # Windowing samples.
                    sample_block[win * n + i] = xi * self.__sine_block[int(self.side_info.block_type[gr][ch])][i]

            if self.side_info.block_type[gr][ch] == 2:
                temp_block = np.copy(sample_block)
                for i in range(6):
                    sample_block[i] = 0
                for i in range(6, 12):
                    sample_block[i] = temp_block[0 + i - 6]
                for i in range(12, 18):
                    sample_block[i] = temp_block[0 + i - 6] + temp_block[12 + i - 12]
                for i in range(18, 24):
                    sample_block[i] = temp_block[12 + i - 12] + temp_block[24 + i - 18]
                for i in range(24, 30):
                    sample_block[i] = temp_block[24 + i - 18]
                for i in range(30, 36):
                    sample_block[i] = 0

            # Overlap.
            for i in range(18):
                self.__samples[gr][ch][sample + i] = sample_block[i] + self.__prev_samples[ch][block][i]
                self.__prev_samples[ch][block][i] = sample_block[18 + i]
            sample += 18

    def __frequency_inversion(self, gr: int, ch: int):
        for sb in range(1, 18, 2):
            for i in range(1, 32, 2):
                self.__samples[gr][ch][i * 18 + sb] *= -1

    def __synth_filterbank(self, gr: int, ch: int):

        s, u, w = np.zeros(32), np.zeros(512), np.zeros(512)
        pcm = np.zeros(576)

        for sb in range(18):
            for i in range(32):
                s[i] = self.__samples[gr][ch][i * 18 + sb]

            for i in range(1023, 63, -1):
                self.__fifo[ch][i] = self.__fifo[ch][i - 64]

            for i in range(64):
                self.__fifo[ch][i] = 0.0
                for j in range(32):
                    self.__fifo[ch][i] += s[j] * self.__synth_filterbank_block[i][j]

            for i in range(8):
                for j in range(32):
                    u[i * 64 + j] = self.__fifo[ch][i * 128 + j]
                    u[i * 64 + j + 32] = self.__fifo[ch][i * 128 + j + 96]

            for i in range(512):
                w[i] = u[i] * synth_window[i]

            for i in range(32):
                sum = 0
                for j in range(16):
                    sum += w[j * 32 + i]
                pcm[32 * sb + i] = sum

        self.__samples[gr][ch] = pcm

    def __get_tsc_bytes(self):
        index = np.array(range(0, NUM_OF_SAMPLES))
        tsc_data = bytearray()
        for gr in range(2):
            for ch in range(self.__header.channels):
                tsc = compress_tsc(np.column_stack([index, self.__samples[gr, ch, 0:NUM_OF_SAMPLES]]), num_indices_to_keep="all")
                bitmap = np.zeros(NUM_OF_SAMPLES, dtype=bool)
                bitmap[tsc[:, 0].astype(int) - 0] = True
                tsc_bytes = tsc[:, 1].astype(np.int16).tobytes()
                # remainder_bytes = self.__samples[cutoff:].astype(np.int8).tobytes()
                bitmap_bytes = np.packbits(bitmap).tobytes()
                tsc_data.extend(tsc_bytes)
        return tsc_data

    def __get_bwt_samples(self):
        pass

    @property
    def frame_size(self):
        return self.__frame_size

    @frame_size.setter
    def frame_size(self, frame_size):
        self.__frame_size = frame_size

    @property
    def prev_frame_size(self):
        return self.__prev_frame_size

    @prev_frame_size.setter
    def prev_frame_size(self, prev_frame_size):
        self.__prev_frame_size = prev_frame_size

    @property
    def side_info(self):
        return self.__side_info

    @property
    def sampling_rate(self):
        return self.__header.sampling_rate

    def init_header_params(self, buffer):
        self.__header.init_header_params(buffer)

    def get_samples(self):
        return self.__export_samples.astype(np.int16)

    def interleave(self):
        pcm = np.zeros((2 * NUM_OF_SAMPLES, self.__header.channels))
        for gr in range(2):
            for sample in range(576):
                for ch in range(self.__header.channels):
                    pcm[sample + NUM_OF_SAMPLES * gr][ch] = self.__samples[gr][ch][sample]
        return pcm

    def get_3pm_bytes(self):
        """
        Provide alternative methods for encoded MP3 (i.e. 3PM) frames
        :return: Frame bytes
        """
        # Reset ArithmeticBitCoder if using naive probability model
        # self.__encoder = ArithmeticBitCoder()

        # Use following line to export uncoded bytes or to compress it as whole (TSC, order-0 arithmetic coding)
        # new_main_data = bytes((self.__samples.flatten()).astype(np.int16))

        # Use following four lines with the MP3 probability model
        encoded = bytearray()
        for gr in range(2):
           for ch in range(self.__header.channels):
               encoded.extend(self.__encoder.encode(self.__samples[gr, ch, :].astype(np.int16).tobytes()))

        # Use following line for the naive (order-0) probability model
        # new_main_data = self.__export_samples.flatten().astype(np.int16).tobytes()

        # Use following line to enable TSC compression
        # new_main_data = self.__get_tsc_bytes()

        # If new_main_data is set, encode it
        # encoded = self.__encoder.encode(new_main_data)

        # Diagnosis code
        # for b in new_main_data:
        #     self.__byte_freq[b] += 1
        self.enc_sizes.append(len(encoded))
        if len(self.enc_sizes) % 10 == 0:
#            print(f"Uncompressed: {len(new_main_data)}")
            print(f"Mean length: {np.median(self.enc_sizes)} std: {np.std(self.enc_sizes)}")

        new_frame: bytearray = self.__header_bytes + self.__sideinfo_bytes + np.int16(len(encoded)).tobytes() + encoded
        return new_frame