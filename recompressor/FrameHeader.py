# Original code from: https://github.com/tomershay100/mp3-steganography-lib

from enum import Enum
from tables import *
import math

MAX_BYTE_VALUE = 256


class Band:
    def __init__(self):
        self.long_win: list = []
        self.short_win: list = []


class ChannelMode(Enum):
    Stereo = 0
    JointStereo = 1
    DualChannel = 2
    Mono = 3


class Emphasis(Enum):
    NONE = 0
    MS5015 = 1
    Reserved = 2
    CCITJ17 = 3


class FrameHeader:
    def __init__(self):
        # Declarations
        self.__buffer: list = []
        self.__mpeg_version: float = 0.0
        self.__layer: int = 0
        self.__crc: bool = False
        self.__bit_rate: int = 0
        self.__sampling_rate: int = 0
        self.__padding: bool = False
        self.__channel_mode: ChannelMode = ChannelMode(0)
        self.__channels: int = 0
        self.__mode_extension: list = [0, 0]
        self.__emphasis: Emphasis = Emphasis(0)
        self.__info: list = [False, False, False]
        self.__band_index: Band = Band()
        self.__band_width: Band = Band()

    # Unpack the MP3 header.
    # @param buffer A pointer that points to the first byte of the frame header.
    def init_header_params(self, buffer):
        self.__buffer = buffer
        self.__set_mpeg_version()
        self.__set_layer(self.__buffer[1])
        self.__set_crc()
        self.__set_info()
        self.__set_emphasis()
        self.__set_sampling_rate()
        self.__set_tables()
        self.__set_channel_mode()
        self.__set_mode_extension()
        self.__set_padding()
        self.__set_bit_rate()

    # Determine the MPEG version.
    def __set_mpeg_version(self):
        if (self.__buffer[1] & 0x10) == 0x10 and (self.__buffer[1] & 0x08) == 0x08:
            self.__mpeg_version = 1
        elif (self.__buffer[1] & 0x10) == 0x10 and (self.__buffer[1] & 0x08) != 0x08:
            self.__mpeg_version = 2
        elif (self.__buffer[1] & 0x10) != 0x10 and (self.__buffer[1] & 0x08) == 0x08:
            self.__mpeg_version = 0
        elif (self.__buffer[1] & 0x10) != 0x10 and (self.__buffer[1] & 0x08) != 0x08:
            self.__mpeg_version = 2.5

    # Determine layer
    def __set_layer(self, byte):
        byte = (byte << 5) % MAX_BYTE_VALUE
        byte = (byte >> 6) % MAX_BYTE_VALUE
        self.__layer = 4 - byte

    # Cyclic redundancy check. If set, two bytes after the header information are used up by the CRC.
    def __set_crc(self):
        self.__crc = self.__buffer[1] & 0x01

    # Additional information (not important)
    def __set_info(self):
        self.__info = [bool(self.__buffer[2] & 0x01), bool(self.__buffer[3] & 0x08), bool(self.__buffer[3] & 0x04)]

    # Although rarely used, there is no method for emphasis.
    def __set_emphasis(self):
        value = (((self.__buffer[3] << 6)) % MAX_BYTE_VALUE >> 6) % MAX_BYTE_VALUE
        self.__emphasis = Emphasis(value)

    def __set_sampling_rate(self):
        rates = [[44100, 48000, 32000], [22050, 24000, 16000], [11025, 12000, 8000]]
        ceil_mpeg_version = int(math.floor(self.__mpeg_version))
        if (self.__buffer[2] & 0x08) != 0x08 and (self.__buffer[2] & 0x04) != 0x04:
            self.__sampling_rate = rates[ceil_mpeg_version - 1][0]
        elif (self.__buffer[2] & 0x08) != 0x08 and (self.__buffer[2] & 0x04) == 0x04:
            self.__sampling_rate = rates[ceil_mpeg_version - 1][1]
        elif (self.__buffer[2] & 0x08) == 0x08 and (self.__buffer[2] & 0x04) != 0x04:
            self.__sampling_rate = rates[ceil_mpeg_version - 1][2]

    # During the decoding process different tables are used depending on the sampling rate.
    def __set_tables(self):
        if self.__sampling_rate == 32000:
            self.__band_index.short_win = band_index_table.short_32
            self.__band_width.short_win = band_width_table.short_32
            self.__band_index.long_win = band_index_table.long_32
            self.__band_width.long_win = band_width_table.long_32
        elif self.__sampling_rate == 44100:
            self.__band_index.short_win = band_index_table.short_44
            self.__band_width.short_win = band_width_table.short_44
            self.__band_index.long_win = band_index_table.long_44
            self.__band_width.long_win = band_width_table.long_44
        elif self.__sampling_rate == 48000:
            self.__band_index.short_win = band_index_table.short_48
            self.__band_width.short_win = band_width_table.short_48
            self.__band_index.long_win = band_index_table.long_48
            self.__band_width.long_win = band_width_table.long_48

    # 0 -> Stereo
    # 1 -> Joint stereo (this option requires use of mode_extension)
    # 2 -> Dual channel
    # 3 -> Single channel
    def __set_channel_mode(self):
        value = (self.__buffer[3] >> 6) % MAX_BYTE_VALUE
        self.__channel_mode = ChannelMode(value)
        self.__channels = 1 if self.__channel_mode == ChannelMode.Mono else 2

    # Applies only to joint stereo.
    def __set_mode_extension(self):
        if self.__layer == 3:
            self.__mode_extension = [self.__buffer[3] & 0x20, self.__buffer[3] & 0x10]

    # If set, the frame size is 1 byte larger.
    def __set_padding(self):
        self.__padding = bool(self.__buffer[2] & 0x02)

    # For variable bit rate (VBR) files, this data has to be gathered constantly.
    def __set_bit_rate(self):
        if self.__mpeg_version == 1:
            if self.__layer == 1:
                self.__bit_rate = self.__buffer[2] * 32
            elif self.__layer == 2:
                rates = [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
                self.__bit_rate = rates[(self.__buffer[2] >> 4) % MAX_BYTE_VALUE - 1] * 1000
            elif self.__layer == 3:
                rates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
                self.__bit_rate = rates[(self.__buffer[2] >> 4) % MAX_BYTE_VALUE - 1] * 1000
            else:
                self.__valid = False
        else:
            if self.__layer == 1:
                rates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
                self.__bit_rate = rates[(self.__buffer[2] >> 4) % MAX_BYTE_VALUE - 1] * 1000
            elif self.__layer < 4:
                rates = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
                self.__bit_rate = rates[(self.__buffer[2] >> 4) % MAX_BYTE_VALUE - 1] * 1000
            else:
                self.__valid = False

    @property
    def layer(self):
        return self.__layer

    @property
    def bit_rate(self):
        return self.__bit_rate

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    @property
    def mpeg_version(self):
        return self.__mpeg_version

    @property
    def padding(self):
        return self.__padding

    @property
    def channel_mode(self):
        return self.__channel_mode

    @property
    def channels(self):
        return self.__channels

    @property
    def crc(self):
        return self.__crc

    @property
    def band_index(self):
        return self.__band_index

    @property
    def band_width(self):
        return self.__band_width

    @property
    def mode_extension(self):
        return self.__mode_extension


