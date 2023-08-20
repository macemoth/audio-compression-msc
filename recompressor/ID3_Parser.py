# Original code from: https://github.com/tomershay100/mp3-steganography-lib

from enum import Enum
import util

MPEG_VERSION = 2


class ID3Flags(Enum):

    @property
    def flag(self):
        return self.name

    FooterPresent = 0
    ExperimentalIndicator = 1
    ExtendedHeader = 2
    Unsynchronisation = 3


class ID3FrameFlags(Enum):

    @property
    def flag(self):
        return self.name

    DiscardFrameOnTagAlter = 0
    DiscradFrameOnFileAlter = 1
    ReadOnly = 2
    ZLIBCompression = 3
    FrameEncrypted = 4
    FrameContainsGroupInformation = 5


class ID3Frame:
    def __init__(self, frame_id: list, flags: int, content: bytes):
        self.__frame_id: list = frame_id
        self.__content: bytes = content
        self.__frame_flags: list = []
        self.__set_flags(flags)

    def __set_flags(self, flags: int):
        for bit_num in range(3):
            if flags >> bit_num & 1:
                self.__frame_flags.append(True)
            else:
                self.__frame_flags.append(False)
        for bit_num in range(8, 11):
            if flags >> bit_num & 1:
                self.__frame_flags.append(True)
            else:
                self.__frame_flags.append(False)

    @property
    def id(self):
        chrs = [chr(k) for k in self.__frame_id]
        return ''.join(chrs)

    @property
    def content(self):
        return self.__content.decode('utf-8')

    @property
    def frame_flags(self):
        flags = []
        for i, flag in enumerate(self.__frame_flags):
            if flag:
                flags.append(ID3FrameFlags(i).flag)
        return flags


class ID3:
    # TODO parse extended header
    def __init__(self, buffer: list):
        # Declarations
        self.__buffer: list = buffer
        self.__offset: int
        self.__valid: bool
        self.__start: int
        self.__version: str
        self.__id3_flags: list = [False, False, False, False]
        self.__extended_header_size: int
        self.__id3_frames: list = []

        if chr(buffer[0]) == 'I' and chr(buffer[1]) == 'D' and chr(buffer[2]) == '3':
            self.__set_version(self.__buffer[3], self.__buffer[4])
            if self.__set_flags(self.__buffer[5]):
                self.__valid = True
                self.__set_offset(util.char_to_int(self.__buffer[6:10]))
                self.__set_extended_header_size(util.char_to_int(self.__buffer[10:14]))
                self.__set_frames(10 + self.__extended_header_size)
            else:
                self.__valid = False
        else:
            self.__valid = False

    def __set_version(self, version: int, revision: int):
        self.__version = f'{MPEG_VERSION}.{version}.{revision}'

    def __set_offset(self, offset: int):
        if self.__id3_flags[ID3Flags.FooterPresent.value]:
            self.__offset = offset + 20
        else:
            self.__offset = offset + 10

    def __set_flags(self, flags: int):
        # These flags must be unset for frame to be valid (protected bits)
        for bit_num in range(4):
            if flags >> bit_num & 1:
                return False
        # Check flags
        for bit_num in range(4, 8):
            self.__id3_flags[bit_num - 4] = True if flags >> bit_num & 1 else False

        self.__id3_flags = tuple(self.__id3_flags)
        return True

    def __set_extended_header_size(self, size: int):
        if self.__id3_flags[2]:
            self.__extended_header_size = size
        else:
            self.__extended_header_size = 0

    def __set_frames(self, start):
        footer_size = self.__id3_flags[0] * 10
        size = self.__offset - self.__extended_header_size - footer_size
        i = 0

        valid = True
        while i < size and valid:
            frame_id = self.__buffer[start + i: start + i + 4]
            for c in frame_id:
                if not (chr(c).isupper() or chr(c).isdigit()):  # Check for legal ID
                    valid = False
                    break
            if valid:
                i += 4
                field_size = util.char_to_int(self.__buffer[start + i: start + i + 4])  # 4 Bytes
                i += 4
                frame_flags = util.get_bits(self.__buffer, util.BYTE_LENGTH * (start + i), 16)  # 2 Bytes
                i += 2
                frame_content = bytes(self.__buffer[start + i: start + i + field_size])
                i += field_size
                self.__id3_frames.append(ID3Frame(frame_id, frame_flags, frame_content))

    @property
    def offset(self):
        return self.__offset

    @property
    def is_valid(self):
        return self.__valid

    @property
    def version(self):
        return self.__version

    @property
    def id3_flags(self):
        flags = []
        for i, flag in enumerate(self.__id3_flags):
            if flag:
                flags.append(ID3Flags(i).flag)
        return flags

    @property
    def extended_header_size(self):
        return self.__extended_header_size

    @property
    def id3_frames(self):
        return self.__id3_frames
