# Original code from: https://github.com/tomershay100/mp3-steganography-lib

import sys
import time

from Recompressor import Recompressor
from ID3_Parser import ID3


def parse_metadata(file_name: str, id3_parser: ID3):
    with open('METADATA.txt', 'w') as metadata:
        metadata.write(f'METADATA FOR FILE: {file_name}\n')
        metadata.write('################################\n\n\n')
        metadata.write(f'ID3 Version: {id3_parser.version}\n')
        if len(id3_parser.id3_flags) > 0:
            metadata.write('ID3 Flags:\n')
            for flag in id3_parser.id3_flags:
                metadata.write(f'- {flag}\n')
            metadata.write('\n')

        metadata.write('\nID3 Frames:\n')
        for i, frame in enumerate(id3_parser.id3_frames):
            metadata.write(f'Frame number: {i}\n')
            metadata.write(f'Frame ID: {frame.id}\n')
            metadata.write(f'Content: {frame.content}\n')
            if len(frame.frame_flags) > 0:
                metadata.write('Frame Flags:\n')
                for flag in frame.frame_flags:
                    metadata.write(f'- {flag}\n')
            metadata.write('\n')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Unexpected number of arguments.")
        exit(-1)
    if len(sys.argv) < 2:
        print("No directory specified.")
        exit(-1)
    file_path = sys.argv[1]

    with open(file_path, 'rb') as f:
        hex_data = [c for c in f.read()]
        f.close()

    ################################

    id3_decoder = ID3(hex_data)
    if id3_decoder.is_valid:
        parse_metadata(file_path, id3_decoder)
        offset = id3_decoder.offset

    else:
        offset = 0

    r = Recompressor(hex_data, offset, file_path)
    start = time.time()
    num_of_parsed_frames = r.parse_file()
    parsing_time = time.time() - start
    print('Parsed', num_of_parsed_frames, 'frames in', parsing_time, 'seconds')
    # r.export_samples("allsamples_trance.npy")
    r.write_to_3pm()
