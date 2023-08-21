from arithmetic_model import *


class ArithmeticBinaryDecoder:

    def __init__(self, model="MP3"):
        self.model = model
        self.__freq, self.__cumfreq, self.__ch2idx, self.__idx2ch = get_model_params(model)

        self.__high = MAX_VALUE
        self.__low = 0

        self.__buffer = None
        self.__value = 0


    def update_tables(self, symbol_index: int):
        """
        Dynamic table update
        """
        if self.model: # No updates needed for static model
            return
        if self.__cumfreq[0] == MAX_FREQ:
            cum = 0
            for i in range(N_SYMBOLS, -1, -1):
                self.__freq[i] = (self.__freq[i] + 1) / 2
                self.__cumfreq[i] = cum
                cum += self.__freq[i]
        self.__freq[0] = 0;

        i = symbol_index
        while self.__freq[i] == self.__freq[i - 1]: i -= 1
        if i < symbol_index:
            ch_i = self.__idx2ch[i]  # chars are one lower than indexes
            ch_symbol = self.__idx2ch[symbol_index]
            self.__idx2ch[i] = ch_symbol
            self.__idx2ch[symbol_index] = ch_i
            self.__ch2idx[ch_i] = symbol_index
            self.__ch2idx[ch_symbol] = i
        self.__freq[i] += 1
        while i > 0:
            i -= 1
            self.__cumfreq[i] += 1


    def decode(self, data: bytes):
        self.__buffer = ReadBuffer(data)

        decoded = bytearray()

        for i in range(1, CODE_VALUE + 1):
            self.__value = 2 * self.__value + self.__buffer.read_bit()


        while True:
            d = self.decode_symbol()
            if d == EOF or self.__buffer.end_encoding:
                break
            b = self.__idx2ch[d]
            decoded.append(b)
            self.update_tables(d)
        return decoded

    def decode_symbol(self):
        range = self.__high - self.__low
        cum = int((((self.__value - self.__low) + 1) * self.__cumfreq[0] - 1) // range)

        symbol_index = 1
        while self.__cumfreq[symbol_index] > cum:
            symbol_index += 1

        self.__high = int(self.__low + (range * self.__cumfreq[symbol_index - 1] / self.__cumfreq[0]))
        self.__low = int(self.__low + (range * self.__cumfreq[symbol_index] / self.__cumfreq[0]))

        while True:
            if self.__high < HALF:
                pass
            elif self.__low >= HALF:
                self.__value -= HALF
                self.__low -= HALF
                self.__high -= HALF
            elif self.__low >= FIRST_QUARTER and self.__high < THIRD_QUARTER:
                self.__value -= FIRST_QUARTER
                self.__low -= FIRST_QUARTER
                self.__high -= FIRST_QUARTER
            else:
                break
            self.__low *= 2
            self.__high *= 2
            self.__value = 2 * self.__value + self.__buffer.read_bit()
        return symbol_index


class ReadBuffer:
    def __init__(self, data: bytes):
        self.__byte = 0
        self.__curr_byte_index = 0
        self.__bits_to_read = 0
        self.__garbage_bits = 0
        self.__data = data
        self.end_encoding = False

    def read_bit(self):
        if self.__bits_to_read == 0:
            if self.__curr_byte_index == len(self.__data):
                self.end_encoding = True
                return -1
            self.__byte = self.__data[self.__curr_byte_index]
            self.__bits_to_read = 8
            self.__curr_byte_index += 1

        bit = self.__byte & 0x1 # get last bit
        self.__byte >>= 1
        self.__bits_to_read -= 1
        return bit

if __name__ == "__main__":
    data = bytearray(b'I\xe0\x9d\xdcL\xe3\x95\x7f\xad\x9b\x8b\x10\xcep\xe3\x93e\xe5\x15\x00')
    ad = ArithmeticBinaryDecoder(None)
    original = ad.decode(data)
    print(original)