import numpy as np


# Python translation of fpaq0 coder by M. Mahoney (http://mattmahoney.net/dc/fpaq0.cpp)
class ArithmeticBitCoder:

    def __init__(self):
        self.predictor = NaivePredictor()
        self.x1 = np.int64(0)
        self.x2 = np.int64(0xffffffff)
        self.x = np.int64(0)
        self.compressed = bytearray()
        self.context = 1

    def encode(self, data: bytes):
        self.x1 = np.int64(0)
        self.x2 = np.int64(0xffffffff)
        self.x = np.int64(0)
        self.compressed = bytearray()
        for b in data:
            self.encode_symbol(0)
            for i in range(7, -1, -1):
                self.encode_symbol((b >> i) & 1)
        self.encode_symbol(1)
        self.flush()
        return self.compressed

    def encode_symbol(self, y: int):
        xmid = np.int64(self.x1 + (
                (self.x2 - self.x1) >> 12) * self.predictor.p())  # shift by twelve accounts for the 12-bit probability
        assert (self.x1 <= xmid < self.x2)
        # print(xmid)
        if y != 0:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1

        self.predictor.update(y)
        self.context += self.context + y
        if self.context >= 512:
            self.context = 1

        while ((self.x1 ^ self.x2) & 0xff000000) == 0:
            # print(f"Shifting with x1 {self.x1} and x2 {self.x2}")
            self.compressed.append((self.x2 >> 24).tobytes()[0])
            self.x1 <<= 8
            self.x2 = (self.x2 << 8) + 255

    def decode(self, compressed: bytearray):
        self.compressed = compressed
        decompressed = bytearray()
        for i in range(4):  # initialise first four bytes of x with compressed file
            if len(compressed) == 0:
                b = 0
            else:
                b = compressed.pop(0)
            self.x = (self.x << 8) + (b & 0xff)

        while self.decode_symbol() == 0:
            b = 1
            while b < 256:
                b += b + self.decode_symbol()
            decompressed.append(b - 256)
        return decompressed

    def decode_symbol(self):
        xmid = np.int64(self.x1 + (
                (self.x2 - self.x1) >> 12) * self.predictor.p())
        assert self.x1 <= xmid < self.x2
        y = 0
        if self.x <= xmid:
            y = 1
            self.x2 = xmid
        else:
            self.x1 = xmid + 1
        self.predictor.update(y)

        while ((self.x1 ^ self.x2) & 0xff000000) == 0:
            self.x1 <<= 8
            self.x2 = (self.x2 << 8) + 255
            if len(self.compressed) == 0:
                c = 0
            else:
                c = self.compressed.pop(0)
            self.x = (self.x << 8) + c

        return y

    def flush(self):
        while ((self.x1 ^ self.x2) & 0xff000000) == 0:
            self.compressed.append((self.x2 >> 24).tobytes()[0])
            self.x1 <<= 8
            self.x2 = (self.x2 << 8) + 255
        self.compressed.append((self.x2 >> 24).tobytes()[0])


class NaivePredictor:

    def __init__(self):
        self.context = 1
        self.counter = [[0] * 2 for i in range(512)]

    def p(self):
        return np.int64(4096 * (self.counter[self.context][1] + 1) / (
                self.counter[self.context][0] + self.counter[self.context][1] + 2))

    def update(self, y):
        self.counter[self.context][y] += 1
        if self.counter[self.context][y] > 65534:
            self.counter[self.context][0] >>= 1
            self.counter[self.context][1] >>= 1
        self.context += self.context + y
        if self.context >= 512:
            self.context = 1


def test():
    data = b'\r\x00\x05\x00\xfb\xff\xf5\xff\x02\x00\xcf\xff\xf6\xff\xfa\xff\x01\x00\xfc\xff\r\x00\xfb\xff\x0e\x00\xfd\xff\x03\x00\xff\xff\x07\x00\xff\xff\x06\x00\x00\x00\x06\x00\xfa\xff\x04\x00\xfe\xff\xfd\xff\xfe\xff\xff\xff\xff\xff\xf4\xff\xfb\xff\x01\x00\xfe\xff\x01\x00\x02\x00\x0b\x00\x04\x00\xfe\xff\xff\xff\xff\xff\xff\xff\xff\xff\x06\x00\x01\x00\x03\x00\xfe\xff\xff\xff\x02\x00\xff\xff\x00\x00\x00\x00\x00\x00\x04\x00\x07\x00\xfe\xff\x00\x00\x00\x00\xff\xff\xfe\xff\xfb\xff\x00\x00\x01\x00\xfe\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\xfb\xff\x04\x00\xff\xff\x01\x00\x02\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\xfe\xff\x02\x00\x00\x00\x00\x00\xff\xff\xfe\xff\x03\x00\xfe\xff\xfe\xff\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xfe\xff\x01\x00\x02\x00\x02\x00\x01\x00\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x01\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x01\x00\xff\xff\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x01\x00\x02\x00\x00\x00\x01\x00\x01\x00\x00\x00\x02\x00\xff\xff\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x01\x00\x00\x00\xfd\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x01\x00\x01\x00\xff\xff\x01\x00\x00\x00\x00\x00\x01\x00\x01\x00\xff\xff\x00\x00\x01\x00\xff\xff\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\xff\xff\x01\x00\x01\x00\x00\x00\x01\x00\xff\xff\x01\x00\x01\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\r\x00\xf3\xff\x05\x00\xfe\xff\x0b\x00\x04\x00\x01\x00\xff\xff\x06\x00\x05\x00\xfb\xff\xfa\xff\xfe\xff\x00\x00\x03\x00\xfe\xff\x05\x00\xff\xff\x02\x00\xfc\xff\xfc\xff\xfd\xff\xfe\xff\x02\x00\xf8\xff\x04\x00\xfa\xff\xf9\xff\xff\xff\x02\x00\x02\x00\x00\x00\x03\x00\x05\x00\xfe\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x02\x00\x02\x00\xff\xff\x00\x00\xff\xff\x01\x00\x00\x00\xff\xff\x01\x00\x02\x00\x02\x00\x04\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\xfe\xff\x00\x00\x01\x00\xff\xff\x01\x00\xff\xff\xff\xff\x00\x00\x00\x00\x01\x00\xff\xff\x02\x00\xff\xff\x01\x00\x01\x00\x01\x00\xfd\xff\x03\x00\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x01\x00\x01\x00\xff\xff\xff\xff\xff\xff\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x01\x00\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x00\xfa\xff\xf8\xff\xf8\xff\x19\x00\xb5\xff\xc8\xff\xf6\xff\t\x00\x0c\x00\x15\x00\xe1\xff\x06\x00\xfb\xff\x06\x00\x07\x00\xfc\xff\x1a\x00\xfc\xff\xfe\xff\xee\xff\x01\x00\xfc\xff\x06\x00\x02\x00\x02\x00\xfd\xff\x00\x00\r\x00\xf4\xff\x03\x00\xff\xff\x01\x00\x04\x00\xeb\xff\xf4\xff\x00\x00\x00\x00\xff\xff\x01\x00\xfd\xff\xf6\xff\x03\x00\t\x00\xff\xff\x02\x00\xfe\xff\xff\xff\x02\x00\xfe\xff\xfd\xff\xff\xff\xfd\xff\xff\xff\x00\x00\x00\x00\x02\x00\xfd\xff\x01\x00\x03\x00\x01\x00\x01\x00\xfe\xff\xff\xff\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\xfa\xff\x05\x00\xfc\xff\x04\x00\xff\xff\x01\x00\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\xff\xff\x00\x00\xff\xff\x02\x00\x01\x00\xfe\xff\x00\x00\x01\x00\x00\x00\x01\x00\xff\xff\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\xfe\xff\x02\x00\xfd\xff\xfd\xff\x01\x00\x01\x00\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\x01\x00\xff\xff\xff\xff\x00\x00\x01\x00\x00\x00\x01\x00\x01\x00\xff\xff\x00\x00\x00\x00\x01\x00\xff\xff\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\xff\xff\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\xff\xff\x02\x00\xfe\xff\xff\xff\x01\x00\x01\x00\x01\x00\x01\x00\xff\xff\x01\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\xff\xff\x01\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x06\x00\x0e\x00\x04\x00\xfd\xff\x06\x00\x08\x00\x00\x00\xfe\xff\x02\x00\xfa\xff\x05\x00\xfc\xff\x01\x00\x06\x00\x00\x00\xfc\xff\xfb\xff\xfd\xff\x01\x00\xf6\xff\x05\x00\x03\x00\xfc\xff\xfe\xff\xfe\xff\x01\x00\x01\x00\x03\x00\x06\x00\xfa\xff\xfd\xff\xff\xff\xff\xff\xf5\xff\xfd\xff\x01\x00\x01\x00\x00\x00\xff\xff\x03\x00\xfb\xff\x00\x00\x01\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x03\x00\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x04\x00\x00\x00\x01\x00\x02\x00\xff\xff\xff\xff\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\xfd\xff\xfe\xff\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\xff\xff\x00\x00\x01\x00\xfd\xff\x02\x00\x01\x00\x00\x00\xfe\xff\x01\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x00\x02\x00\xff\xff\x00\x00\xfc\xff\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfe\xff\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00\xff\xff\x01\x00\xff\xff\xff\xff\xff\xff\xfe\xff\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x01\x00\xff\xff\xff\xff\x01\x00\x01\x00\x00\x00\x02\x00\x01\x00\x00\x00\xff\xff\xff\xff\xff\xff\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    bitcoder = ArithmeticBitCoder()
    enc = bitcoder.encode(data)
    bitcoder.__init__()
    dec = bitcoder.decode(enc)
    print(dec)


def test_decode():
    pass


if __name__ == "__main__":
    test()
