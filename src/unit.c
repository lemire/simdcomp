/**
 * This code is released under a BSD License.
 */
#include <stdio.h>
#include "simdbitpacking.h"
#include "simdcomputil.h"
#include "simdintegratedbitpacking.h"

int main() {
    int N = 5000 * SIMDBlockSize;
    __m128i * buffer = malloc(SIMDBlockSize * sizeof(uint32_t));
    uint32_t * datain = malloc(N * sizeof(uint32_t));
    uint32_t * backbuffer = malloc(SIMDBlockSize * sizeof(uint32_t));
    for (int gap = 1; gap <= 387420489; gap *= 3) {
        printf(" gap = %u \n", gap);
        for (int k = 0; k < N; ++k)
            datain[k] = k * gap;
        uint32_t offset = 0;
        for (int k = 0; k * SIMDBlockSize < N; ++k) {
            const uint32_t b = maxbits(datain + k * SIMDBlockSize);
            simdpackwithoutmask(datain + k * SIMDBlockSize, buffer, b);//compressed
            simdunpack(buffer, backbuffer, b);//uncompressed
            for (int j = 0; j < SIMDBlockSize; ++j) {
                if (backbuffer[j] != datain[k * SIMDBlockSize + j]) {
                    printf("bug in simdpack\n");
                    return -2;
                }
            }
            const uint32_t b1 = simdmaxbitsd1(offset,
                    datain + k * SIMDBlockSize);
            simdpackwithoutmaskd1(offset, datain + k * SIMDBlockSize, buffer,
                    b1);//compressed
            simdunpackd1(offset, buffer, backbuffer, b1);//uncompressed
            for (int j = 0; j < SIMDBlockSize; ++j) {
                if (backbuffer[j] != datain[k * SIMDBlockSize + j]) {
                    printf("bug in simdpack d1\n");
                    return -3;
                }
            }
            offset = datain[k * SIMDBlockSize + SIMDBlockSize - 1];

        }
    }
    free(buffer);
    free(datain);
    free(backbuffer);
    printf("Code looks good.\n");
    return 0;
}
