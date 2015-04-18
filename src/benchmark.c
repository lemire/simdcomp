/**
 * This code is released under a BSD License.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "simdcomp.h"


void benchmarkSelect() {
    uint32_t buffer[128];
    uint32_t backbuffer[128];
    uint32_t initial = 33;
    uint32_t b;
    clock_t S1, S2, S3;
    int i;
    printf("benchmarking select \n");

    /* this test creates delta encoded buffers with different bits, then
     * performs lower bound searches for each key */
    for (b = 0; b <= 32; b++) {
        uint32_t prev = initial;
    	uint32_t out[128];
        /* initialize the buffer */
        for (i = 0; i < 128; i++) {
            buffer[i] =  ((uint32_t)(1431655765 * i + 0xFFFFFFFF)) ;
            if(b < 32) buffer[i] %= (1<<b);
        }
        for (i = 0; i < 128; i++) {
           buffer[i] = buffer[i] + prev;
           prev = buffer[i];
        }

        for (i = 1; i < 128; i++) {
        	if(buffer[i] < buffer[i-1] )
        		buffer[i] = buffer[i-1];
        }
        assert(simdmaxbitsd1(initial, buffer)<=b);

        for (i = 0; i < 128; i++) {
        	out[i] = 0; /* memset would do too */
        }

        /* delta-encode to 'i' bits */
        simdpackwithoutmaskd1(initial, buffer, (__m128i *)out, b);

        S1 = clock();
        for (i = 0; i < 128 * 10; i++) {
        	uint32_t valretrieved = simdselectd1(initial, (__m128i *)out, b, (uint32_t)i % 128);
            assert(valretrieved == buffer[i%128]);
        }
        S2 = clock();
        for (i = 0; i < 128 * 10; i++) {
        	simdunpackd1(initial,  (__m128i *)out, backbuffer, b);
        	assert(backbuffer[i % 128] == buffer[i % 128]);
        }
        S3 = clock();
        printf("bit width = %d, fast select function time = %lu, naive time = %lu  \n", b, (S2-S1), (S3-S2));
    }

}
int main() {
	benchmarkSelect();
        return 0;
}
