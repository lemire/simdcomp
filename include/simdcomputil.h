/**
 * This code is released under a BSD License.
 */

#ifndef SIMDCOMPUTIL_H_
#define SIMDCOMPUTIL_H_

#include <emmintrin.h>// SSE2 is required
#include <stdint.h> // use a C99-compliant compiler, please

__attribute__((always_inline))
inline __m128i PrefixSum(__m128i curr, __m128i prev) {
    const __m128i _tmp1 = _mm_add_epi32(_mm_slli_si128(curr, 8), curr);
    const __m128i _tmp2 = _mm_add_epi32(_mm_slli_si128(_tmp1, 4), _tmp1);
    return _mm_add_epi32(_tmp2, _mm_shuffle_epi32(prev, 0xff));
}

__attribute__((always_inline))
inline __m128i Delta(__m128i curr, __m128i prev) {
    return _mm_sub_epi32(curr,
            _mm_or_si128(_mm_slli_si128(curr, 4), _mm_srli_si128(prev, 12)));
}

// returns the integer logarithm of v (bit width)
uint32_t bits(const uint32_t v) {
#ifdef _MSC_VER
    if (v == 0) {
        return 0;
    }
    unsigned long answer;
    _BitScanReverse(&answer, v);
    return answer + 1;
#else
    return v == 0 ? 0 : 32 - __builtin_clz(v); // assume GCC-like compiler if not microsoft
#endif
}

__attribute__ ((pure))
uint32_t maxbits(const uint32_t * begin, const uint32_t * end) {
    uint32_t accumulator = 0;
    for (const uint32_t * k = begin; k != end; ++k) {
        accumulator |= *k;
    }
    return bits(accumulator);
}

static uint32_t maxbitas32int(const __m128i accumulator) {
    uint32_t tmparray[4];
    _mm_storeu_si128((__m128i *) (tmparray), accumulator);
    return bits(tmparray[0] | tmparray[1] | tmparray[2] | tmparray[3]);
}

const uint32_t SIMDBlockSize = 128;

// maxbit over 128 integers (SIMDBlockSize) with provided initial value
uint32_t simdmaxbitsd1(uint32_t initvalue, const uint32_t * in) {
    __m128i  initoffset = _mm_set1_epi32 (initvalue);
    const __m128i* pin = (const __m128i*)(in);
    __m128i newvec = _mm_load_si128(pin);
    __m128i accumulator = Delta(newvec , initoffset);
    __m128i oldvec = newvec;
    for(uint32_t k = 1; 4*k < SIMDBlockSize; ++k) {
        newvec = _mm_load_si128(pin+k);
        accumulator = _mm_or_si128(accumulator,Delta(newvec , oldvec));
        oldvec = newvec;
    }
    initoffset = oldvec;
    return maxbitas32int(accumulator);
}




#endif /* SIMDCOMPUTIL_H_ */
