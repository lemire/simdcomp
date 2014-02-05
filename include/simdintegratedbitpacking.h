/**
 * This code is released under a BSD License.
 */

#ifndef SIMD_INTEGRATED_BITPACKING_H
#define SIMD_INTEGRATED_BITPACKING_H

#include <emmintrin.h>// SSE2 is required
#include <stdint.h> // use a C99-compliant compiler, please

#include "simdcomputil.h"

void simdpackd1(uint32_t initvalue, const uint32_t *  in,__m128i *  out, uint32_t bit);
void simdpackwithoutmaskd1(uint32_t initvalue, const uint32_t *  in,__m128i *  out, uint32_t bit);
void simdunpackd1(uint32_t initvalue, const __m128i *  in,uint32_t *  out, uint32_t bit);


#endif
