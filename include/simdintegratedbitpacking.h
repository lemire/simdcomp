/**
 * This code is released under a BSD License.
 */

#ifndef SIMD_INTEGRATED_BITPACKING_H
#define SIMD_INTEGRATED_BITPACKING_H

#include "portability.h"

/* SSE2 is required */
#include <emmintrin.h>

#include "simdcomputil.h"

#ifdef __cplusplus
extern "C" {
#endif

/* reads 128 values from "in", writes  "bit" 128-bit vectors to "out"
   integer values should be in sorted order (for best results) */
void simdpackd1(uint32_t initvalue, const uint32_t *  in,__m128i *  out, const uint32_t bit);


/* reads 128 values from "in", writes  "bit" 128-bit vectors to "out"
   integer values should be in sorted order (for best results) */
void simdpackwithoutmaskd1(uint32_t initvalue, const uint32_t *  in,__m128i *  out, const uint32_t bit);


/* reads "bit" 128-bit vectors from "in", writes  128 values to "out" */
void simdunpackd1(uint32_t initvalue, const __m128i *  in,uint32_t *  out, const uint32_t bit);

/* searches "bit" 128-bit vectors from "in" for the first encoded uint32 value
 * which is >= |key|, and returns its position.
 * The encoded key is stored in "*presult". */
int simdsearchd1(uint32_t initvalue, const __m128i *in, uint32_t bit,
                int length, uint32_t key, uint32_t *presult);

/* returns the value stored at the specified "slot". */
uint32_t simdselectd1(uint32_t initvalue, const __m128i *in, uint32_t bit,
                int slot);


#ifdef __cplusplus
} // extern "C"
#endif

#endif
