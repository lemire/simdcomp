/**
 * This code is released under a BSD License.
 */
#ifndef INCLUDE_SIMDFOR_H_
#define INCLUDE_SIMDFOR_H_

#include "portability.h"

/* SSE2 is required */
#include <emmintrin.h>

#include "simdcomputil.h"

#ifdef __cplusplus
extern "C" {
#endif

/* reads 128 values from "in", writes  "bit" 128-bit vectors to "out" */
void simdpackFOR(uint32_t initvalue, const uint32_t *  in,__m128i *  out, const uint32_t bit);


/* reads "bit" 128-bit vectors from "in", writes  128 values to "out" */
void simdunpackFOR(uint32_t initvalue, const __m128i *  in,uint32_t *  out, const uint32_t bit);


#ifdef __cplusplus
} // extern "C"
#endif




#endif /* INCLUDE_SIMDFOR_H_ */
