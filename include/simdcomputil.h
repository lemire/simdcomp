/**
 * This code is released under a BSD License.
 */

#ifndef SIMDCOMPUTIL_H_
#define SIMDCOMPUTIL_H_

#include "portability.h"

/* SSE2 is required */
#include <emmintrin.h>




/* returns the integer logarithm of v (bit width) */
uint32_t bits(const uint32_t v);

/* max integer logarithm over a range of SIMDBlockSize integers (128 integer) */
uint32_t maxbits(const uint32_t * begin);

enum{ SIMDBlockSize = 128};

/* like maxbit over 128 integers (SIMDBlockSize) with provided initial value 
   and using differential coding */
uint32_t simdmaxbitsd1(uint32_t initvalue, const uint32_t * in);




#endif /* SIMDCOMPUTIL_H_ */
