/**
 * This code is released under a BSD License.
 */
#ifndef SIMDBITCOMPAT_H_
#define SIMDBITCOMPAT_H_

#include <iso646.h> /* mostly for Microsoft compilers */
#include <string.h>

#if defined(_MSC_VER) && _MSC_VER < 1600
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
#else
#include <stdint.h> /* part of Visual Studio 2010 and better, others likely anyway */
#endif

#if defined(_MSC_VER)
#define SIMDCOMP_ALIGNED(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define SIMDCOMP_ALIGNED(x) __attribute__ ((aligned(x)))
#endif
#endif

#if defined(_MSC_VER)
# include <intrin.h>
/* 64-bit needs extending */
# define SIMDCOMP_CTZ(result, mask) do { \
		unsigned long index; \
		if (!_BitScanForward(&(index), (mask))) { \
			(result) = 32U; \
		} else { \
			(result) = (uint32_t)(index); \
		} \
	} while (0)
#else
# include <x86intrin.h> 
# define SIMDCOMP_CTZ(result, mask) \
	result = __builtin_ctz(mask)
#endif

#endif /* SIMDBITCOMPAT_H_ */

