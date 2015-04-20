/**
 * This code is released under a BSD License.
 */
#include <algorithm>
#include <assert.h>
#include <smmintrin.h>
#include "simdintegratedbitpacking.h"

const static __m128i shuffle_mask[16] = {
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,7,6,5,4),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,11,10,9,8),
        _mm_set_epi8(15,14,13,12,11,10,9,8,11,10,9,8,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,11,10,9,8,7,6,5,4),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,15,14,13,12),
        _mm_set_epi8(15,14,13,12,11,10,9,8,15,14,13,12,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,15,14,13,12,7,6,5,4),
        _mm_set_epi8(15,14,13,12,15,14,13,12,7,6,5,4,3,2,1,0),
        _mm_set_epi8(15,14,13,12,11,10,9,8,15,14,13,12,11,10,9,8),
        _mm_set_epi8(15,14,13,12,15,14,13,12,11,10,9,8,3,2,1,0),
        _mm_set_epi8(15,14,13,12,15,14,13,12,11,10,9,8,7,6,5,4),
        _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
    };

#define PrefixSum(ret, curr, prev) do {                                     \
    const __m128i _tmp1 = _mm_add_epi32(_mm_slli_si128(curr, 8), curr);     \
    const __m128i _tmp2 = _mm_add_epi32(_mm_slli_si128(_tmp1, 4), _tmp1);   \
    ret = _mm_add_epi32(_tmp2, _mm_shuffle_epi32(prev, 0xff));              \
	} while (0)

        //__m128i key4 = _mm_set1_epi32(key - 2147483648);
        //__m128i conversion = _mm_set1_epi32(2147483648); 
        //__m128i tmpout = _mm_sub_epi32(out,conversion);
        //__m128i cmp_mask = _mm_cmpeq_epi32(tmpout, key4);  

static const __m128i conversion = _mm_set1_epi32(2147483648u);

// perform a lower-bound search for |key| in |out|; the resulting uint32
// is stored in |*presult|.
#define CHECK_AND_INCREMENT(i, out, length, key, presult)                   \
      do {                                                                  \
     /* printf("key ?: %u\n", key);                       */                \
     /* printf("key 0: %u\n", _mm_extract_epi32(out, 0)); */                \
     /* printf("key 1: %u\n", _mm_extract_epi32(out, 1)); */                \
     /* printf("key 2: %u\n", _mm_extract_epi32(out, 2)); */                \
     /* printf("key 3: %u\n", _mm_extract_epi32(out, 3)); */                \
        __m128i key4 = _mm_set1_epi32(key - 2147483648u);                   \
        __m128i tmpout = _mm_sub_epi32(out, conversion);                    \
        uint32_t mask = _mm_movemask_ps((__m128)_mm_or_si128(               \
                                      _mm_cmpeq_epi32(tmpout, key4),        \
                                      _mm_cmpgt_epi32(tmpout, key4)));      \
        /* mask has value 1<<i where i is the correct index */              \
        if (mask != 0) {                                                    \
          const __m128i p = _mm_shuffle_epi8(out, shuffle_mask[mask]);      \
          *presult = _mm_cvtsi128_si32(p);                                  \
          int offset = __builtin_ctz(mask);                                 \
          int remaining = length - i;                                       \
       /* printf("mask is %x\n", mask); */                                  \
          if (offset < remaining)                                           \
            return (i + offset);                                            \
        }                                                                   \
        i += 4;                                                             \
        if (i >= length) { /* reached end of array? */                      \
          *presult = key + 1;                                               \
          return (length);                                                  \
        }                                                                   \
      } while (0)

static int
iunpacksearch0(__m128i  initOffset , const __m128i * /*_in*/, int length,
                uint32_t key, uint32_t *presult)
{
	if (length > 0) {
		uint32_t repeatedvalue = (uint32_t) _mm_extract_epi32(initOffset, 3);
		if (repeatedvalue >= key) {
			*presult = repeatedvalue;
			return 0;
		}
	}
  *presult = key + 1;
  return (length);
}

static int
iunpacksearch1(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<1)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch2(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<2)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch3(__m128i initOffset, const  __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<3)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 3-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 3-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch4(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<4)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch5(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<5)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch6(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<6)-1);
 
  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch7(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<7)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch8(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<8)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch9(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<9)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch10(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<10)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch11(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<11)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch12(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<12)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch13(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<13)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch14(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<14)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch15(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<15)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch16(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<16)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch17(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<17)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch18(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<18)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch19(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<19)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch20(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<20)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch21(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<21)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch22(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<22)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch23(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<23)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch24(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<24)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch25(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<25)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch26(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<26)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch27(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<27)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch28(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<28)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch29(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<29)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-27), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch30(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<30)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch31(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<31)-1);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-30), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-29), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-27), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
iunpacksearch32(__m128i /*initOffset*/, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  uint32_t *begin = (uint32_t *)in;
  uint32_t *end = begin + length;
  uint32_t *it = std::lower_bound(begin, end, key);
  if (it == end) {
    *presult = key + 1;
    return (length);
  }
  *presult = *it;
  return (it - begin);
}

extern "C" {

int
simdsearchd1(uint32_t initvalue, const __m128i *in, uint32_t bit, int length,
                uint32_t key, uint32_t *presult)
{
   __m128i initOffset = _mm_set1_epi32 (initvalue);
   switch (bit) {
    case 0: return iunpacksearch0(initOffset, in, length, key, presult);

    case 1: return iunpacksearch1(initOffset, in, length, key, presult);

    case 2: return iunpacksearch2(initOffset, in, length, key, presult);

    case 3: return iunpacksearch3(initOffset, in, length, key, presult);

    case 4: return iunpacksearch4(initOffset, in, length, key, presult);

    case 5: return iunpacksearch5(initOffset, in, length, key, presult);

    case 6: return iunpacksearch6(initOffset, in, length, key, presult);

    case 7: return iunpacksearch7(initOffset, in, length, key, presult);

    case 8: return iunpacksearch8(initOffset, in, length, key, presult);

    case 9: return iunpacksearch9(initOffset, in, length, key, presult);

    case 10: return iunpacksearch10(initOffset, in, length, key, presult);

    case 11: return iunpacksearch11(initOffset, in, length, key, presult);

    case 12: return iunpacksearch12(initOffset, in, length, key, presult);

    case 13: return iunpacksearch13(initOffset, in, length, key, presult);

    case 14: return iunpacksearch14(initOffset, in, length, key, presult);

    case 15: return iunpacksearch15(initOffset, in, length, key, presult);

    case 16: return iunpacksearch16(initOffset, in, length, key, presult);

    case 17: return iunpacksearch17(initOffset, in, length, key, presult);

    case 18: return iunpacksearch18(initOffset, in, length, key, presult);

    case 19: return iunpacksearch19(initOffset, in, length, key, presult);

    case 20: return iunpacksearch20(initOffset, in, length, key, presult);

    case 21: return iunpacksearch21(initOffset, in, length, key, presult);

    case 22: return iunpacksearch22(initOffset, in, length, key, presult);

    case 23: return iunpacksearch23(initOffset, in, length, key, presult);

    case 24: return iunpacksearch24(initOffset, in, length, key, presult);

    case 25: return iunpacksearch25(initOffset, in, length, key, presult);

    case 26: return iunpacksearch26(initOffset, in, length, key, presult);

    case 27: return iunpacksearch27(initOffset, in, length, key, presult);

    case 28: return iunpacksearch28(initOffset, in, length, key, presult);

    case 29: return iunpacksearch29(initOffset, in, length, key, presult);

    case 30: return iunpacksearch30(initOffset, in, length, key, presult);

    case 31: return iunpacksearch31(initOffset, in, length, key, presult);

    case 32: return iunpacksearch32(initOffset, in, length, key, presult);

    default: break;
   }
   return (-1);
}

} // extern "C"























/****************
 *
 * New
 */









#undef CHECK_AND_INCREMENT

// perform a lower-bound search for |key| in |out|; the resulting uint32
// is stored in |*presult|.
#define CHECK_AND_INCREMENT(i, out, length, key, presult)                   \
      do {                                                                  \
     /* printf("key ?: %u\n", key);                       */                \
     /* printf("key 0: %u\n", _mm_extract_epi32(out, 0)); */                \
     /* printf("key 1: %u\n", _mm_extract_epi32(out, 1)); */                \
     /* printf("key 2: %u\n", _mm_extract_epi32(out, 2)); */                \
     /* printf("key 3: %u\n", _mm_extract_epi32(out, 3)); */                \
                           \
        __m128i tmpout = _mm_sub_epi32(out, conversion);                    \
        uint32_t mask = _mm_movemask_ps((__m128)_mm_or_si128(               \
                                      _mm_cmpeq_epi32(tmpout, key4),        \
                                      _mm_cmpgt_epi32(tmpout, key4)));      \
        /* mask has value 1<<i where i is the correct index */              \
        if (mask != 0) {                                                    \
          const __m128i p = _mm_shuffle_epi8(out, shuffle_mask[mask]);      \
          *presult = _mm_cvtsi128_si32(p);                                  \
          int offset = __builtin_ctz(mask);                                 \
          int remaining = length - i;                                       \
       /* printf("mask is %x\n", mask); */                                  \
          if (offset < remaining)                                           \
            return (i + offset);                                            \
        }                                                                   \
        i += 4;                                                             \
        if (i >= length) { /* reached end of array? */                      \
          *presult = key + 1;                                               \
          return (length);                                                  \
        }                                                                   \
      } while (0)

static int
newiunpacksearch0(__m128i  initOffset , const __m128i * /*_in*/, int length,
                uint32_t key, uint32_t *presult)
{
	if (length > 0) {
		uint32_t repeatedvalue = (uint32_t) _mm_extract_epi32(initOffset, 3);
		if (repeatedvalue >= key) {
			*presult = repeatedvalue;
			return 0;
		}
	}
  *presult = key + 1;
  return (length);
}

static int
newiunpacksearch1(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<1)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch2(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<2)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch3(__m128i initOffset, const  __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<3)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 3-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 3-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch4(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<4)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch5(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<5)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 5-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch6(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<6)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 6-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch7(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<7)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 7-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch8(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<8)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch9(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<9)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 9-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch10(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<10)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 10-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch11(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<11)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 11-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch12(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<12)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 12-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch13(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<13)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 13-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch14(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<14)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 14-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch15(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<15)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 15-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch16(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<16)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch17(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<17)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 17-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch18(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<18)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 18-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch19(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<19)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 19-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch20(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<20)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 20-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch21(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<21)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 21-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch22(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<22)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 22-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch23(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<23)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 23-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch24(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<24)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 24-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch25(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<25)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 25-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch26(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<26)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 26-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch27(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<27)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 27-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch28(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<28)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 28-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch29(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<29)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-27), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 29-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch30(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<30)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 30-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch31(__m128i initOffset, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  int i = 0;
  __m128i InReg = _mm_loadu_si128(in);
  __m128i out;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U<<31)-1);
  __m128i conversion = _mm_set1_epi32(2147483648U);
  __m128i key4 = _mm_set1_epi32(key - 2147483648U);


  tmp = InReg;
  out = _mm_and_si128(tmp, mask);
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,31);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-30), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,30);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-29), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,29);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-28), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,28);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-27), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,27);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-26), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,26);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-25), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,25);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-24), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,24);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-23), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,23);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-22), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,22);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-21), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,21);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-20), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,20);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-19), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,19);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-18), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,18);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-17), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,17);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-16), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,16);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-15), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,15);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-14), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,14);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-13), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,13);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-12), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,12);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-11), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,11);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-10), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,10);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-9), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,9);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-8), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,8);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-7), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,7);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-6), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,6);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-5), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,5);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-4), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,4);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-3), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,3);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-2), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,2);
  out = tmp;
  ++in;  InReg = _mm_loadu_si128(in);
  out = _mm_or_si128(out, _mm_and_si128(_mm_slli_epi32(InReg, 31-1), mask));

  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  tmp = _mm_srli_epi32(InReg,1);
  out = tmp;
  PrefixSum(out, out, initOffset);
  initOffset = out;
  CHECK_AND_INCREMENT(i, out, length, key, presult);

  *presult = key + 1;
  return (128);
}

static int
newiunpacksearch32(__m128i /*initOffset*/, const __m128i *in, int length,
                uint32_t key, uint32_t *presult)
{
  uint32_t *begin = (uint32_t *)in;
  uint32_t *end = begin + length;
  uint32_t *it = std::lower_bound(begin, end, key);
  if (it == end) {
    *presult = key + 1;
    return (length);
  }
  *presult = *it;
  return (it - begin);
}

extern "C" {

int
newsimdsearchd1(uint32_t initvalue, const __m128i *in, uint32_t bit, int length,
                uint32_t key, uint32_t *presult)
{
   __m128i initOffset = _mm_set1_epi32 (initvalue);
   switch (bit) {
    case 0: return newiunpacksearch0(initOffset, in, length, key, presult);

    case 1: return newiunpacksearch1(initOffset, in, length, key, presult);

    case 2: return newiunpacksearch2(initOffset, in, length, key, presult);

    case 3: return newiunpacksearch3(initOffset, in, length, key, presult);

    case 4: return newiunpacksearch4(initOffset, in, length, key, presult);

    case 5: return newiunpacksearch5(initOffset, in, length, key, presult);

    case 6: return newiunpacksearch6(initOffset, in, length, key, presult);

    case 7: return newiunpacksearch7(initOffset, in, length, key, presult);

    case 8: return newiunpacksearch8(initOffset, in, length, key, presult);

    case 9: return newiunpacksearch9(initOffset, in, length, key, presult);

    case 10: return newiunpacksearch10(initOffset, in, length, key, presult);

    case 11: return newiunpacksearch11(initOffset, in, length, key, presult);

    case 12: return newiunpacksearch12(initOffset, in, length, key, presult);

    case 13: return newiunpacksearch13(initOffset, in, length, key, presult);

    case 14: return newiunpacksearch14(initOffset, in, length, key, presult);

    case 15: return newiunpacksearch15(initOffset, in, length, key, presult);

    case 16: return newiunpacksearch16(initOffset, in, length, key, presult);

    case 17: return newiunpacksearch17(initOffset, in, length, key, presult);

    case 18: return newiunpacksearch18(initOffset, in, length, key, presult);

    case 19: return newiunpacksearch19(initOffset, in, length, key, presult);

    case 20: return newiunpacksearch20(initOffset, in, length, key, presult);

    case 21: return newiunpacksearch21(initOffset, in, length, key, presult);

    case 22: return newiunpacksearch22(initOffset, in, length, key, presult);

    case 23: return newiunpacksearch23(initOffset, in, length, key, presult);

    case 24: return newiunpacksearch24(initOffset, in, length, key, presult);

    case 25: return newiunpacksearch25(initOffset, in, length, key, presult);

    case 26: return newiunpacksearch26(initOffset, in, length, key, presult);

    case 27: return newiunpacksearch27(initOffset, in, length, key, presult);

    case 28: return newiunpacksearch28(initOffset, in, length, key, presult);

    case 29: return newiunpacksearch29(initOffset, in, length, key, presult);

    case 30: return newiunpacksearch30(initOffset, in, length, key, presult);

    case 31: return newiunpacksearch31(initOffset, in, length, key, presult);

    case 32: return newiunpacksearch32(initOffset, in, length, key, presult);

    default: break;
   }
   return (-1);
}

} // extern "C"
