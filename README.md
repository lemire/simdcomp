The SIMDComp library 
====================

A simple C library for compressing lists of integers using binary packing and SIMD instructions.
The assumption is either that you have a list of 32-bit integers where most of them are small, or a list of 32-bit integers where differences between successive integers are small. No software is able to reliably compress an array of random numbers.

This library can decode at least 4 billions of compressed integers per second on most
desktop or laptop processors. That is, it can decompress data at a rate of 15 GB/s.
This is significantly faster than generic codecs like gzip, LZO, Snappy or LZ4.


What is it for?
-------------

This is a low-level library for fast integer compression. By design it does not define a compressed
format. It is up to the (sophisticated) user to create a compressed format.

Requirements
-------------

- Your processor should support SSE2 (Pentium4 or better)
- C99 compliant compiler (GCC is assumed)
- A Linux-like distribution is assumed by the makefile

Usage
-------

Compression works over blocks of 128 integers.

For a complete working example, see example.c (you can build it and
run it with "make example; ./example").



1) Lists of integers in random order.

        const uint32_t b = maxbits(datain);// computes bit width
        simdpackwithoutmask(datain, buffer, b);//compressed to buffer
        simdunpack(buffer, backbuffer, b);//uncompressed to backbuffer

While 128 32-bit integers are read, only b 128-bit words are written. Thus, the compression ratio is 32/b.

2) Sorted lists of integers.

We used differential coding: we store the difference between successive integers. For this purpose, we need an initial value (called offset).
            
        uint32_t offset = 0; 
        uint32_t b1 = simdmaxbitsd1(offset,datain); // bit width
        simdpackwithoutmaskd1(offset, datain, buffer, b1);//compressed
        simdunpackd1(offset, buffer, backbuffer, b1);//uncompressed

Setup
---------


make
make test

and if you are daring:

make install 

Go
--------

If you are a go user, there is a "go" folder where you will find a simple demo.

Other libraries
----------------

FastPFOR is a C++ research library well suited to compress unsorted arrays:
https://github.com/lemire/FastPFor

SIMDCompressionAndIntersection is a C++ research library well suited for sorted arrays (differential coding)
and computing intersections:
https://github.com/lemire/SIMDCompressionAndIntersection

References
------------

* Daniel Lemire, Leonid Boytsov, Nathan Kurz, SIMD Compression and the Intersection of Sorted Integers, Software Practice & Experience (to appear) http://arxiv.org/abs/1401.6399
* Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second through vectorization, Software Practice & Experience 45 (1), 2015.  http://arxiv.org/abs/1209.2137 http://onlinelibrary.wiley.com/doi/10.1002/spe.2203/abstract
* Jeff Plaisance, Nathan Kurz, Daniel Lemire, Vectorized VByte Decoding, International Symposium on Web Algorithms 2015, 2015. http://arxiv.org/abs/1503.07387
* Wayne Xin Zhao, Xudong Zhang, Daniel Lemire, Dongdong Shan, Jian-Yun Nie, Hongfei Yan, Ji-Rong Wen, A General SIMD-based Approach to Accelerating Compression Algorithms, ACM Transactions on Information Systems 33 (3), 2015. http://arxiv.org/abs/1502.01916


