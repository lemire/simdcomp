simdcomp
========

A simple C library for compressing lists of integers


Usage
-------

Compression works over blocks of 128 integers.

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



References
------------


Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second through vectorization, Software: Practice & Experience, 2013. 
http://dx.doi.org/10.1002/spe.2203

Daniel Lemire, Leonid Boytsov, Nathan Kurz, SIMD Compression and the
Intersection of Sorted Integers
http://arxiv.org/abs/1401.6399

