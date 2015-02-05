// Copyright (c) 2010 Ivan Vashchaev. All rights reserved.
// Retrieved from: https://code.google.com/p/vjson/
// Distributed under the MIT license.

/*
 The MIT License (MIT)
 
 Copyright (c) <year> <copyright holders>
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef __DependencyTreeRNN____block_allocator__
#define __DependencyTreeRNN____block_allocator__

class block_allocator
{
private:
    struct block
    {
        size_t size;
        size_t used;
        char *buffer;
        block *next;
    };
    
    block *m_head;
    size_t m_blocksize;
    
    block_allocator(const block_allocator &);
    block_allocator &operator=(block_allocator &);
    
public:
    block_allocator(size_t blocksize);
    ~block_allocator();
    
    // exchange contents with rhs
    void swap(block_allocator &rhs);
    
    // allocate memory
    void *malloc(size_t size);
    
    // free all allocated blocks
    void free();
};

#endif /* defined(__DependencyTreeRNN____block_allocator__) */
