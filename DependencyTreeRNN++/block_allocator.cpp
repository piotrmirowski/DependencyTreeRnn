// Note: this code is part of an external library. The copyright notice does
// not imply that the author of the paper is the owner of the copyright
// of this file.

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


#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "block_allocator.h"

block_allocator::block_allocator(size_t blocksize): m_head(0), m_blocksize(blocksize)
{
}

block_allocator::~block_allocator()
{
    while (m_head)
    {
        block *temp = m_head->next;
        ::free(m_head);
        m_head = temp;
    }
}

void block_allocator::swap(block_allocator &rhs)
{
    //std::swap(m_blocksize, rhs.m_blocksize);
    size_t temp1 = m_blocksize;
    m_blocksize = rhs.m_blocksize;
    rhs.m_blocksize = temp1;
    //std::swap(m_head, rhs.m_head);
    //block *temp2 = m_head;
    m_head = rhs.m_head;
    rhs.m_head = m_head;
}

void *block_allocator::malloc(size_t size)
{
    if ((m_head && m_head->used + size > m_head->size) || !m_head)
    {
        // calc needed size for allocation
        //size_t alloc_size = std::max(sizeof(block) + size, m_blocksize);
        size_t temp = sizeof(block) + size;
        size_t alloc_size = (temp > m_blocksize) ? temp : m_blocksize;
        
        // create new block
        char *buffer = (char *)::malloc(alloc_size);
        block *b = reinterpret_cast<block *>(buffer);
        b->size = alloc_size;
        b->used = sizeof(block);
        b->buffer = buffer;
        b->next = m_head;
        m_head = b;
    }
    
    void *ptr = m_head->buffer + m_head->used;
    m_head->used += size;
    return ptr;
}

void block_allocator::free()
{
    block_allocator(0).swap(*this);
}
