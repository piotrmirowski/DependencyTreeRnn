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


#ifndef __DependencyTreeRNN____json__
#define __DependencyTreeRNN____json__

#include "block_allocator.h"

enum json_type
{
    JSON_NULL,
    JSON_OBJECT,
    JSON_ARRAY,
    JSON_STRING,
    JSON_INT,
    JSON_FLOAT,
    JSON_BOOL,
};

struct json_value
{
    json_value *parent;
    json_value *next_sibling;
    json_value *first_child;
    json_value *last_child;
    
    char *name;
    union
    {
        char *string_value;
        int int_value;
        float float_value;
    };
    
    json_type type;
};

json_value *json_parse(char *source, char **error_pos, const char **error_desc, int *error_line, block_allocator *allocator);

#endif /* defined(__DependencyTreeRNN____json__) */
