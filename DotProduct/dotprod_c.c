/* 
   dotprod_c.c -- C version of the dot product kernel.
   This is the answer key the x86-64 version is checked against.

   MSVC compiles this to scalar movsd/mulsd/addsd: under the default
   /fp:precise it may not reorder a floating-point sum, so it cannot
   vectorise the loop. The assembly kernel adds the terms in the same
   order, which is why the two results match bit for bit. */

#include <stddef.h>

void dotprod_c(size_t n, const double *A, const double *B, double *sdot)
{
    double sum = 0.0;
    size_t i;

    for (i = 0; i < n; i++)
        sum += A[i] * B[i];

    *sdot = sum;
}
