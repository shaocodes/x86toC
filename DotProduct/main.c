/* main.c -- driver for the C and x86-64 dot product kernels.
   Times only the kernel call, 20 runs each, and checks the assembly result
   against the C result. */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#define NUM_RUNS  20
#define MAX_SIZES 8

/* RCX = n, RDX = A, R8 = B, R9 = sdot  (Microsoft x64 convention) */
void dotprod_c  (size_t n, const double *A, const double *B, double *sdot);
void dotprod_asm(size_t n, const double *A, const double *B, double *sdot);

static const int DEFAULT_EXP[] = { 20, 24, 28, 29 };   /* 2^30 would need 16 GiB */
#define NUM_DEFAULT (sizeof DEFAULT_EXP / sizeof DEFAULT_EXP[0])

static volatile double g_sink;   /* keeps the kernel calls from being optimised out */
static double g_freq;
static unsigned long long g_rng = 0x9E3779B97F4A7C15ULL;

static double now_ms(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / g_freq;
}

/* xorshift64*, seeded so every run uses the same data */
static double rng_unit(void)
{
    unsigned long long x = g_rng;
    x ^= x >> 12;  x ^= x << 25;  x ^= x >> 27;
    g_rng = x;
    return (double)((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

static unsigned long long bits(double d)
{
    unsigned long long u;
    memcpy(&u, &d, sizeof u);
    return u;
}

typedef struct {
    double c_ms, a_ms;      /* average kernel time */
    double c_res, a_res;    /* sdot from each kernel */
} result_t;

/* Times both kernels on the same data. They are interleaved and the call order
   alternates each run, so neither one gains from following the other. */
static result_t bench(size_t n, const double *A, const double *B)
{
    result_t r = { 0.0, 0.0, 0.0, 0.0 };
    double t0, t1, t2;
    int i;

    dotprod_c  (n, A, B, &r.c_res);         /* untimed warm-up */
    dotprod_asm(n, A, B, &r.a_res);
    g_sink += r.c_res + r.a_res;

    for (i = 0; i < NUM_RUNS; i++) {
        if (i & 1) {
            t0 = now_ms();  dotprod_asm(n, A, B, &r.a_res);
            t1 = now_ms();  dotprod_c  (n, A, B, &r.c_res);
            t2 = now_ms();
            r.a_ms += t1 - t0;  r.c_ms += t2 - t1;
        } else {
            t0 = now_ms();  dotprod_c  (n, A, B, &r.c_res);
            t1 = now_ms();  dotprod_asm(n, A, B, &r.a_res);
            t2 = now_ms();
            r.c_ms += t1 - t0;  r.a_ms += t2 - t1;
        }
        g_sink += r.c_res + r.a_res;
    }

    r.c_ms /= NUM_RUNS;
    r.a_ms /= NUM_RUNS;
    return r;
}

/* Small run that can be checked by hand. */
static void sanity_demo(void)
{
    enum { N = 10 };
    double A[N], B[N], c = 0.0, a = 0.0, sum = 0.0;
    int i;

    g_rng = 20240607ULL;
    for (i = 0; i < N; i++) {
        A[i] = rng_unit() * 10.0 - 5.0;
        B[i] = rng_unit() * 10.0 - 5.0;
    }

    printf("SANITY DEMO (n = %d)\n", N);
    printf("  %3s %14s %14s %16s %16s\n", "i", "A[i]", "B[i]", "A[i]*B[i]", "running sum");
    for (i = 0; i < N; i++) {
        sum += A[i] * B[i];
        printf("  %3d %14.9f %14.9f %16.9f %16.9f\n", i, A[i], B[i], A[i] * B[i], sum);
    }

    dotprod_c  (N, A, B, &c);
    dotprod_asm(N, A, B, &a);
    printf("\n  C   sdot = %.15f  [0x%016llX]\n", c, bits(c));
    printf("  ASM sdot = %.15f  [0x%016llX]\n", a, bits(a));
    printf("  -> x86-64 output is %s\n\n",
           bits(c) == bits(a) ? "CORRECT (bit-for-bit identical to C)" : "INCORRECT");
}

static int run_size(int e, double *out_c, double *out_a, int *ok)
{
    size_t n = (size_t)1 << e, bytes = n * sizeof(double), i;
    double *A, *B;
    result_t r;

    printf("n = 2^%d = %llu elements  (%.0f MiB total)\n",
           e, (unsigned long long)n, (double)(2 * bytes) / (1024.0 * 1024.0));

    A = _aligned_malloc(bytes, 64);
    B = _aligned_malloc(bytes, 64);
    if (!A || !B) {
        printf("  allocation failed -- not enough memory, skipping\n\n");
        _aligned_free(A);
        _aligned_free(B);
        return 0;
    }

    g_rng = 0xC0FFEEULL + (unsigned)e;
    for (i = 0; i < n; i++) {
        A[i] = rng_unit() * 2.0 - 1.0;
        B[i] = rng_unit() * 2.0 - 1.0;
    }

    r = bench(n, A, B);
    _aligned_free(A);
    _aligned_free(B);

    *ok = (bits(r.c_res) == bits(r.a_res));
    printf("  C   sdot = % .15e  [0x%016llX]\n", r.c_res, bits(r.c_res));
    printf("  ASM sdot = % .15e  [0x%016llX]\n", r.a_res, bits(r.a_res));
    printf("  difference = %.1e  -> x86-64 output is %s\n",
           fabs(r.a_res - r.c_res),
           *ok ? "CORRECT (bit-for-bit identical to C)" : "INCORRECT");
    printf("  C %.4f ms   ASM %.4f ms   (%.3fx)\n\n", r.c_ms, r.a_ms, r.c_ms / r.a_ms);

    *out_c = r.c_ms;
    *out_a = r.a_ms;
    return 1;
}

int main(int argc, char **argv)
{
    int exps[MAX_SIZES], ran[MAX_SIZES], ok[MAX_SIZES];
    double c_ms[MAX_SIZES], a_ms[MAX_SIZES];
    int ns = 0, i, all_ok = 1;
    LARGE_INTEGER f;

    QueryPerformanceFrequency(&f);
    g_freq = (double)f.QuadPart;

    /* Keeps the timings steady: no core migration, not preempted mid-run. */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);

    for (i = 1; i < argc && ns < MAX_SIZES; i++) {
        int e = atoi(argv[i]);
        if (e >= 1 && e <= 31) exps[ns++] = e;
    }
    if (ns == 0)
        for (i = 0; i < (int)NUM_DEFAULT; i++) exps[ns++] = DEFAULT_EXP[i];

    printf("DOT PRODUCT: C vs x86-64 ASSEMBLY  (scalar SIMD: MOVSD/MULSD/ADDSD)\n");
#ifdef _DEBUG
    printf("Build: DEBUG (/Od)");
#else
    printf("Build: RELEASE (/O2)");
#endif
    printf("   %d timed runs per kernel, kernel call only\n\n", NUM_RUNS);

    sanity_demo();

    for (i = 0; i < ns; i++) {
        ran[i] = run_size(exps[i], &c_ms[i], &a_ms[i], &ok[i]);
        if (ran[i] && !ok[i]) all_ok = 0;
    }

    printf("SUMMARY (average of %d runs, milliseconds)\n", NUM_RUNS);
    printf("  %-8s %12s %12s %10s %9s\n", "n", "C", "ASM", "speedup", "correct");
    for (i = 0; i < ns; i++) {
        if (!ran[i]) {
            printf("  2^%-6d %12s %12s %10s %9s\n", exps[i], "-", "-", "-", "SKIPPED");
            continue;
        }
        printf("  2^%-6d %12.4f %12.4f %9.3fx %9s\n",
               exps[i], c_ms[i], a_ms[i], c_ms[i] / a_ms[i], ok[i] ? "YES" : "NO");
    }
    printf("\n%s\n", all_ok ? "x86-64 kernel output is CORRECT for every size tested"
                            : "x86-64 kernel output is INCORRECT");
    return all_ok ? 0 : 1;
}
