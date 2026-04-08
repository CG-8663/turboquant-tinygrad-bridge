/**
 * Quick check: compare our PCG32 + Box-Muller output to NumPy's default_rng(42).
 * Build: cc -o rng_check benchmarks/rng_parity_check.c -lm && ./rng_check
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

typedef struct { uint64_t state; uint64_t inc; } pcg32;

static uint32_t pcg32_next(pcg32 *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(pcg32 *rng) {
    return (double)pcg32_next(rng) / 4294967296.0;
}

static double pcg32_normal(pcg32 *rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(void) {
    pcg32 rng;
    rng.state = 42;
    rng.inc = (42ULL << 1) | 1;
    pcg32_next(&rng);
    pcg32_next(&rng);

    printf("C PCG32+BoxMuller seed=42, first 10 standard_normal:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] %.15f\n", i, pcg32_normal(&rng));
    }
    return 0;
}
