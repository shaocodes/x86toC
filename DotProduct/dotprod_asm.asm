; dotprod_asm.asm -- x86-64 dot product kernel (MASM / ml64)
;
;   void dotprod_asm(size_t n, const double *A, const double *B, double *sdot);
;
;   RCX = n, RDX = A, R8 = B, R9 = sdot   (Microsoft x64 convention)
;
; Scalar SIMD registers:    XMM0 (running sum), XMM1 (product)
; Scalar SIMD instructions: MOVSD, MULSD, ADDSD
;
; Every floating-point operation is scalar double-precision. There are no
; packed instructions, and the accumulator is zeroed with a MOVSD load of 0.0
; rather than the usual XORPD, so every instruction touching XMM state is a
; scalar SIMD floating-point instruction.
;
; XMM0/XMM1 and RAX are volatile under this ABI, so this leaf function needs
; no prologue, no epilogue and no stack.

                .const
                ALIGN   8
DZERO           REAL8   0.0

                .code
                PUBLIC  dotprod_asm

dotprod_asm     PROC

                movsd   xmm0, QWORD PTR [DZERO] ; sum = 0.0
                xor     rax, rax                ; i = 0
                test    rcx, rcx                ; n == 0?
                jz      store_result

main_loop:
                movsd   xmm1, QWORD PTR [rdx + rax*8]   ; A[i]
                mulsd   xmm1, QWORD PTR [r8  + rax*8]   ; A[i] * B[i]
                addsd   xmm0, xmm1                      ; sum += A[i] * B[i]
                inc     rax
                cmp     rax, rcx
                jb      main_loop

store_result:
                movsd   QWORD PTR [r9], xmm0    ; *sdot = sum
                ret

dotprod_asm     ENDP

                END
