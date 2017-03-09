//
// objs/quicksort_ispc_avx.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#ifndef ISPC_OBJS_QUICKSORT_ISPC_AVX_H
#define ISPC_OBJS_QUICKSORT_ISPC_AVX_H

#include <stdint.h>



#ifdef __cplusplus
namespace ispc { /* namespace */
#endif // __cplusplus

#ifndef __ISPC_ALIGN__
#if defined(__clang__) || !defined(_MSC_VER)
// Clang, GCC, ICC
#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))
#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)
#else
// Visual Studio
#define __ISPC_ALIGN__(s) __declspec(align(s))
#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct
#endif
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void mergesort_lch(int32_t * array, int32_t len);
    extern void mergesort_lch_bi(int32_t * array, int32_t len);
    extern void mergesort_ldp(int32_t * array, int32_t len);
    extern void mergesort_ldp_bi(int32_t * array, int32_t len);
    extern void quicksort_lch(int32_t * array, int32_t len);
    extern void quicksort_lch_bi(int32_t * array, int32_t len);
    extern void quicksort_ldp(int32_t * array, int32_t len);
    extern void quicksort_ldp_bi(int32_t * array, int32_t len);
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus

#endif // ISPC_OBJS_QUICKSORT_ISPC_AVX_H
