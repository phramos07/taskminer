//
// objs/bookfilter_ispc_sse4.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#ifndef ISPC_OBJS_BOOKFILTER_ISPC_SSE4_H
#define ISPC_OBJS_BOOKFILTER_ISPC_SSE4_H

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

#ifndef __ISPC_STRUCT_String__
#define __ISPC_STRUCT_String__
struct String {
    int32_t length;
    int8_t * data;
};
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void bookfilter_lch(const struct String * page, const int32_t num_pages, const struct String &pattern, struct String * output);
    extern void bookfilter_ldp(const struct String * page, const int32_t num_pages, const struct String &pattern, struct String * output);
    extern void bookfilter_par(const struct String * page, const int32_t num_pages, const struct String &pattern, struct String * output);
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus

#endif // ISPC_OBJS_BOOKFILTER_ISPC_SSE4_H
