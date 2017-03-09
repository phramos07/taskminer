//
// objs/dfs_ispc_avx2.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#ifndef ISPC_OBJS_DFS_ISPC_AVX2_H
#define ISPC_OBJS_DFS_ISPC_AVX2_H

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

#ifndef __ISPC_STRUCT_Graph__
#define __ISPC_STRUCT_Graph__
struct Graph {
    int32_t num_nodes;
    int32_t num_edges;
    struct Node * node;
    struct Edge * edge;
};
#endif

#ifndef __ISPC_STRUCT_Node__
#define __ISPC_STRUCT_Node__
struct Node {
    bool visited;
    int32_t length;
    struct Edge * edge;
    int32_t * distance;
};
#endif

#ifndef __ISPC_STRUCT_Edge__
#define __ISPC_STRUCT_Edge__
struct Edge {
    int32_t node;
    int32_t weight;
};
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void graph_build(struct Graph &graph, struct Node * node, const int32_t &num_nodes, struct Edge * edge, const int32_t &num_edges, int32_t * distance);
    extern void graph_dfs(struct Graph &graph, int32_t root);
    extern void graph_dfs_par(struct Graph &graph, int32_t root);
    extern void graph_dump(struct Graph &graph);
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus

#endif // ISPC_OBJS_DFS_ISPC_AVX2_H
