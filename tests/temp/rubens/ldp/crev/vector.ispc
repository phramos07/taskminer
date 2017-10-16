# include "constants.ispc"


# ifndef CREV_VECTOR
# define CREV_VECTOR

# define crev_vector(type)                                                    \
struct vector_##type {                                                        \
                                                                              \
	uniform type * uniform data;                                              \
                                                                              \
	uniform int empty;                                                        \
	uniform int length;                                                       \
	uniform int allocated;                                                    \
                                                                              \
};	                                                                          \
	                                                                          \
export void vector_##type##_build(uniform vector_##type& v,                   \
	uniform int len = vector_std_len) {                                       \
                                                                              \
	v.data = uniform new type[len];                                           \
                                                                              \
	v.empty = len;                                                            \
	v.length = 0;                                                             \
	v.allocated = len;                                                        \
                                                                              \
}	                                                                          \
	                                                                          \
void vector_##type##_destroy(uniform vector_##type& v) {                      \
                                                                              \
	delete[] v.data;                                                          \
	v.data = NULL;                                                            \
                                                                              \
	v.empty = 0;                                                              \
	v.length = 0;                                                             \
	v.allocated = 0;                                                          \
                                                                              \
}	                                                                          \
	                                                                          \
export void vector_##type##_push_back(uniform vector_##type& v,               \
	const uniform type& e) {                                                  \
                                                                              \
	if (v.empty <= 0) {                                                       \
                                                                              \
		v.empty = v.allocated << vector_std_inc;                              \
                                                                              \
		type * uniform data = v.data;                                         \
		type * uniform tmp = uniform new type[v.empty + v.allocated];         \
		for (uniform int i = 0; i < v.length; ++i) tmp[i] = data[i];          \
                                                                              \
		delete[] v.data;                                                      \
		v.data = tmp;                                                         \
                                                                              \
	}                                                                         \
                                                                              \
	v.data[v.length] = e;                                                     \
	++v.length; --v.empty;                                                    \
                                                                              \
}                                                                             \
                                                                              \
export void vector_##type##_pop_back(uniform vector_##type& v) {              \
	--v.length; ++v.empty;                                                    \
}                                                                             \
                                                                              \
export uniform type vector_##type##_assign(uniform vector_##type& v,          \
	const uniform int pos, uniform const type& value) {                       \
	return (v.data[pos] = value);                                             \
}                                                                             \
                                                                              \
export uniform type vector_##type##_at(uniform vector_##type& v,              \
	const uniform int pos) {                                                  \
	return v.data[pos];                                                       \
}                                                                             \

crev_vector(int)
crev_vector(int8)
crev_vector(int16)
crev_vector(int32)
crev_vector(int64)
crev_vector(float)
crev_vector(double)


# endif
