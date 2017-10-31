# define vector_std_len	36
# define vector_std_inc 2

# define decl_vector(type) 													\
struct vector_##type {														\
																			\
	type * uniform data;													\
																			\
	uniform int empty;														\
	uniform int length;														\
	uniform int allocated;													\
																			\
};																			\
																			\
void vector_##type##_build(vector_##type& v) {								\
																			\
	v.data = uniform new type[vector_std_len];										\
																			\
	v.empty = vector_std_len;												\
	v.length = 0;															\
	v.allocated = vector_std_len;											\
																			\
}																			\
																			\
void vector_##type##_destroy(vector_##type& v) {							\
																			\
	delete[] v.data;														\
	v.data = NULL;															\
																			\
	v.empty = 0;															\
	v.length = 0;															\
	v.allocated = 0;														\
																			\
}																			\
																			\
void vector_##type##_push_back(vector_##type& v, const uniform type& e) {	\
																			\
	if (v.empty <= 0) {														\
																			\
		v.empty = v.allocated << vector_std_inc;							\
																			\
		type * uniform data = v.data;										\
		type * uniform tmp = uniform new type[v.empty + v.allocated];		\
		foreach (i = 0 ... v.length) tmp[i] = data[i];						\
																			\
		delete[] v.data;													\
		v.data = tmp;														\
																			\
	}																		\
																			\
	v.data[v.length] = e;													\
	++v.length; --v.empty;													\
                                                                              \
}                                                                             \
                                                                              \
uniform type& vector_##type##_at(vector_##type& v, const uniform int pos) {   \
	 return v.data[pos];                                                      \
}                                                                             \

decl_vector(int)
decl_vector(int8)
decl_vector(int16)
decl_vector(int32)
decl_vector(int64)
decl_vector(float)
decl_vector(double)


void bar(uniform int& a, uniform int& p, uniform float& q) {
	++a; ++p; ++q;
}

export void fooo() {

	vector_int v;
	vector_int_build(v);

	vector_int_at(v, 0);

	print("length: %\n", v.length);

	uniform int z = 123;
	varying int m = programIndex;
	varying float n = programIndex;

	print("z --> %\n", z);
	print("m --> %\n", m);
	print("n --> %\n", n);

	ldp bar(z, m, n);

	print("z --> %\n", z);
	print("m --> %\n", m);
	print("n --> %\n", n);

}


task void f(varying int& data) {
	if (programIndex % 2 == 0) data = data + programIndex;
	else data = data - programIndex;
}

void f_matrix(varying int data[], uniform int length) {
	for (uniform int i = 0; i < length; ++i) launch f(data[i]);
	sync;
}

export void proc_matrix() {

	varying int data[10];
	uniform int length = 10;

	for (uniform int i = 0; i < length; ++i) data[i] = 0;
	f_matrix(data, length);

	for (uniform int i = 0; i < length; ++i) {
		print("data[%]: %\n", i, data[i]);
	}

}

// ISPC file: helloworld.ispc
export void hello_world() {
	print ("Warplength:%\n", programCount);
	print ("Activethreads(all):%\n", programIndex);
	if (programIndex % 2 == 0){
		print ("Activethreads(evenids):%\n", programIndex);
	} else {
		print ("Activethreads(oddids):%\n", programIndex);
	}
}


export void mem_access(uniform int * uniform data[], uniform int length) {

	for (uniform int i = 0; i < length; ++i) {
		// for (varying int j = programIndex; j < length; ++j) {
		for (varying int j = programIndex; j < length; j += programCount) {
			print ("data[%] = %\n", i, data[i][j]);
		}
	}

}

void simd_reset(uniform int data[], uniform int length) {
	print ("%\n", programIndex);
	for (varying int i = programIndex; i < length; i += programCount) data[i] = 0;
}

// foreach(i = BEG ... END):
// uses the threads available and guarantees that every step will run
// w.r.t. loop induction variable (none is skept due to inactive threads)

// foreach_active(i):
// restricts the control flow to the specific i-th active thread, i.e.,
// anything within this loop will be only executed by thread i

// common bug is to start loop induction var with 0 instead of programIndex

export void toy(uniform int * uniform data[], uniform int length) {
	if (programIndex % 2 == 0) return;
	for (uniform int i = 0; i < length; i += 2) simd_reset(data[i], length);
}

/*
void bar1(uniform int a[], uniform int b) {
	a[programIndex] = b;
	print("-> tid: %\n   a: %\n   b: %\n", programIndex, a[programIndex], b);
}

void bar2(uniform int * uniform a, varying int b) {
	a[programIndex] = b;
	print("-> tid: %\n   a: %\n   b: %\n", programIndex, a[programIndex], b);
}

void bar3(varying int a, uniform int b) {
	print("-> tid: %\n   a: %\n   b: %\n", programIndex, a, b);
}

void bar4(varying int a, varying int b) {
	print("-> tid: %\n   a: %\n   b: %\n", programIndex, a, b);
}
*/

/*
export void foo(const uniform int num_pages, const uniform int length) {

	uniform int x[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	varying int y = programIndex;

	ldp bar1(x, y);		// Valid coercion from varying Y to uniform B
	ldp bar2(x, y);		// Simple forwarding of values
	// ldp bar3(x, y);	// Invalid coercion from uniform X to varying A
	// ldp bar4(x, y);	// Same invalid coercion of uniform X to varying A

*/

/*
	foreach (i = 0 ... num_pages) {

		uniform int id[programCount] = {10};
		bar(&id[programIndex]);

		for (uniform int j = 0; j < length; ++j) {

			print("Bla! % ? % -> %\n", j, i, programIndex);

			uniform int id[programCount] = {10};
			ldp bar(&id[programIndex]);

			print("% came out\n", id[programIndex]);

			if (i == j) {
				print("going home %\n", j);
				break;
			}

		}

	}
		*/

//}
