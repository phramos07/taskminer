/*
-- ----------------------------------- --
-- EQUIVALENCE OF TYPES:

   // Fundamental types default to varying
   // ref: http://ispc.github.io/ispc.html#uniform-and-varying-qualifiers
   int x;
   varying int x;

   // Pointed-to types default to uniform
   // Pointer types default to varying
   // ref: http://ispc.github.io/ispc.html#pointer-types
   void foo(uniform int x[]);
   void foo(uniform int * uniform x);

   void foo(int *x);
   void foo(uniform int *x);
   void foo(int * varying x);
   void foo(uniform int * varying x);

-- ----------------------------------- --
*/

// Bad!
varying int bad_dest[4] = {
	((8 - programIndex) * -10) - 1,
	((8 - programIndex) * -10) - 2,
	((8 - programIndex) * -10) - 3,
	((8 - programIndex) * -10) - 4
};

varying int bad_src[4] = {
	10 * (programIndex + 1) + 1,
	10 * (programIndex + 1) + 2,
	10 * (programIndex + 1) + 3,
	10 * (programIndex + 1) + 4
};

// Ugly -- but Good!
uniform int * varying ugly_dest = new uniform int[4];
ugly_dest[0] = ((8 - programIndex) * -10) - 1;
ugly_dest[1] = ((8 - programIndex) * -10) - 2;
ugly_dest[2] = ((8 - programIndex) * -10) - 3;
ugly_dest[3] = ((8 - programIndex) * -10) - 4;

uniform int * varying ugly_src = new uniform int[4];
ugly_src[0] = 10 * (programIndex + 1) + 1;
ugly_src[1] = 10 * (programIndex + 1) + 2;
ugly_src[2] = 10 * (programIndex + 1) + 3;
ugly_src[3] = 10 * (programIndex + 1) + 4;

// Good -- a.k.a., slightly less Ugly!
int *dest = new int[4];
dest[0] = ((8 - programIndex) * -10) - 1;
dest[1] = ((8 - programIndex) * -10) - 2;
dest[2] = ((8 - programIndex) * -10) - 3;
dest[3] = ((8 - programIndex) * -10) - 4;

int *src = new int[4];
src[0] = 10 * (programIndex + 1) + 1;
src[1] = 10 * (programIndex + 1) + 2;
src[2] = 10 * (programIndex + 1) + 3;
src[3] = 10 * (programIndex + 1) + 4;

// Dumping
print("\nBAD:\n");
for (int i = 0; i < 4; ++i) {
	print("DEST: %\tSRC: %\n", bad_dest[i], bad_src[i]);
}

print("\nUGLY:\n");
for (int i = 0; i < 4; ++i) {
	print("DEST: %\tSRC: %\n", ugly_dest[i], ugly_src[i]);
}

print("\nGOOD:\n");
for (int i = 0; i < 4; ++i) {
	print("DEST: %\tSRC: %\n", dest[i], src[i]);
}
