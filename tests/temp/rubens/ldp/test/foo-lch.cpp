# include <string>
# include <vector>
# include <iostream>

# include "foo_ispc.h"


int main(int argc, char *argv[]) {

	ispc::proc_matrix();

	ispc::hello_world();

	int len = 8, *data[len];
	for (int i = 0; i < len; ++i) {
		data[i] = new int[len];
		for (int j = 0; j < len; ++j) data[i][j] = 10 * i + j;
	}

	ispc::mem_access(data, len);
	ispc::toy(data, len);
	ispc::mem_access(data, len);

}
