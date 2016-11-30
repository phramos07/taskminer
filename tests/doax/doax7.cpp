/**
 * Example of DOAX loop.
 * Author: Divino Cesar <divcesar@gmail.com>
 */
#include <iostream>
#include <stdio.h>
using namespace std;

class node {
public:
	int value;
	node* next;
}; 

int doax7(node* head) {
	int sum = 0;

	while (head != NULL) {
		sum = sum + head->value;
		head = head->next;
	}

	return sum;
}
