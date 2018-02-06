#ifdef __cplusplus
extern "C" {
#endif

//===============================================================================================================================================================================================================200
//	TIMER CODE
//===============================================================================================================================================================================================================200

//======================================================================================================================================================150
//	INCLUDE/DEFINE
//======================================================================================================================================================150

#include <stdlib.h>


//======================================================================================================================================================150
//	FUNCTIONS
//======================================================================================================================================================150

//====================================================================================================100
//	DISPLAY TIME
//====================================================================================================100

 // Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

//===============================================================================================================================================================================================================200
//	END TIMER CODE
//===============================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif

