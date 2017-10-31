void func(int a, int b){
  int n[100];
  long long int AI1[4];
  AI1[0] = b > 0;
  AI1[1] = (AI1[0] ? b : 0);
  AI1[2] = 4 * AI1[1];
  AI1[3] = 4 + AI1[2];
  acc_create((void*) n, AI1[3]);
  acc_copyin((void*) n, AI1[3]);
  for(int i = 0; i < b; i++){
  	  n[i+1] = n[i+1] + 1;
  }
acc_copyout_and_keep((void*) n, AI1[3]);
}

