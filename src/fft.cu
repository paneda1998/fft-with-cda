//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"
#include "math.h"
#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define Pi 3.141592654f
#define pi 3.141592654f

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

//-----------------------------------------------------------------------------


__device__ unsigned int reverse4(register unsigned int x,const unsigned int M)
	
{
   // x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    x = ((x >> 16) | (x << 16));
    x>>=(32-M);
return x;


}
__device__ unsigned int reverse2(register unsigned int x,const unsigned int M)
	
{
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
    x = ((x >> 16) | (x << 16));
    x>>=(32-M);
return x;


}

__global__ void reverseb2(float* input,float* input1,const unsigned int N, unsigned int M) //radix-2
{

	float temp;
	float temp1;
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
  
unsigned int reverse_num=reverse2(i,M);
    if(i<reverse_num)
{
temp=input[reverse_num];
temp1=input1[reverse_num];

input[reverse_num]=input[i];
input1[reverse_num]=input1[i];

input[i]=temp;
input1[i]=temp1;
}

}
__global__ void reverseb4(float* input,float* input1,const unsigned int N, unsigned int M) //radix-2
{

	float temp;
	float temp1;
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
      //int i =  bx * 1024  + tx;
unsigned int reverse_num=reverse4(i,M);

    if(i<reverse_num)
{
temp=input[reverse_num];
temp1=input1[reverse_num];

input[reverse_num]=input[i];
input1[reverse_num]=input1[i];

input[i]=temp;
input1[i]=temp1;
}

}



__global__ void kernelFunc(float* x_r_d, float* x_i_d ,const unsigned int N, unsigned int M) //radix-2
{
//int i = (by * gridDim.x + bx) * blockDim.x  + tx;
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
	int S = M;
	float WnR;                   
	float WnI;
    //int i =  bx * 1024  + tx;
	float temp_r_1, temp_r_2, temp_i_1, temp_i_2;
	
	temp_r_1 = x_r_d[i+(i/S)*S];         // in each step every butterfly take two inputs 
	temp_r_2 = x_r_d[i+(i/S)*S+(S)];
	
	WnR = cos(-1 * (2 * pi) * ((i*(N/(2*S)))-(i/S)*(N/2)) / (N));      //  wn = e ^ (-j * 2 * pi*  k / N)
	WnI = sin(-1 * (2 * pi) * ((i*(N/(2*S)))-(i/S)*(N/2)) / (N));	
	
	temp_i_1 = x_i_d[i+(i/S)*S];
	temp_i_2 = x_i_d[i+(i/S)*S+(S)];
   
	x_r_d[i+(i/S)*S] = temp_r_1 + WnR * temp_r_2 - WnI * temp_i_2;
	x_i_d[i+(i/S)*S] = temp_i_1 + WnR * temp_i_2 + WnI * temp_r_2;
	
	x_r_d[i+(i/S)*S+(S)] = temp_r_1 - WnR * temp_r_2 + WnI * temp_i_2;
	x_i_d[i+(i/S)*S+(S)] = temp_i_1 - WnR * temp_i_2 - WnI * temp_r_2;		
	
	
}




__global__ void kernelFunc2(float* x_r_d, float* x_i_d, const unsigned int N, const unsigned int M) //radix-4
{
	//...
	int S = M;
    //int i = (by * gridDim.x + bx) * blockDim.x  + tx;
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
	float temp_r_1, temp_r_2, temp_i_1, temp_i_2, temp_r_4, temp_r_3, temp_i_4, temp_i_3;	
	float v_i_1, v_i_2, v_i_3, v_i_4, v_r_1, v_r_2, v_r_3, v_r_4;
	float angle  = -2*pi*(i%S) / (S*4);	
	
	
	temp_r_1 = x_r_d[((i/S)*(4*S)+(i%S))];         // in each step every butterfly take two inputs 
	temp_r_2 = x_r_d[((i/S)*(4*S)+(i%S))+(S)];
	temp_r_3 = x_r_d[((i/S)*(4*S)+(i%S)) + 2 * (S)];
	temp_r_4 = x_r_d[((i/S)*(4*S)+(i%S)) + 3 * (S)];
	
	temp_i_1 = x_i_d[((i/S)*(4*S)+(i%S))];
	temp_i_2 = x_i_d[((i/S)*(4*S)+(i%S)) + (S)];
	temp_i_3 = x_i_d[((i/S)*(4*S)+(i%S)) + 2 * (S)];
	temp_i_4 = x_i_d[((i/S)*(4*S)+(i%S)) + 3 * (S)];	
	
	v_r_1 = temp_r_1;
	v_i_1 = temp_i_1;
	
	v_r_2 = temp_r_2 * cos(angle) - temp_i_2 * sin(angle);
	v_i_2 = temp_r_2 * sin(angle) + temp_i_2 * cos(angle);

	v_r_3 = temp_r_3 * cos(2*angle) - temp_i_3 * sin(2*angle);
	v_i_3 = temp_r_3 * sin(2*angle) + temp_i_3 * cos(2*angle);

	v_r_4 = temp_r_4 * cos(3*angle) - temp_i_4 * sin(3*angle);
	v_i_4 = temp_r_4 * sin(3*angle) + temp_i_4 * cos(3*angle);	
	
	//int index = (i/S)*S*4+i%S;
	//__syncthread;
	x_r_d[(i/S)*S*4+i%S] = v_r_1 + v_r_2 + v_r_3 + v_r_4;
	x_i_d[(i/S)*S*4+i%S] = v_i_1 + v_i_2 + v_i_3 + v_i_4;
	
	x_r_d[(i/S)*S*4+i%S + S] = v_r_1 + v_i_2 - v_r_3 - v_i_4;
	x_i_d[(i/S)*S*4+i%S + S] = v_i_1 - v_r_2 - v_i_3 + v_r_4;
	
	x_r_d[(i/S)*S*4+i%S + 2 * S] = v_r_1 - v_r_2 + v_r_3 - v_r_4;
	x_i_d[(i/S)*S*4+i%S + 2 * S] = v_i_1 - v_i_2 + v_i_3 - v_i_4;
	
	x_r_d[(i/S)*S*4+i%S + 3 * S] = v_r_1 - v_i_2 - v_r_3 + v_i_4;
	x_i_d[(i/S)*S*4+i%S + 3 * S] = v_i_1 + v_r_2 - v_i_3 - v_r_4;
	
}








void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{
 
int S=0;

	

	dim3 dimGrid1((N / (512*512)), 32, 32);
	dim3 dimBlock1(1024/4, 1, 1);


	if((M==24) | (M==26))
{
	dim3 dimGrid3((N / (1024 *256*4)), 32, 32);

	dim3 dimBlock3(1024/4, 1, 1);
        reverseb4 <<<  dimGrid1, dimBlock1  >>>(x_r_d, x_i_d, N, M);
	for ( S = 1; S < N; S*=4)  //  stage = log S
	{
	
	    kernelFunc2 <<< dimGrid3, dimBlock3 >>>(x_r_d, x_i_d, N, S);

		
		}

}
else
{
	dim3 dimGrid((N / (1024 *256*2)), 32, 32);
	dim3 dimBlock(1024/4, 1, 1);
        reverseb2 <<<  dimGrid1, dimBlock1  >>>(x_r_d, x_i_d, N, M);
	for ( S = 1; S < N; S*=2)  //  stage = log S
	{
	
	    kernelFunc <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, N, S);

		
		}



}

	
}
