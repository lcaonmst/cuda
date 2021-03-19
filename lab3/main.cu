#include <iostream>
#include <stdio.h>

__global__ kernel() {
    while (true) {};
}

int main() {
    const int N = 10;

//    int* gpu_arr;
//    cudaMalloc(&gpu_arr, N * sizeof(int));
    kernel<<<1, 256>>>();
    cout << "siema";
}
