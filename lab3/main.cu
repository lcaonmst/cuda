#include <iostream>
#include <stdio.h>

__global__ void kernel() {
    printf("zrob cos prosze\n");
}

int main() {
    const int N = 10;

//    int* gpu_arr;
//    cudaMalloc(&gpu_arr, N * sizeof(int));
    kernel<<<1, 256>>>();
    std::cout << "siema";
}
