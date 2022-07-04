#define NO_FREETYPE
#include <pngwriter.h>
#include <iostream>
#include <chrono>


__device__
int compute_iter1(double x_pos, double y_pos, int ITER) {
    int res = 255;
    double a = x_pos;
    double b = y_pos;
    for (int j = 0; j < ITER; j++) {
        double new_a = a*a - b*b;
        double new_b = 2 * a * b;
        a = new_a + x_pos;
        b = new_b + y_pos;
        if (a*a + b*b >= 4) {
            res = j;
            break;
        }
    }
    return res;
}

#define update          tZx = Zx*Zx - Zy*Zy;    \
                        tZy = 2*Zx*Zy;          \
                        Zx = tZx + x_pos;       \
                        Zy = tZy + y_pos;       \
                        i++;

__device__
int compute_iter2(double x_pos, double y_pos, int ITER){

    int i = 0;
    double Zx = x_pos, Zy = y_pos, tZx, tZy;
    int IN=1;
//    uint poz = (blockIdx.x * blockDim.x) + threadIdx.x;
//    uint pion = (blockIdx.y * blockDim.y) + threadIdx.y;
        while ( (i<36) && IN ){
            update;
            if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        }
        while ( (i<42) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<48) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<55) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<63) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<73) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<84) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<97) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<111) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<127) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<147) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<168) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<194) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<222) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        while ( (i<255) && IN ){
            update;
        }
        if (IN) IN = ((Zx*Zx+Zy*Zy)<4) ? 1 : 0;
        if (IN) i++;

        
	return i;
}

__global__
void kernel(int* Mandel, int X0, int Y0, int X1, int Y1, int POZ, int PION, int ITER) {
    double x_step = (X1 - X0) / (double)(POZ - 1);
    double y_step = (Y1 - Y0) / (double)(PION - 1);
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= POZ * PION) {
        return;
    }

//    for (int i = myId; i < POZ * PION; i += STEP) {
        int x = myId % PION;
        int y = myId / PION;

        double x_pos = X0 + x_step * x;
        double y_pos = Y0 + y_step * y;
        Mandel[myId] = compute_iter1(x_pos, y_pos, ITER);
//    }
}

__global__
void kernel2(int* Mandel, int X0, int Y0, int X1, int Y1, int POZ, int PION, int ITER) {
    double x_step = (X1 - X0) / (double)(POZ - 1);
    double y_step = (Y1 - Y0) / (double)(PION - 1);

    int poz = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pion = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (poz >= POZ || pion >= PION) {
        return;
    }

    double x_pos = X0 + x_step * poz;
    double y_pos = Y0 + y_step * pion;
    Mandel[pion * POZ + poz] = compute_iter1(x_pos, y_pos, ITER);
}

void makePicture(int *Mandel,int width, int height, int MAX){
    
    float scale = 256.0/MAX;
    
    int red_value, green_value, blue_value;
    
    int MyPalette[33][3]={
        {255,255,255}, {255,0,255}, {248,0,240}, {240,0,224},
        {232,0,208}, {224,0,192}, {216,0,176}, {208,0,160},
        {200,0,144}, {192,0,128}, {184,0,112}, {176,0,96},
        {168,0,80},  {160,0,64},  {152,0,48},  {144,0,32},
        {136,0,16},  {128,0,0},   {120,16,0},  {112,32,0},
        {104,48,0},  {96,64,0},   {88,80,0},   {80,96,0},
        {72,112,0},  {64,128,0},  {56,144,0},  {48,160,0},
        {40,176,0},  {32,192,0},  {16,224,0},  {8,240,0}, {0,0,0}
    };
    
    FILE *f = fopen("Mandel.ppm", "wb");
    
    fprintf(f, "P3\n%i %i 255\n", width, height);
    printf("MAX = %d, scale %lf\n",MAX,scale);
    for (int j=(height-1); j>=0; j--) {
        for (int i=0; i<width; i++)
        {
            //if ( ((i%4)==0) && ((j%4)==0) ) printf("%d ",Mandel[j*width+i]);
            //red_value = (int) round(scale*(Mandel[j*width+i])/16);
            //green_value = (int) round(scale*(Mandel[j*width+i])/16);
            //blue_value = (int) round(scale*(Mandel[j*width+i])/16);
            int indx= (int) round(4.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            
            fprintf(f,"%d ",red_value);   // 0 .. 255
            fprintf(f,"%d ",green_value); // 0 .. 255
            fprintf(f,"%d ",blue_value);  // 0 .. 255
        }
        fprintf(f,"\n");
        //if ( (j%4)==0)  printf("\n");

    }
    fclose(f);
    
}




int main(int argc, char** argv) {
    double X0 = std::stod(std::string(argv[1]));
    double Y0 = std::stod(std::string(argv[2]));
    double X1 = std::stod(std::string(argv[3]));
    double Y1 = std::stod(std::string(argv[4]));
    int POZ = std::stoi(std::string(argv[5]));
    int PION = std::stoi(std::string(argv[6]));
    int ITER = std::stoi(std::string(argv[7]));
    
    int WATKI = 1024;
    int BLOKI = PION * POZ / WATKI + 1;
    
    int* Mandel;
    if (cudaMalloc(&Mandel, POZ * PION * sizeof(int))) {
        std::cout << "error in alloc\n";
        return 0;
    }
    float min_time = 99999;
    float avg_time = 0;

    dim3 threadsPerBlock(WATKI, 1, 1);
    dim3 numBlocks(BLOKI, 1, 1);

    // settings for cudaMandelbrot2D and cudaMandelbrot_steps
//    int block_width=4;
//    int block_height=4;
//    dim3 threadsPerBlock(block_width,block_height,1);
//    dim3 numBlocks(POZ/block_width+1,PION/block_height+1,1);

    for (int i = 0; i < 15; i++) {
        auto start = clock();
        kernel<<<numBlocks, threadsPerBlock>>>(Mandel, X0, Y0, X1, Y1, POZ, PION, ITER);
        cudaDeviceSynchronize();
        auto end = clock();
        auto diff = end - start;
        float t = std::chrono::duration <float, std::milli> (diff).count();

        printf("Computing #%d took %lf ms\n\n",i, t);
        min_time = min(min_time, t);
        avg_time += t;
    }
    avg_time /= 15;
    printf("Averege time: %lf ms\nMinimum time: %lf ms\n\n", avg_time, min_time);


    int* myMandel = new int[POZ * PION];

    if (cudaMemcpy(myMandel, Mandel, POZ * PION * sizeof(int), cudaMemcpyDeviceToHost)) {
        std::cout << "error in cpy\n";
        return 0;
    }
    makePicture(myMandel, POZ, PION, ITER);
    cudaFree(Mandel);
    delete[] myMandel;


}