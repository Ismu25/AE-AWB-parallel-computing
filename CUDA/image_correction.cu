#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

// includes, project
 #include <cuda.h>
 #include <cuda_runtime.h>

// ayuda con los ejemplos
// These are CUDA Helper functions for initialization and error checking
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

////////////////////////////////////////////////////////////////////////////////

// includes, kernels
#include "image_correction_kernel.cu"

// Tamaño en bytes de un pixel
#define PIXEL_SIZE 4
// Número de canales esperado por el programa
#define CHANNEL_NUMBER 4
// Valor máximo del canal de color - 2^8-1
#define MAX_COLOR_CHANNEL_VALUE 255
// tipo de procesamiento
#define PROCESS_COMPLETE    0
#define PROCESS_COLOR       1
#define PROCESS_EXPOSITION  2
// Número máximo de threads
#define MAX_THREADS 32

bool verbose = 0;
cudaError_t err = cudaSuccess;

////////////////////////////////////////////////////////////////////////////////
uint8_t * processBuffer(int width, int height, uint8_t* image1, uint8_t* image2, int number_threads, int padding_pix, int type) {
    
    dim3 grid( (width  + number_threads - 1) / number_threads,
               (height + number_threads - 1) / number_threads);
    dim3 block(number_threads, number_threads);

    int number_blocks = grid.x * grid.y;
    if(verbose)
        printf("Grid X: %d, Grid Y: %d, Block X: %d, Block Y:%d, Padding_pix:%d\n", grid.x, grid.y, block.x, block.y, padding_pix);
    

    Pixel * image1_d;
    Pixel * image2_d;
    Pixel * imageRes, *imageRes_d, *imageStep_d;

    imageRes = (Pixel *) malloc(sizeof(Pixel) * width * height);
    
    double sum;
    double dev_red_g, dev_green_g, dev_blue_g;
    double mean_red_g, mean_green_g, mean_blue_g;

    double * sum_y; 
    double * mean_red_d, *mean_green_d, *mean_blue_d;
    double * mean_red_dg, *mean_green_dg, *mean_blue_dg;
    double * dev_red_d, *dev_green_d, *dev_blue_d;
    double * dev_red_dg, *dev_green_dg, *dev_blue_dg;    

    // Reserving Memory
    if(verbose)
        printf("Allocating memory\n");
    
    checkCudaErrors(cudaMalloc((void **) &image1_d, sizeof(Pixel) * (width + padding_pix) * height));
    checkCudaErrors(cudaMalloc((void **) &imageRes_d, sizeof(Pixel) * width * height));
    
    if(image2 != NULL)
        checkCudaErrors(cudaMalloc((void **) &image2_d, sizeof(Pixel) * (width + padding_pix) * height));
    
    if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
        checkCudaErrors(cudaMalloc((void **) &sum_y, sizeof(double)));
    
    if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
        checkCudaErrors(cudaMalloc((void **) &mean_red_d,   sizeof(double) * number_blocks));
        checkCudaErrors(cudaMalloc((void **) &mean_green_d, sizeof(double) * number_blocks));
        checkCudaErrors(cudaMalloc((void **) &mean_blue_d,  sizeof(double) * number_blocks));

        checkCudaErrors(cudaMalloc((void **) &dev_red_d,   sizeof(double) * number_blocks));
        checkCudaErrors(cudaMalloc((void **) &dev_green_d, sizeof(double) * number_blocks));
        checkCudaErrors(cudaMalloc((void **) &dev_blue_d,  sizeof(double) * number_blocks));

        checkCudaErrors(cudaMalloc((void **) &dev_red_dg,   sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &dev_green_dg, sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &dev_blue_dg,  sizeof(double)));

        checkCudaErrors(cudaMalloc((void **) &mean_red_dg,   sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &mean_green_dg, sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &mean_blue_dg,  sizeof(double)));
    }


    // Sending information
    if(verbose)
        printf("Sending information to the kernel\n");

    checkCudaErrors(cudaMemcpy(image1_d, image1, sizeof(Pixel) * (width + padding_pix) * height, cudaMemcpyHostToDevice));

    
    if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
        checkCudaErrors(cudaMemset(sum_y, 0,  sizeof(double)));
    
    if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
        checkCudaErrors(cudaMemset(mean_red_d, 0,   sizeof(double) * number_blocks));
        checkCudaErrors(cudaMemset(mean_green_d, 0, sizeof(double) * number_blocks));
        checkCudaErrors(cudaMemset(mean_blue_d, 0,  sizeof(double) * number_blocks));

        checkCudaErrors(cudaMemset(dev_red_d, 0,   sizeof(double) * number_blocks));
        checkCudaErrors(cudaMemset(dev_green_d, 0, sizeof(double) * number_blocks));
        checkCudaErrors(cudaMemset(dev_blue_d, 0,  sizeof(double) * number_blocks));

        checkCudaErrors(cudaMemset(dev_red_dg, 0,   sizeof(double)));
        checkCudaErrors(cudaMemset(dev_green_dg, 0, sizeof(double)));
        checkCudaErrors(cudaMemset(dev_blue_dg, 0,  sizeof(double)));

        checkCudaErrors(cudaMemset(mean_red_dg, 0,   sizeof(double)));
        checkCudaErrors(cudaMemset(mean_green_dg, 0, sizeof(double)));
        checkCudaErrors(cudaMemset(mean_blue_dg, 0,  sizeof(double)));
    }
    // Calculing
    if(verbose)
        printf("Starting Processing in the GPU\n");
    
    // AE
    if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE){
        if(verbose)
            printf("Starting Exposition Correction\nCalculating Exposure\n");

        calculate_sum<<<grid, block, sizeof(double) * number_threads * number_threads >>>(image1_d, sum_y, width, height, padding_pix);
        if(image2 != NULL)
            checkCudaErrors(cudaMemcpy(image2_d, image2,  sizeof(Pixel) * (width + padding_pix) * height, cudaMemcpyHostToDevice));
        
        cudaDeviceSynchronize();

        if(image2 != NULL) {
            if(verbose)
                printf("Applaying Fusion\n");
            apply_fusion<<<grid, block>>>(image1_d, image2_d, imageRes_d, width, height, padding_pix);
            cudaDeviceSynchronize();
            imageStep_d = imageRes_d;
        } else {
            imageStep_d = image1_d;
        }
    } else {
        imageStep_d = image1_d;
    }

    // AWB
    if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
        if(verbose)
            printf("Calculating mean\n");

        calculate_mean<<<grid, block, 3 *sizeof(double) * number_threads * number_threads >>>(imageStep_d, width, height, imageStep_d == image1_d ? padding_pix : 0 , mean_red_d, mean_green_d, mean_blue_d);
        cudaDeviceSynchronize();
        
        if(verbose)
            printf("Calculating deviation\n");
        calculate_dev<<<grid, block, 3 *sizeof(double) * number_threads * number_threads>>>(imageStep_d, width, height, imageStep_d == image1_d ? padding_pix : 0 , 
                                                                        mean_red_d, mean_green_d, mean_blue_d,
                                                                        dev_red_d,  dev_green_d,  dev_blue_d,
                                                                        dev_red_dg, dev_green_dg, dev_blue_dg);
        cudaDeviceSynchronize();
        
        dim3 grid2( (number_blocks + number_threads * number_threads - 1) / (number_threads * number_threads));
        dim3 block2(number_threads * number_threads);
        if(verbose)
            printf("Starting weighted mean\n");
        calculate_weighted_mean<<<grid2, block2>>>(mean_red_d, mean_green_d, mean_blue_d,
                                                dev_red_d,  dev_green_d,  dev_blue_d,
                                                dev_red_dg, dev_green_dg, dev_blue_dg,
                                                mean_red_dg, mean_green_dg, mean_blue_dg, number_blocks);
        cudaDeviceSynchronize();

        if(verbose)
            printf("Retriving Weighted values\n");

        checkCudaErrors(cudaMemcpy(&mean_red_g, mean_red_dg, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&mean_green_g, mean_green_dg, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&mean_blue_g, mean_blue_dg, sizeof(double), cudaMemcpyDeviceToHost));
        
        if(verbose)
            printf("Applying corrections\n");
        apply_correction<<<grid, block>>>(imageStep_d, imageRes_d, height, width, imageStep_d == image1_d ? padding_pix : 0, 
                                        (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_red_g),
                                        (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_green_g),
                                        (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_blue_g));


        cudaDeviceSynchronize();
    }

                                            
    // Retriving Results
    if(verbose)
        printf("Retriving Results\n");

    checkCudaErrors(cudaMemcpy(imageRes, imageRes_d, sizeof(Pixel) * width * height, cudaMemcpyDeviceToHost));
    if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
        checkCudaErrors(cudaMemcpy(&sum, sum_y, sizeof(double), cudaMemcpyDeviceToHost));

    if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
        checkCudaErrors(cudaMemcpy(&dev_red_g, dev_red_dg, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&dev_green_g, dev_green_dg, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&dev_blue_g, dev_blue_dg, sizeof(double), cudaMemcpyDeviceToHost));
    }
       
    if((type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE) && verbose)
        printf("Calculate Luminance %f\n", sum /  MAX_COLOR_CHANNEL_VALUE);
    
    if((type == PROCESS_COLOR || type == PROCESS_COMPLETE) && verbose){
        printf("Calculate Dev Red %f\n", dev_red_g);
        printf("Calculate Dev Green %f\n", dev_green_g);
        printf("Calculate Dev Blue %f\n", dev_blue_g);
        printf("Calculate Weigthed Mean Red %f\n", mean_red_g);
        printf("Calculate Weigthed Mean Green %f\n", mean_green_g);
        printf("Calculate Weigthed Mean Blue %f\n", mean_blue_g);
        printf("Correction value for Channel Red %f\n", (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_red_g));
        printf("Correction value for Channel Green %f\n", (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_green_g));
        printf("Correction value for Channel Blue %f\n", (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_blue_g));
    }

    // Freeing Memory
    if(verbose)
        printf("Freeing Memory\n");

    checkCudaErrors(cudaFree((void *) image1_d));
    checkCudaErrors(cudaFree((void *) imageRes_d));
    if(image2 != NULL)
        checkCudaErrors(cudaFree((void *) image2_d));

    if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
        checkCudaErrors(cudaFree((void *) sum_y));

    if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
        checkCudaErrors(cudaFree((void *) mean_red_d));
        checkCudaErrors(cudaFree((void *) mean_green_d));
        checkCudaErrors(cudaFree((void *) mean_blue_d));

        checkCudaErrors(cudaFree((void *) dev_red_d));
        checkCudaErrors(cudaFree((void *) dev_green_d));
        checkCudaErrors(cudaFree((void *) dev_blue_d));

        checkCudaErrors(cudaFree((void *) dev_red_dg));
        checkCudaErrors(cudaFree((void *) dev_green_dg));
        checkCudaErrors(cudaFree((void *) dev_blue_dg));

        checkCudaErrors(cudaFree((void *) mean_red_dg));
        checkCudaErrors(cudaFree((void *) mean_green_dg));
        checkCudaErrors(cudaFree((void *) mean_blue_dg));
    }
    if(verbose)
        printf("Finished Process\n");
    return (uint8_t*)imageRes;
}

void print_help(){
    printf("image_correction\n");
    printf("\t-i=<string>: input file name\n");
    printf("\t-help : Help\n" );
    printf("\t-verb : Verbose mode\n");  
    printf("\t-o=<string>: Output file name\n"); 
    printf("\t-s=<string>: Second input file name\n");  
    printf("\t-color : Color correction only (-s option must not be included)\n");
    printf("\t-exp : Exposure correction only (-s option must be included to)\n");
    printf("\t-b=<int>: Size of block ( x <= 32, x >= 2, default = 32)\n");
    printf("\t-force-padding=<int>: Number of Pixels to simulate padding with\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    bool help = false;
    bool error = false;

    bool color_correction = 0;
    bool exposure_correction = 0;
    int type = PROCESS_COMPLETE;
    
    int thread_size_block = 32;
    
    char * fileInput1 = NULL;
    char * fileInput2 = NULL;
    char * fileOutput = NULL;

    uint8_t* image1 = NULL;
    uint8_t* image2 = NULL;

    int width, height, channels;
    int force_pad = 0;

    float processing_time;
    cudaEvent_t start_time;
    cudaEvent_t stop_time;

    // Obtención de parametros de argumentos
    help =  checkCmdLineFlag(argc, (const char **)argv, "help");
    verbose = checkCmdLineFlag(argc, (const char **) argv, (const char *) "verb");

    thread_size_block = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "b")?:thread_size_block;

    force_pad = getCmdLineArgumentInt(argc, (const char **)argv, "force-padding")?: force_pad;
    getCmdLineArgumentString(argc, (const char **) argv, (const char *) "o", &fileOutput);
    getCmdLineArgumentString(argc, (const char **) argv, (const char *) "i", &fileInput1);
    getCmdLineArgumentString(argc, (const char **) argv, (const char *) "s", &fileInput2);
    

    color_correction = checkCmdLineFlag(argc, (const char **) argv, (const char *) "color");
    exposure_correction = checkCmdLineFlag(argc, (const char **) argv, (const char *) "exp");
    
    if (color_correction && exposure_correction) {
        printf("-e & -c are incompatible\n");
        error = 1;
    } else if (color_correction)
        type = PROCESS_COLOR;
    else if(exposure_correction)
        type = PROCESS_EXPOSITION;

    if(force_pad < 0){
        printf("Force padding must be equal or greater than 0\n");
        error = 1;
    }

    if (fileInput1 == NULL){
        printf("Input file 1 is obligartory\n");
        error = 1;
    }

    if(type == PROCESS_EXPOSITION && fileInput2 == NULL){
        printf("For exposure correction second input is needed\n");
        error = 1;
    }

    if (type == PROCESS_COLOR && fileInput2 != NULL) {
        printf("For color correction second input is not allowed\n");
        error = 1;
    }

    if (thread_size_block > MAX_THREADS) {
        printf("The max number of threads in a dimension is %d\n",MAX_THREADS);
        error = 1;
    }
    if (thread_size_block <= 1) {
        printf("The min number of threads in a dimension is 2\n");
        error = 1;
    }
    if(help || error){
        print_help();
        return error ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    image1 = stbi_load(fileInput1, &width, &height, &channels, 0);
    if (image1 == NULL) {
        printf("Could not load image 1\n");
        return EXIT_FAILURE;
    }
    if (channels != CHANNEL_NUMBER){
        printf("Images´ format didnt have four channels\n");
        return EXIT_FAILURE;
    }

    if(width - force_pad <= 0){
        stbi_image_free(image1);
        printf("Force_pad needs to be lower than the width\n");
        return EXIT_FAILURE;
    }

    if (fileInput2 != NULL) {
        int width2, height2, channels2;
        image2 = stbi_load(fileInput2, &width2, &height2, &channels2, 0);
        if (image2 == NULL) {
            stbi_image_free(image1);
            printf("Could not load image 2\n");
            return EXIT_FAILURE;
        }

        if (width != width2 || height != height2 || channels != channels2) {
            stbi_image_free(image1);
            stbi_image_free(image2);
            printf("Images´ format dont match\n");
            return EXIT_FAILURE;
        }
    }
    width = width - force_pad;

    if(verbose)
        printf("width: %d, height %d, thread_size_block %d\n", width, height, thread_size_block);

    checkCudaErrors(cudaEventCreate(&start_time,0));
    checkCudaErrors(cudaEventCreate(&stop_time ,0));
    checkCudaErrors(cudaEventRecord(start_time ,0));
    
    uint8_t * imageRes = processBuffer(width, height, image1, image2, thread_size_block, force_pad,type);
    
    checkCudaErrors(cudaEventRecord(stop_time, 0));        
    cudaEventSynchronize(stop_time);   
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_time, stop_time));       
    
    
    printf("Tiempo: %f (ms)\n", processing_time);  


    int result = stbi_write_png(fileOutput == NULL ? "output.png" : fileOutput, width, height, CHANNEL_NUMBER, imageRes, width * channels);
    if (result) {
        if (verbose) printf("\nCompleted saving image to file\n");
    } else {
        printf("Error al guardar la imagen\n");
    }

    free(imageRes);

    stbi_image_free(image1);
    if (fileInput2 != NULL) {
        stbi_image_free(image2); 
    }

    // Finalización
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}