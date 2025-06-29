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

// tipo de procesamiento
#define PROCESS_COMPLETE    0
#define PROCESS_COLOR       1
#define PROCESS_EXPOSITION  2
// Número máximo de threads
#define MAX_THREADS 32

bool verbose = 0;
cudaError_t err = cudaSuccess;


////////////////////////////////////////////////////////////////////////////////

// -------------------------------------- Sección Auto Exposición --------------------------------------

// Operación para el calculo de la luminosidad de un pixel
#define RGB_Y(R,G,B) 0.299f * (double)R + 0.587f * (double)G + 0.114f * (double) B
// Cálculo de exposición de bloque
double calculateExposition(Pixel * data, int startY, int startX, int endY, int endX, int height, int width, int padding_pix) {
    double blockLuminance = 0;
    for(int i = startY; i < endY; i++){
        for (int j = startX; j < endX; j++)
        {
            Pixel* pixel_p = &(data[i * (width + padding_pix) + j]);
            blockLuminance += RGB_Y(pixel_p->r, pixel_p->g, pixel_p->b);
        }
    }
    // Media de la luminancia del bloque
    return blockLuminance / (width * height);
}

// Aplicación de la fusión
void applyFusion(Pixel * im1, Pixel * im2, Pixel * imRes, int startY, int startX, int endY, int endX, int height, int width, int padding_pix) {
    for(int i = startY; i < endY; i++){
        for (int j = startX; j < endX; j++) {
            Pixel* pixel_1 = &(im1[i * (width + padding_pix) + j]);
            Pixel* pixel_2 = &(im2[i * (width + padding_pix) + j]);
            Pixel* pixel_Res = &(imRes[i * width + j]);
            // Se indica el stride a 255 (valor por defecto) y se guarda la media del color de los pixeles en la matriz resultado
            pixel_Res->r = (pixel_1->r + pixel_2->r) >> 1;
            pixel_Res->g = (pixel_1->g + pixel_2->g) >> 1;
            pixel_Res->b = (pixel_1->b + pixel_2->b) >> 1;
            pixel_Res->stride = MAX_COLOR_CHANNEL_VALUE;
        }
    }
}

// ---------------------------------- Sección AutoBalanceo de Blancos ----------------------------------

// Cálculo de la media del canal de color de un bloque
void calculateMeanChanelValue(Pixel * image, int startY, int startX, int endY, int endX, int height, int width, int padding_pix, 
                              double & red, double & blue, double & green) {
    red = 0, green = 0, blue = 0; // Variables donde sumar los valores de los canales de color
    for(int i = startY; i < endY; i++){
        for (int j = startX; j < endX; j++) {
            Pixel* pixel = &(image[i * (width + padding_pix) + j]);
            red += pixel->r;
            green += pixel->g;
            blue += pixel->b;
        }
    }
    // Cálculo de la media de canal de color en el bloque
    double blockSize = (endX - startX) * (endY - startY);
    red = red / blockSize;
    green = green / blockSize;
    blue = blue / blockSize;
}

// Cálculo de la desviación en un bloque
void calculateDeviationChanelValue(Pixel * image, int startY, int startX, int endY, int endX, int height, int width, int padding_pix, 
                                   double mean_red, double mean_green, double mean_blue, double & red, double & blue, double & green) {
    
    red = 0, green = 0, blue = 0; // Variables donde sumar los valores de los canales de color
    for(int i = startY; i < endY; i++){
        for (int j = startX; j < endX; j++) {
            Pixel* pixel = &(image[i * (width + padding_pix) + j]);
            red   += pow(pixel->r - mean_red  , 2);
            green += pow(pixel->g - mean_green, 2);
            blue  += pow(pixel->b - mean_blue , 2);
        }
    }

    // Cálculo final de la deviación de los canales de color en el bloque
    double blockSize = (endX - startX) * (endY - startY);
    red   = sqrt(red   / blockSize);
    green = sqrt(green / blockSize);
    blue  = sqrt(blue  / blockSize);
}


// Aplicar corrección a los buffers
void applyCorrection(Pixel * image_org, Pixel * image_dest, int startY, int startX, int endY, int endX, int height, int width, int padding_pix, 
                     double corrR, double corrG, double corrB) {
    for(int i = startY; i < endY; i++){
        for (int j = startX; j < endX; j++) {
            Pixel* pixel_org = &(image_org[i * (width + padding_pix) + j]);
            Pixel* pixel_res = &(image_dest[i * width + j]);
            // Multiplica el valor del pixel original por el valor de corrección
	    
	    double red = pixel_org->r * corrR;
            double green = pixel_org->g * corrG;
            double blue = pixel_org->b * corrB;
            
            pixel_res->stride = pixel_org->stride;
            pixel_res->r = red < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) red : MAX_COLOR_CHANNEL_VALUE;
            pixel_res->g = green < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) green : MAX_COLOR_CHANNEL_VALUE;
            pixel_res->b = blue < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) blue : MAX_COLOR_CHANNEL_VALUE;
            
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
uint8_t * processBuffer(int width, int height, uint8_t* image1, uint8_t* image2, int number_threads, int padding_pix, int type, int cpu_blocks) {
    
    int number_blocks_x = (width  + number_threads - 1) / number_threads;
    int number_blocks_y = (height + number_threads - 1) / number_threads;

    int new_height = ((number_blocks_y - cpu_blocks) * number_threads);
    new_height = new_height < height ? new_height : height;

    dim3 grid( number_blocks_x, number_blocks_y - cpu_blocks);
    dim3 block(number_threads, number_threads);

    int number_blocks_gpu = grid.x * grid.y;
    int number_blocks_cpu = number_blocks_x * cpu_blocks;

    if(verbose){
        printf("Grid X: %d, Grid Y: %d, Block X: %d, Block Y:%d, Padding_pix:%d\n", grid.x, grid.y, block.x, block.y, padding_pix);
        printf("CPU X: %d, CPU Y: %d\n", number_blocks_x, cpu_blocks);
    }

    Pixel * image1_d;
    Pixel * image2_d;
    Pixel * imageRes, *imageStep; 
    Pixel * imageRes_d, *imageStep_d;

    imageRes = (Pixel *) malloc(sizeof(Pixel) * width * height);
    
    double * sum_y;    // Valor de la luminancia en la GPU
    double luminance_local = 0; // Valor de la luminancia calculada en cada hilo
    double luminance_total = 0; // Valor total de la luminancia

    // g : global, d : device, l : local

    double dev_red_g = 0, dev_green_g = 0, dev_blue_g = 0;    // Variables con el valor de la suma de la desvicaion
    double mean_red_g = 0, mean_green_g = 0, mean_blue_g = 0; // Variables con el valor de media ponderada

    // Lista de medias y deviaciones para los bloques de la cpu
    double * mean_red = (double *) malloc(sizeof(double) * number_blocks_cpu);
    double * mean_green = (double *) malloc(sizeof(double) * number_blocks_cpu);
    double * mean_blue = (double *) malloc(sizeof(double) * number_blocks_cpu);
    double * dev_red = (double *) malloc(sizeof(double) * number_blocks_cpu);
    double * dev_green = (double *) malloc(sizeof(double) * number_blocks_cpu);
    double * dev_blue = (double *) malloc(sizeof(double) * number_blocks_cpu);
    
    double * dev_red_dg, *dev_green_dg, *dev_blue_dg;    // Variables con el valor de media ponderada para la GPU
    double * mean_red_dg, *mean_green_dg, *mean_blue_dg; // Variables con el valor de la suma de la desvicaion para la GPU

    // Lista de medias y deviaciones para los bloques de la gpu
    double * mean_red_d, *mean_green_d, *mean_blue_d;
    double * dev_red_d, *dev_green_d, *dev_blue_d;

    // Valores locales de la media y la deviación
    double mean_red_l = 0, mean_green_l = 0, mean_blue_l = 0;
    double dev_red_l = 0, dev_green_l = 0, dev_blue_l = 0;

    int iam;
    #pragma omp parallel private(iam) firstprivate(luminance_local, mean_red_l, mean_green_l, mean_blue_l, dev_red_l, dev_green_l, dev_blue_l) shared(luminance_total, imageStep,  dev_red_g, dev_green_g, dev_blue_g, mean_red_g, mean_green_g, mean_blue_g)
    {
        iam = omp_get_thread_num();

        if(iam == 0){
            // Reserving Memory
            if(verbose)
                printf("Allocating memory\n");
            checkCudaErrors(cudaMalloc((void **) &image1_d, sizeof(Pixel) * (width + padding_pix) * new_height ));
            checkCudaErrors(cudaMalloc((void **) &imageRes_d, sizeof(Pixel) * width * new_height));
            
            if(image2 != NULL)
                checkCudaErrors(cudaMalloc((void **) &image2_d, sizeof(Pixel) * (width + padding_pix) * new_height ));
            
            if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
                checkCudaErrors(cudaMalloc((void **) &sum_y, sizeof(double)));

            if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
                checkCudaErrors(cudaMalloc((void **) &mean_red_d,   sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMalloc((void **) &mean_green_d, sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMalloc((void **) &mean_blue_d,  sizeof(double) * number_blocks_gpu));
        
                checkCudaErrors(cudaMalloc((void **) &dev_red_d,   sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMalloc((void **) &dev_green_d, sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMalloc((void **) &dev_blue_d,  sizeof(double) * number_blocks_gpu));
        
                checkCudaErrors(cudaMalloc((void **) &dev_red_dg,   sizeof(double)));
                checkCudaErrors(cudaMalloc((void **) &dev_green_dg, sizeof(double)));
                checkCudaErrors(cudaMalloc((void **) &dev_blue_dg,  sizeof(double)));
    
                checkCudaErrors(cudaMalloc((void **) &mean_red_dg,   sizeof(double)));
                checkCudaErrors(cudaMalloc((void **) &mean_green_dg, sizeof(double)));
                checkCudaErrors(cudaMalloc((void **) &mean_blue_dg,  sizeof(double)));
            }
        }
    

        // Enviando la inforamción de las imagenes, solo de los datos que se usarán en la gpu
        if(iam == 0){
            if(verbose)
                printf("Sending information to the kernel\n");
            
            checkCudaErrors(cudaMemcpy(image1_d, image1, sizeof(Pixel) * (width + padding_pix) * new_height, cudaMemcpyHostToDevice));
                        
            if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE)
                checkCudaErrors(cudaMemset(sum_y, 0,  sizeof(double)));
        }
                
        // Calculing
        if(verbose && iam == 0)
            printf("Starting Processing in the GPU\n");
        
        // AE
        if(type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE){
            if(iam == 0){
                if(verbose)
                    printf("Starting Exposition Correction\nCalculating Exposure\n");
                // Cálculo de la luminancia en la GPU
                calculate_sum<<<grid, block, sizeof(double) * number_threads * number_threads >>>(image1_d, sum_y, width, height, padding_pix);
                if(image2 != NULL)
                    checkCudaErrors(cudaMemcpy(image2_d, image2,  sizeof(Pixel) * (width + padding_pix) * new_height, cudaMemcpyHostToDevice));
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMemcpy(&luminance_local, sum_y, sizeof(double), cudaMemcpyDeviceToHost));
            } else {
                // Cálculo de la luminancia en la CPU
                #pragma omp for collapse(2) schedule(dynamic, 10) nowait
                for(int by = number_blocks_y - cpu_blocks; by < number_blocks_y; by++){
                    for(int bx = 0; bx < number_blocks_x; bx++){
                        luminance_local += calculateExposition((Pixel *) image1, by * number_threads, bx * number_threads, 
                        by == number_blocks_y - 1 ? height : by * number_threads + number_threads, 
                        bx == number_blocks_x - 1 ? width  : bx * number_threads + number_threads, height, width, padding_pix);
                    }
                }
            }

            // Suma atomica de los valores de luminosidad local es en cada hilo
            #pragma omp atomic
            luminance_total += luminance_local;
            #pragma omp barrier
            
            
            if(iam == 0){
                if(image2 != NULL) {
                    if(verbose)
                        printf("Applaying Fusion\n");
                    // Aplicación de la fusión en la imagen para la GPU
                    apply_fusion<<<grid, block>>>(image1_d, image2_d, imageRes_d, width, height, padding_pix);
                    cudaDeviceSynchronize();
                    imageStep_d = imageRes_d;
                    imageStep = imageRes;
                } else {
                    imageStep_d = image1_d;
                    imageStep = (Pixel *) image1;
                }
            } else {
            	if(image2 != NULL) {
		        // Aplicación de la fusión en la imagen para la CPU
		        #pragma omp for collapse(2) schedule(dynamic, 10) nowait 
		        for(int by = number_blocks_y - cpu_blocks; by < number_blocks_y; by++){
		            for(int bx = 0; bx < number_blocks_x; bx++){
		                applyFusion((Pixel*) image1, (Pixel *) image2, (Pixel *) imageRes, by * number_threads, bx * number_threads, 
		                        by == number_blocks_y - 1 ? height : by * number_threads + number_threads, 
		                        bx == number_blocks_x - 1 ? width  : bx * number_threads + number_threads, height, width, padding_pix);
		            }
		        }
                }
            }
        } else {
            if(iam == 0){
                imageStep_d = image1_d;
                imageStep = (Pixel *) image1;
            }
        }

        #pragma omp barrier
        
        // AWB
        if(type == PROCESS_COLOR || type == PROCESS_COMPLETE){
            if(iam == 0){
                // Preparación de la memoria
                checkCudaErrors(cudaMemset(mean_red_d, 0,   sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMemset(mean_green_d, 0, sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMemset(mean_blue_d, 0,  sizeof(double) * number_blocks_gpu));
        
                checkCudaErrors(cudaMemset(dev_red_d, 0,   sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMemset(dev_green_d, 0, sizeof(double) * number_blocks_gpu));
                checkCudaErrors(cudaMemset(dev_blue_d, 0,  sizeof(double) * number_blocks_gpu));
        
                checkCudaErrors(cudaMemset(dev_red_dg, 0,   sizeof(double)));
                checkCudaErrors(cudaMemset(dev_green_dg, 0, sizeof(double)));
                checkCudaErrors(cudaMemset(dev_blue_dg, 0,  sizeof(double)));
        
                checkCudaErrors(cudaMemset(mean_red_dg, 0,   sizeof(double)));
                checkCudaErrors(cudaMemset(mean_green_dg, 0, sizeof(double)));
                checkCudaErrors(cudaMemset(mean_blue_dg, 0,  sizeof(double)));
                
                if(verbose)
                    printf("GPU Calculating mean\n");
                
                // Cálculo de los valores de la media en los bloques de la GPU
                calculate_mean<<<grid, block, 3 *sizeof(double) * number_threads * number_threads >>>(imageStep_d, width, height, imageStep_d == image1_d ? padding_pix : 0 , mean_red_d, mean_green_d, mean_blue_d);
                cudaDeviceSynchronize();
            } else {
                // Cálculo de los valores de la media en los bloques de la CPU
                #pragma omp for collapse(2) schedule(dynamic, 10) nowait 
                for(int by = number_blocks_y - cpu_blocks; by < number_blocks_y; by++){
                    for(int bx = 0; bx < number_blocks_x; bx++){
                        int index = (by - (number_blocks_y - cpu_blocks)) * number_blocks_x + bx;
                        calculateMeanChanelValue((Pixel*) imageStep, by * number_threads, bx * number_threads,
                                    by == number_blocks_y - 1 ? height : by * number_threads + number_threads,
                                    bx == number_blocks_x - 1 ? width  : bx * number_threads + number_threads, height, width, imageStep == (Pixel*) image1 ? padding_pix : 0 ,
                                    mean_red[index], mean_blue[index], mean_green[index]);
                    }
                }  
            }

            #pragma omp barrier

            if(iam == 0){
                if(verbose)
                printf("GPU Calculating deviation\n");

                // Cálculo de los valores de la desviación en los bloques de la GPU
                calculate_dev<<<grid, block, 3 *sizeof(double) * number_threads * number_threads>>>(imageStep_d, width, height, imageStep_d == image1_d ? padding_pix : 0 , 
                                                                                mean_red_d, mean_green_d, mean_blue_d,
                                                                                dev_red_d,  dev_green_d,  dev_blue_d,
                                                                                dev_red_dg, dev_green_dg, dev_blue_dg);
                cudaDeviceSynchronize();
                
                // Traer los valores de vuelta a la CPU para su suma
                checkCudaErrors(cudaMemcpy(&dev_red_l, dev_red_dg, sizeof(double), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&dev_green_l, dev_green_dg, sizeof(double), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&dev_blue_l, dev_blue_dg, sizeof(double), cudaMemcpyDeviceToHost));

            } else {
                 // Cálculo de los valores de la desviación en los bloques de la CPU
                 dev_red_l = 0;
                 dev_green_l = 0; 
                 dev_blue_l = 0;
                 #pragma omp for collapse(2) schedule(dynamic, 10) nowait 
                 for(int by = number_blocks_y - cpu_blocks; by < number_blocks_y; by++){
                     for(int bx = 0; bx < number_blocks_x; bx++){
                         int index = (by - (number_blocks_y - cpu_blocks)) * number_blocks_x + bx;
                         calculateDeviationChanelValue((Pixel*) imageStep, by * number_threads, bx * number_threads,
                                 by == number_blocks_y - 1 ? height : by * number_threads + number_threads,
                                 bx == number_blocks_x - 1 ? width  : bx * number_threads + number_threads, height, width, imageStep == (Pixel*) image1 ? padding_pix : 0,
                                 mean_red[index],  mean_green[index], mean_blue[index], dev_red[index], dev_blue[index], dev_green[index]);
                         dev_red_l   += dev_red[index];
                         dev_green_l += dev_green[index]; 
                         dev_blue_l  += dev_blue[index];
                     }
                 }
            }
            
            // Reducción de los valores locales de la suma de la desviación
            #pragma omp atomic
            dev_red_g   += dev_red_l; 
            #pragma omp atomic
            dev_green_g += dev_green_l; 
            #pragma omp atomic
            dev_blue_g  += dev_blue_l;

            #pragma omp barrier

            if(iam == 0){
                // Reducción de los valores de la GPU
                dim3 grid2( (number_blocks_gpu + number_threads * number_threads - 1) / (number_threads * number_threads));
                dim3 block2(number_threads * number_threads);
                if(verbose)
                    printf("Starting weighted mean\n");
                calculate_weighted_mean<<<grid2, block2>>>(mean_red_d, mean_green_d, mean_blue_d,
                                                        dev_red_d,  dev_green_d,  dev_blue_d,
                                                        dev_red_g, dev_green_g, dev_blue_g,
                                                        mean_red_dg, mean_green_dg, mean_blue_dg, number_blocks_gpu);
                cudaDeviceSynchronize();

                checkCudaErrors(cudaMemcpy(&mean_red_l, mean_red_dg, sizeof(double), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&mean_green_l, mean_green_dg, sizeof(double), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&mean_blue_l, mean_blue_dg, sizeof(double), cudaMemcpyDeviceToHost));
            } else {
                #pragma omp for schedule(guided, 20) nowait 
                for(int i = 0; i < number_blocks_cpu; i++) {
                    mean_red_l   += mean_red[i]   * (dev_red[i]   / dev_red_g);
                    mean_green_l += mean_green[i] * (dev_green[i] / dev_green_g);
                    mean_blue_l  += mean_blue[i]  * (dev_blue[i]  / dev_blue_g);
                }
            }

            #pragma omp atomic
            mean_red_g   += mean_red_l; 
            #pragma omp atomic
            mean_green_g += mean_green_l; 
            #pragma omp atomic
            mean_blue_g  += mean_blue_l;

            #pragma omp barrier

            if(iam == 0){
                if(verbose)
                printf("Applying corrections\n");
                apply_correction<<<grid, block>>>(imageStep_d, imageRes_d, height, width, imageStep_d == image1_d ? padding_pix : 0, 
                                            (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_red_g),
                                            (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_green_g),
                                            (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_blue_g));


                cudaDeviceSynchronize();

                if(verbose)
                printf("Retriving Results\n");

                checkCudaErrors(cudaMemcpy(imageRes, imageRes_d, sizeof(Pixel) * width * new_height, cudaMemcpyDeviceToHost));
            } else {
                #pragma omp for collapse(2) schedule(dynamic, 10) nowait 
                for(int by = number_blocks_y - cpu_blocks; by < number_blocks_y; by++){
                    for(int bx = 0; bx < number_blocks_x; bx++){
                        applyCorrection(imageStep, imageRes, by * number_threads, bx * number_threads, 
                                by == number_blocks_y - 1 ? height : by * number_threads + number_threads, 
                                bx == number_blocks_x - 1 ? width  : bx * number_threads + number_threads, height, width,  
                                imageStep == (Pixel*) image1 ? padding_pix : 0,
                                (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_red_g),
                                (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_green_g),
                                (mean_red_g + mean_green_g + mean_blue_g) / (3 * mean_blue_g));
                    }
                }
            }
        }
        
        // Retriving Results
        #pragma omp barrier

        if(iam == 0){
            if((type == PROCESS_EXPOSITION || type == PROCESS_COMPLETE) && verbose)
               printf("Calculate Sum %f\n", luminance_total / MAX_COLOR_CHANNEL_VALUE);
            
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
            
        }
        
    }
    checkCudaErrors(cudaDeviceSynchronize());
    if(verbose)
        printf("Finished Process\n");

    free(mean_red);
    free(mean_green);
    free(mean_blue);
    free(dev_red);
    free(dev_green);
    free(dev_blue);

    
    return (uint8_t*) imageRes;
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
    printf("\t-t=<int>: Number of CPU threads (x >= 2, default == max)\n");
    printf("\t-c=<int>: Lines of Blocks to  (x >= 0)\n");
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
    int block_cpu = 0;

    float processing_time;
    cudaEvent_t start_time;
    cudaEvent_t stop_time;

    int cpu_threads = omp_get_max_threads();

    // Obtención de parametros de argumentos
    help =  checkCmdLineFlag(argc, (const char **)argv, "help");
    verbose = checkCmdLineFlag(argc, (const char **) argv, (const char *) "verb");

    thread_size_block = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "b")?:thread_size_block;
    cpu_threads = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "t")?:cpu_threads;
    block_cpu = getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "c")?:block_cpu;

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

    if(cpu_threads <= 1){
        printf("CPU threads must be greater than 1\n");
        error = 1;
    }
    if(force_pad < 0){
        printf("Force padding must be equal or greater than 0\n");
        error = 1;
    }
    if(block_cpu < 0){
        printf("Blocks lines assign to the cpu must be greater or equal than 0\n");
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
    if(((height + thread_size_block - 1) / thread_size_block) <= block_cpu){
        stbi_image_free(image1);
        printf("Blocks lines assign to the cpu must be lower to the total number of blocks in the y dimensions\n");
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
    if(verbose)
    	printf("cpu_threads : %d\n", cpu_threads);
    omp_set_num_threads(cpu_threads);
    uint8_t * imageRes = processBuffer(width, height, image1, image2, thread_size_block, force_pad,type, block_cpu);
    
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

