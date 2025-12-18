

// ---------------------------- Estructuras, Variables gobales y Constantes ----------------------------

// Usado por STB_IMAGE para definir las funciones que se van a usar
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <omp.h>
#include <mpi.h>
#include <time.h>

// Tamaño en bytes de un pixel
#define PIXEL_SIZE 4
// Número de canales esperado por el programa
#define CHANNEL_NUMBER 4
// Valor máximo del canal de color - 2^8-1
#define MAX_COLOR_CHANNEL_VALUE 255

// Organización de memoria de un pixel
struct Pixel {
    uint8_t r; // Canal de color rojo
    uint8_t g; // Canal de color verde
    uint8_t b; // Canal de color azul
    uint8_t stride; // Byte de desplazamiento entre pixeles según XRGB8888
};

// Buffer con los datos de una imagen
struct ImageBuffer
{
    int lineNumber; // Altura de la imagen
    int lineWidth;  // Anchura de la imagen
    int lineStride; // Número de bytes entre líneas
    uint8_t* mappedData; // Información de la imagen
};

// Buffer con los datos de una imagen
struct ImageBufferSet
{
    int bufferNumber;
    ImageBuffer* imageBuffers;
};

// Información de la media y desviación de color de un bloque de una imagen
struct MeanDeviation
{

    double meanRed;         // Media del canal rojo del bloque
    double deviationRed;    // Desviación del canal rojo del bloque
    double meanGreen;       // Media del canal verde del bloque
    double deviationGreen;  // Desviación del canal verde del bloque
    double meanBlue;        // Media del canal azul del bloque
    double deviationBlue;   // Desviación del canal azul del bloque
};

// Información de un bloque de una imagen
struct ImageBlock
{
    int pixelStartX; // Posición de incio del bloque en la dimensión X
    int pixelStartY; // Posición de incio del bloque en la dimensión Y
    int blockLength; // Longitud del bloque
    int blockWidth;  // Anchura del bloque

    MeanDeviation blockinfo; // Información de medias y desviaciones
};

// Conjuntos de bloques a tratar de una imagen
struct ImageBlockSet
{
    int blockNumber; // Número de bloques de una imagen
    ImageBlock* imageBlocks; // Bloques del Set
};

// Parametros de anchura, altura y canales de la imagen o imágenes
int width, height, channels;
// Activación del modo verboso
int verbose;
// Número de Proceso
int processNumber;
// Identificador del proceso
int rank;



// ------------------------------------- Sección Manejo de Buffers -------------------------------------

// Reserva memoria para la información del buffer de una imagen, no reserva memoria para el buffer
ImageBuffer* createBufferNoAlloc(int lineNumber, int lineWidth, int lineStride) {
    ImageBuffer* im = new ImageBuffer;
    im->lineNumber = lineNumber;
    im->lineWidth = lineWidth;
    im->lineStride = lineStride;
    im->mappedData = NULL;
    return im;
}

// Libera memoria para la información del buffer, no se libera la memoria del buffer
ImageBuffer* deleteBufferNoAlloc(ImageBuffer* imageBuffer) {
    delete imageBuffer;
    return NULL;
}

// Reserva memoria para la información del buffer de una imagen y el propio buffer
ImageBuffer * createBuffer(int lineNumber, int lineWidth, int lineStride) {
    ImageBuffer* im = createBufferNoAlloc(lineNumber, lineWidth, lineStride);
    im->mappedData = new uint8_t[(im->lineWidth * PIXEL_SIZE + im->lineStride) * im->lineNumber];
    return im;
}

// Libera memoria para la información del buffer y el propio buffer
ImageBuffer* deleteBuffer(ImageBuffer* imageBuffer) {
    delete [] imageBuffer->mappedData;
    return deleteBufferNoAlloc(imageBuffer);
}

// Crea el conjunto de bloques y los bloques con las divisiones de la matriz
ImageBlockSet* createBlocks(ImageBuffer* imageBuffer, int xDivisions, int yDivisions) {
    ImageBlockSet* ibs = new ImageBlockSet;
    ibs->blockNumber = xDivisions * yDivisions; 
    ibs->imageBlocks = new ImageBlock[xDivisions * yDivisions];

    int iMod = imageBuffer->lineNumber % yDivisions; // Cuantos bloques tendrán un pixel más en su dimensión Y
    int jMod = imageBuffer->lineWidth % xDivisions;  // Cuantos bloques tendrán un pixel más en su dimensión Y
    int yStart = 0;
    for (int i = 0; i < yDivisions; i++) {
        int xStart = 0;
        for (int j = 0; j < xDivisions; j++) {
            ibs->imageBlocks[i * xDivisions + j].pixelStartX = xStart; // Se define los comienzos como los fines del bloque anterior
            ibs->imageBlocks[i * xDivisions + j].pixelStartY = yStart;
            ibs->imageBlocks[i * xDivisions + j].blockLength = imageBuffer->lineNumber / yDivisions + (iMod > i ? 1 : 0); // Se le suma 1 si la división lo requiere
            ibs->imageBlocks[i * xDivisions + j].blockWidth = imageBuffer->lineWidth / xDivisions + (jMod > j ? 1 : 0);
            xStart += imageBuffer->lineWidth / xDivisions + (jMod > j ? 1 : 0);
        }
        yStart += imageBuffer->lineNumber / yDivisions + (iMod > i ? 1 : 0);
    }
    return ibs;
}

// Libera la memoria que guarda el conjunto de bloques
ImageBlockSet* deleteBlocks(ImageBlockSet* imageBlocks) {
    delete [] imageBlocks->imageBlocks;
    delete imageBlocks;
    return NULL;
}

// Creación de una lista de Bufferes con alocación de memoria 
ImageBufferSet* createBuffers(ImageBlockSet* imageBlocks) {
    ImageBufferSet* imageBufferSet = new ImageBufferSet;
    imageBufferSet->bufferNumber = imageBlocks->blockNumber;
    imageBufferSet->imageBuffers = new ImageBuffer[imageBufferSet->bufferNumber];

    for (int i = 0; i < imageBufferSet->bufferNumber; i++) {
        imageBufferSet->imageBuffers[i].lineNumber = imageBlocks->imageBlocks[i].blockLength;
        imageBufferSet->imageBuffers[i].lineWidth = imageBlocks->imageBlocks[i].blockWidth;
        imageBufferSet->imageBuffers[i].lineStride = 0;
        imageBufferSet->imageBuffers[i].mappedData = new uint8_t[imageBlocks->imageBlocks[i].blockLength * imageBlocks->imageBlocks[i].blockWidth * PIXEL_SIZE];
    }

    return imageBufferSet;
}

// Liberación de una lista de Buffers y memoria asociada
ImageBufferSet* deleteBuffers(ImageBufferSet* imageBufferSet) {
    for (int i = 0; i < imageBufferSet->bufferNumber; i++) {
        delete [] imageBufferSet->imageBuffers[i].mappedData;
    }
     
    delete [] imageBufferSet->imageBuffers;
    delete imageBufferSet;
    return NULL;
}

// -------------------------------------- Sección Auto Exposición --------------------------------------

// Operación para el calculo de la luminosidad de un pixel
#define RGB_TO_Y(R, G, B) ((0.299f * ((double)R)) + (0.587f * ((double)G)) + (0.114f * ((double)B)))
// Cálculo de exposición de bloque
double calculateExposition(ImageBuffer* imageBuffer, ImageBlock* imageBlock, int width, int height) {
    double blockLuminance = 0.0f;
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* line_pointer = imageBuffer->mappedData + i * (imageBuffer->lineWidth * PIXEL_SIZE + imageBuffer->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            Pixel* pixel_p = (Pixel*) (line_pointer + j * PIXEL_SIZE);
            blockLuminance += RGB_TO_Y(pixel_p->r, pixel_p->g, pixel_p->b) / MAX_COLOR_CHANNEL_VALUE;
        }
    }
    // Media de la luminancia
    return blockLuminance / (width * height);
}

// Aplicación de la fusión
void applyFusion(ImageBuffer* iBuf1, ImageBuffer* iBuf2, ImageBuffer* iBufRes, ImageBlock* imageBlock) {
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        // Se cálcula el comienzo de cada línea 
        uint8_t* linePointer1 = iBuf1->mappedData + i * (iBuf1->lineWidth * PIXEL_SIZE + iBuf1->lineStride);
        uint8_t* linePointer2 = iBuf2->mappedData + i * (iBuf2->lineWidth * PIXEL_SIZE + iBuf2->lineStride);
        uint8_t* linePointerRes = iBufRes->mappedData + i * (iBufRes->lineWidth * PIXEL_SIZE + iBufRes->lineStride);

        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            // Se obtiene el pixel de cada imagen
            Pixel* pixel_p1 = (Pixel*)(linePointer1 + j * PIXEL_SIZE);
            Pixel* pixel_p2 = (Pixel*)(linePointer2 + j * PIXEL_SIZE);
            Pixel* pixel_pRes = (Pixel*)(linePointerRes + j * PIXEL_SIZE);

            // Se indica el stride a 255 (valor por defecto) y se guarda la media del color de los pixeles en la matriz resultado
            pixel_pRes->r = (pixel_p1->r + pixel_p2->r) >> 1;
            pixel_pRes->g = (pixel_p1->g + pixel_p2->g) >> 1;
            pixel_pRes->b = (pixel_p1->b + pixel_p2->b) >> 1;
            pixel_pRes->stride = MAX_COLOR_CHANNEL_VALUE;

        }
    }
}

// Procedimiento para la aplicación de la corrección de exposición
ImageBufferSet* processExposureCorrection(ImageBufferSet* imageBuffer, ImageBufferSet* imageFusionTest, ImageBlockSet * imageBlockSet, int org_width, int org_height) {
    ImageBufferSet* imageRes = NULL;
    double luminance = 0.0f;
    int iam;
    
    if (verbose && rank == 0)                                          
        std::cout << "Calculating total exposition" << std::endl;

    #pragma omp parallel private(iam) shared(imageBuffer, imageBlockSet, imageFusionTest, imageRes)
    {

        iam = omp_get_thread_num();
        // Calculo de la exposición en la imagen haciendo uso de la luminancia
        #pragma omp for reduction(+:luminance) schedule(dynamic,1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating exposure: I am %d of process %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, rank, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }
            // Se suma la luminancia de los bloques
            luminance += calculateExposition(&(imageBuffer->imageBuffers[i]), &(imageBlockSet->imageBlocks[i]), org_width, org_height);
        }

        #pragma omp single 
        {
            if (verbose)
                printf("Process %d is done calculating exposition\n", rank);
            MPI_Allreduce(MPI_IN_PLACE, &luminance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        if (verbose && iam == 0) {
           printf("Result of exposure for process %d calculation: %lf\n", rank, luminance);
        }

        // Gestion de la fusión de imagenes
        if (imageFusionTest != NULL) { // Si se quiere realizar fusión
            #pragma omp single
            {
                if (verbose) {
                    printf("\nImage fusion starting, reserving shared memory for process %d \n", rank);
                }
                imageRes = createBuffers(imageBlockSet);
            }

            #pragma omp for
            for (int i = 0; i < imageBlockSet->blockNumber; i++) {
                if (verbose) {
                    printf("Applying fusion: I am %d of process %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                        iam, rank, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                        imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
                }
                applyFusion(&(imageBuffer->imageBuffers[i]), &(imageFusionTest->imageBuffers[i]), &(imageRes->imageBuffers[i]), &(imageBlockSet->imageBlocks[i]));
            }

        }
        else {
            #pragma omp single
            {
                if (verbose && rank == 0) {
                    printf("No second image passed, ignoring step\n");
                }
                imageRes = imageBuffer;
            }   
        }
    }
    
    return imageRes;
}


// ---------------------------------- Sección AutoBalanceo de Blancos ----------------------------------

// Cálculo de la media del canal de color de un bloque
void calculateMeanChanelValue(ImageBuffer* imageBuffer, ImageBlock* imageBlock) {
    
    double red = 0, green = 0, blue = 0; // Variables donde sumar los valores de los canales de color
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* line_pointer = imageBuffer->mappedData + i * (imageBuffer->lineWidth * PIXEL_SIZE + imageBuffer->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            Pixel* pixel_p = (Pixel*)(line_pointer + j * PIXEL_SIZE);
            red += pixel_p->r;
            green += pixel_p->g;
            blue += pixel_p->b;
        }
    }

    // Cálculo de la media dl canal de color en el bloque
    double blockSize = imageBlock->blockWidth * imageBlock->blockLength;
    imageBlock->blockinfo.meanRed = red / blockSize;
    imageBlock->blockinfo.meanBlue = blue / blockSize;
    imageBlock->blockinfo.meanGreen = green / blockSize;
}

// Cálculo de la desviación en un bloque
void calculateDeviationChanelValue(ImageBuffer* imageBuffer, ImageBlock* imageBlock) {
    
    double red = 0, green = 0, blue = 0; // Variables para acumular el sumatorio
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* line_pointer = imageBuffer->mappedData + i * (imageBuffer->lineWidth * PIXEL_SIZE + imageBuffer->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            Pixel* pixel_p = (Pixel*)(line_pointer + j * PIXEL_SIZE);
            red += pow(pixel_p->r - imageBlock->blockinfo.meanRed, 2);
            green += pow(pixel_p->g - imageBlock->blockinfo.meanGreen, 2);
            blue += pow(pixel_p->b - imageBlock->blockinfo.meanBlue, 2);
        }
    }

    // Cálculo final de la deviación de los canales de color en el bloque
    double blockSize = imageBlock->blockWidth * imageBlock->blockLength;
    imageBlock->blockinfo.deviationRed = sqrt(red / blockSize);
    imageBlock->blockinfo.deviationGreen = sqrt(green / blockSize);
    imageBlock->blockinfo.deviationBlue = sqrt(blue / blockSize);
}

// Aplicar corrección a los buffers
void applyCorrection(ImageBuffer* imageBufferOrg, ImageBuffer* imageBufferRes, ImageBlock* imageBlock, double corrR, double corrG, double corrB) {
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        // Comienzo de las lineas
        uint8_t* linePointerOrg = imageBufferOrg->mappedData + i * (imageBufferOrg->lineWidth * PIXEL_SIZE + imageBufferOrg->lineStride);
        uint8_t* linePointerRes = imageBufferRes->mappedData + i * (imageBufferRes->lineWidth * PIXEL_SIZE + imageBufferRes->lineStride);

        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            Pixel* pixel_pOrg = (Pixel*)(linePointerOrg + j * PIXEL_SIZE);
            Pixel* pixel_pRes = (Pixel*)(linePointerRes + j * PIXEL_SIZE);

            // Multiplica el valor del pixel original por el valor de corrección
            double red = pixel_pOrg->r * corrR;
            double green = pixel_pOrg->g * corrG;
            double blue = pixel_pOrg->b * corrB;
            
            pixel_pRes->stride = pixel_pOrg->stride;
            pixel_pRes->r = red < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) red : MAX_COLOR_CHANNEL_VALUE;
            pixel_pRes->g = green < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) green : MAX_COLOR_CHANNEL_VALUE;
            pixel_pRes->b = blue < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) blue : MAX_COLOR_CHANNEL_VALUE;
        }
    }
}

// Procedimiento para la aplicación de la corrección de color
ImageBufferSet* processColorCorrection(ImageBufferSet* imageBuffer, ImageBlockSet* imageBlockSet) {

    if (verbose && rank == 0)
        std::cout << "Calculating Mediums and Deviations" << std::endl;
    
    double deviationSumR = 0; // Las variables con los valores globales de desviación serán calculados mediante reducción
    double deviationSumG = 0;
    double deviationSumB = 0;

    double stdAvgR = 0; // Las medias ponderadas serán calculadas mediante reducción
    double stdAvgG = 0;
    double stdAvgB = 0;

    double corrR = 0; // Las variables de corrección serán calculadas por todos los hilos y su resultado será el mismo para todos
    double corrG = 0;
    double corrB = 0;

    ImageBufferSet* imagesBufferRes = createBuffers(imageBlockSet);

    int iam = 0;
    // Lanzamiento de hilos
    // Cada hilo realizará el cálculo de uno de los bloques asignados al proceso
    #pragma omp parallel private(iam,corrR,corrG,corrB) shared(imageBuffer, imageBlockSet, imagesBufferRes)
    {


        iam = omp_get_thread_num(); // Obtiene el número del thread

        // Cálculo de la suma de las desviaciones de color 
        #pragma omp for reduction(+:deviationSumR, deviationSumG, deviationSumB) schedule(dynamic, 1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating mean and deviation: I am %d of process %d working with block : startX= %i, length=%d, width=%d \n",
                    iam, rank, i, imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }

            calculateMeanChanelValue( &(imageBuffer->imageBuffers[i]), &(imageBlockSet->imageBlocks[i]));
            calculateDeviationChanelValue( &(imageBuffer->imageBuffers[i]), &(imageBlockSet->imageBlocks[i]));

            deviationSumR += imageBlockSet->imageBlocks[i].blockinfo.deviationRed;
            deviationSumG += imageBlockSet->imageBlocks[i].blockinfo.deviationGreen;
            deviationSumB += imageBlockSet->imageBlocks[i].blockinfo.deviationBlue;
        }
        
        // Uno de los hilos se encargará de reducir el valor dentro de MPI
        #pragma omp single 
        {
            if(verbose)
                printf("Deviation sum of process %d : RedDev: %lf GreenDev: %lf BlueDev: %lf\n", rank, deviationSumR, deviationSumG, deviationSumB);
            MPI_Allreduce(MPI_IN_PLACE, &deviationSumR, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &deviationSumG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &deviationSumB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        } // Barrera implicita, los valores se reparten automaticamente gracias a la declaración shared
          // Aseguramos que todos los hilos han actualizado su vista del valor con el flush implícito


        if (verbose && rank == 0 && iam == 0) {
            printf("Calculating weighted average and corrections\n");
        }

        // Cálculo de la media ponderada según las desviaciones
        #pragma omp for reduction(+:stdAvgR, stdAvgG, stdAvgB) schedule(guided,5)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating weighted avarage: I am %d of process %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, rank, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);

            }

            stdAvgR += imageBlockSet->imageBlocks[i].blockinfo.meanRed * (imageBlockSet->imageBlocks[i].blockinfo.deviationRed / deviationSumR);
            stdAvgG += imageBlockSet->imageBlocks[i].blockinfo.meanGreen * (imageBlockSet->imageBlocks[i].blockinfo.deviationGreen / deviationSumG);
            stdAvgB += imageBlockSet->imageBlocks[i].blockinfo.meanBlue * (imageBlockSet->imageBlocks[i].blockinfo.deviationBlue / deviationSumB);
        }

        // Reducción de las medias ponderadas
        #pragma omp single 
        {
            if(verbose)
                printf("Weighted avarage of process %d : RedMean: %lf GreenMean: %lf BlueMean: %lf\n", rank, stdAvgR, stdAvgG, stdAvgB);
            MPI_Allreduce(MPI_IN_PLACE, &stdAvgR, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &stdAvgG, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &stdAvgB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        } // Barrera implicita, los valores se reparten automaticamente gracias a la declaración shared
          // Aseguramos que todos los hilos han actualizado su vista del valor con el flush implcito


        //  Cálculo de las variables de corrección de color
        if (stdAvgR != 0)
            corrR = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgR);
        if (stdAvgG != 0)
            corrG = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgG);
        if (stdAvgB != 0)
            corrB = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgB);
        
        if (verbose && rank == 0 && iam == 0) {
                printf("\nWeighted Avarege for channel Red  : %f\n", stdAvgR);
                printf("Correction Value for channel Red  : %f\n", corrR);
                printf("Weighted Avarege for channel Green: %f\n", stdAvgG);
                printf("Correction Value for channel Green: %f\n", corrG);
                printf("Weighted Avarege for channel Blue : %f\n", stdAvgB);
                printf("Correction Value for channel Blue : %f\n", corrB);
                printf("\nApplying color corrections\n\n");
        }
        
        // Aplicación de la corrección de color final
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Applying Corrections: I am %d working of process %d with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, rank, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }
            applyCorrection(&(imageBuffer->imageBuffers[i]), &(imagesBufferRes->imageBuffers[i]), &(imageBlockSet->imageBlocks[i]), corrR, corrG, corrB);
        }

    }

    return imagesBufferRes;
}


// ------------------------------------------- Programa Main -------------------------------------------

// Crea un tipo de dato MPI para enviar la información de la media y desviación de un bloque de la imagen
void create_block_info_MeanDeviation_type(MPI_Datatype* mpi_block_info) {
    int block_lengths[6] = { 1, 1, 1, 1, 1, 1 };
    MPI_Aint offsets[6];
    MPI_Datatype types[6] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                              MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    offsets[0] = offsetof(MeanDeviation, meanRed);
    offsets[1] = offsetof(MeanDeviation, deviationRed);
    offsets[2] = offsetof(MeanDeviation, meanGreen);
    offsets[3] = offsetof(MeanDeviation, deviationGreen);
    offsets[4] = offsetof(MeanDeviation, meanBlue);
    offsets[5] = offsetof(MeanDeviation, deviationBlue);

    // Creación del tipo de dato
    MPI_Type_create_struct(6, block_lengths, offsets, types, mpi_block_info);
    MPI_Type_commit(mpi_block_info);

}

// Crea un tipo de dato MPI para enviar la información de un bloque de la imagen
void create_block_type(MPI_Datatype* mpi_block, MPI_Datatype mpi_block_info) {
    int block_lengths[5] = { 1, 1, 1, 1, 1 };
    MPI_Aint offsets[5];
    MPI_Datatype types[5] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, mpi_block_info };

    offsets[0] = offsetof(ImageBlock, pixelStartX);
    offsets[1] = offsetof(ImageBlock, pixelStartY);
    offsets[2] = offsetof(ImageBlock, blockLength);
    offsets[3] = offsetof(ImageBlock, blockWidth);
    offsets[4] = offsetof(ImageBlock, blockinfo);

    // Creación del tipo de dato 
    MPI_Type_create_struct(5, block_lengths, offsets, types, mpi_block);
    MPI_Type_commit(mpi_block);
}

// Crea un tipo de dato MPI para el envío de pixeles
void create_pixel_type(MPI_Datatype* mpi_pixel_type) {
    int block_lengths[4] = { 1, 1, 1, 1 };
    MPI_Aint offsets[4];
    MPI_Datatype types[4] = { MPI_UINT8_T, MPI_UINT8_T, MPI_UINT8_T , MPI_UINT8_T };

    offsets[0] = offsetof(Pixel, r);
    offsets[1] = offsetof(Pixel, g);
    offsets[2] = offsetof(Pixel, b);
    offsets[3] = offsetof(Pixel, stride);

    // Creación del tipo de dato 
    MPI_Type_create_struct(4, block_lengths, offsets, types, mpi_pixel_type);
    MPI_Type_commit(mpi_pixel_type);
}

// Crea un tipo de dato MPI para el envío de submatices (elimina padding)
void create_submatrix_type(int global_rows, int global_cols, int sub_rows, int sub_cols, int start_row, int start_col, MPI_Datatype* submatrix_type, MPI_Datatype content_type) {
    int sizes[2] = { global_rows, global_cols };   // Tamaño de la matriz original
    int subsizes[2] = { sub_rows, sub_cols };      // Tamaño de la submatriz
    int starts[2] = { start_row, start_col };      // Offset de la submatriz dentro de la matriz original

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, content_type, submatrix_type);
    MPI_Type_commit(submatrix_type);
}

// Dota de información aleatoria a un buffer de memory
void fill_buffer(uint8_t * buffer, uint x, uint y) {
    srand(time(NULL));
    for(int i=0; i<x*y;i++)
        buffer[i] = rand() % MAX_COLOR_CHANNEL_VALUE;
}


// Copia un bloque de la matriz a otra sección de memoria (elimina padding)
void copy_matrix_buffer(ImageBuffer* iBufOrg, Pixel* iBufRes, ImageBlock* imageBlock) {
    Pixel *iBufResAux = iBufRes;
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* linePointer = iBufOrg->mappedData + i * (iBufOrg->lineWidth * PIXEL_SIZE + iBufOrg->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            *iBufResAux = *((Pixel*)(linePointer + j * PIXEL_SIZE));
            iBufResAux++;            
        }
    }
}

//  Copia una sección de una matriz a un buffer (reintroduce el padding si fuera necesario)
void copy_matrix_Pixel(Pixel* iBufOrg, ImageBuffer* iBufRes,  ImageBlock* imageBlock) {
    Pixel* iBufOrgAux = iBufOrg;
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* linePointer = iBufRes->mappedData + i * (iBufRes->lineWidth * PIXEL_SIZE + iBufRes->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            *((Pixel*)(linePointer + j * PIXEL_SIZE)) = *iBufOrgAux;
            iBufOrgAux++;
        }
    }
}

// Dispersa la submatrices entre los procesos
void dispers_matrix(ImageBlockSet* imageBlockOrg, ImageBlockSet* imageBlockAssigned, ImageBuffer * imageBufferOrg, ImageBufferSet * imageBufferRecv, MPI_Datatype MPI_pixel, MPI_Datatype * matrixTypesArray, int * displs, int * sendcounts){
    if (rank == 0) {

        MPI_Request* block_request = new MPI_Request[imageBlockOrg->blockNumber];
        MPI_Status* status_request = new MPI_Status[imageBlockOrg->blockNumber];

        // Gestión individual de los bloques del proceso 0 
        int block_send_counter = 0;
        while (block_send_counter < sendcounts[0]) {
            copy_matrix_buffer(imageBufferOrg, (Pixel*)imageBufferRecv->imageBuffers[block_send_counter].mappedData, &(imageBlockOrg->imageBlocks[block_send_counter]));
            block_send_counter++;
        }

        for (size_t i = 0; i < imageBlockAssigned->blockNumber; i++)
        {
            if (verbose)
                printf("I am process %d, and I have received the block %d, %d, %d, %d \n", rank, imageBlockAssigned->imageBlocks[i].blockLength, imageBlockAssigned->imageBlocks[i].blockWidth, imageBlockAssigned->imageBlocks[i].pixelStartX, imageBlockAssigned->imageBlocks[i].pixelStartY);
        }

        if (verbose)
            printf("Starting to send the matrix to other processes\n");

        // Envío de las submatrices a los procesos correspondientes
        for (int i = 1; i < processNumber; i++) {
            for (int j = 0; j < sendcounts[i]; j++) {
                MPI_Isend(imageBufferOrg->mappedData, 1, matrixTypesArray[block_send_counter], i, block_send_counter, MPI_COMM_WORLD, &(block_request[block_send_counter]));
                block_send_counter++;
            }
        }
        // Espera para asegurar que todos los procesos han recibido la submatrices
        MPI_Waitall(imageBlockOrg->blockNumber - sendcounts[rank], &(block_request[sendcounts[rank]]), &(status_request[sendcounts[rank]]));

        delete [] block_request;
        delete [] status_request;
    }
    else {
        MPI_Request* block_request = new MPI_Request[imageBlockAssigned->blockNumber];
        MPI_Status* status_request = new MPI_Status[imageBlockAssigned->blockNumber];

        // Recibimiento de las sumatrices enviadas por el proceso 0
        for (int i = 0; i < imageBlockAssigned->blockNumber; i++) {
            MPI_Irecv(imageBufferRecv->imageBuffers[i].mappedData, imageBufferRecv->imageBuffers[i].lineNumber * imageBufferRecv->imageBuffers[i].lineWidth,
                MPI_pixel, 0, displs[rank] + i, MPI_COMM_WORLD, &(block_request[i]));
        }

        // Espera para asegurar que se ha recibido la submatriz
        MPI_Waitall(imageBlockAssigned->blockNumber, block_request, status_request);
        for (size_t i = 0; i < imageBlockAssigned->blockNumber; i++)
        {
            if (verbose)
                printf("I am process %d, and I have received the block %d, %d, %d, %d \n", rank, imageBlockAssigned->imageBlocks[i].blockLength, imageBlockAssigned->imageBlocks[i].blockWidth, imageBlockAssigned->imageBlocks[i].pixelStartX, imageBlockAssigned->imageBlocks[i].pixelStartY);
        }

        delete [] block_request;
        delete [] status_request;
    }
    
}

// Recolecta la submatrices entre los procesos
ImageBuffer*  recolect_matrix(ImageBlockSet* imageBlockOrg, ImageBlockSet* imageBlockAssigned, ImageBuffer* imageBufferOrg, ImageBufferSet* imageBufferSend, MPI_Datatype MPI_pixel, MPI_Datatype* matrixTypesArray, int* displs, int* sendcounts) {
    if (rank == 0) {
        MPI_Request* block_request = new MPI_Request[imageBlockOrg->blockNumber];
        MPI_Status* status_request = new MPI_Status[imageBlockOrg->blockNumber];

        ImageBuffer* imRes = createBuffer(imageBufferOrg->lineNumber, imageBufferOrg->lineWidth, 0);

        // Copia los datos de los bloques del proceso 0
        int block_received_counter = 0;
        for (int i = 0; i < imageBlockAssigned->blockNumber; i++) {
            copy_matrix_Pixel((Pixel*)(imageBufferSend->imageBuffers[i].mappedData), imRes, &(imageBlockAssigned->imageBlocks[i]));
            block_received_counter++;
        }

        // Pide el recibimiento de parte de las matrices al resto de procesos
        for (int i = 1; i < processNumber; i++) {
            for (int j = 0; j < sendcounts[i]; j++) {
                MPI_Irecv(imRes->mappedData, 1, matrixTypesArray[block_received_counter], i, block_received_counter, MPI_COMM_WORLD, &(block_request[block_received_counter]));
                block_received_counter++;
            }
        }
        // Espera al recibimiento de los datos
        MPI_Waitall(imageBlockOrg->blockNumber - imageBlockAssigned->blockNumber, &(block_request[imageBlockAssigned->blockNumber]), &(status_request[imageBlockAssigned->blockNumber]));
        
        delete [] block_request;
        delete [] status_request;
        return imRes;

    }
    else {
        MPI_Request* block_request = new MPI_Request[imageBlockAssigned->blockNumber];
        MPI_Status* status_request = new MPI_Status[imageBlockAssigned->blockNumber];

        // Se envía la submatriz al proceso 0
        for (int i = 0; i < imageBlockAssigned->blockNumber; i++) {
            MPI_Isend(imageBufferSend->imageBuffers[i].mappedData, imageBufferSend->imageBuffers[i].lineNumber * imageBufferSend->imageBuffers[i].lineWidth, MPI_pixel, 0, displs[rank] + i, MPI_COMM_WORLD, &(block_request[i]));
        }
        // Se espera a que se el proceso 0 reciba la submatriz
        MPI_Waitall(imageBlockAssigned->blockNumber, block_request, status_request);
        for (size_t i = 0; i < imageBlockAssigned->blockNumber; i++)
        {
            if (verbose)
                printf("I am process %d, and I have received the block %d, %d, %d, %d \n", rank, imageBlockAssigned->imageBlocks[i].blockLength, imageBlockAssigned->imageBlocks[i].blockWidth, imageBlockAssigned->imageBlocks[i].pixelStartX, imageBlockAssigned->imageBlocks[i].pixelStartY);
        }

        delete [] block_request;
        delete [] status_request;
        return NULL;
    }

}

// Gestión para ambos algoritmos
ImageBufferSet* processImageBuffer(ImageBufferSet* imageBuffer, ImageBufferSet* imageFusionTest, ImageBlockSet* imageBlockSet, int stepFlag, const char* saveStep,
    ImageBuffer* imageBufferOrg, ImageBlockSet* blocksAssigned, ImageBlockSet* blocksOrg, MPI_Datatype MPI_pixel, MPI_Datatype* matrixTypesArray, int* displs, int* sendcounts, int org_width, int org_height) {
    if (verbose && rank == 0)
        std::cout << std::endl << "Starting exposure correction" << std::endl;
    // Procesamiento de exposición
    ImageBufferSet* imageExposureCorrected = processExposureCorrection(imageBuffer, imageFusionTest, imageBlockSet, org_width, org_height);
    if (stepFlag) { // Si se pedía se guarda la imagen intermedia
        if (verbose)
            std::cout << "Saving image" << std::endl;
        if (rank == 0) {
            // Recolección de submatrices 
            ImageBuffer* imRes = recolect_matrix(blocksOrg, blocksAssigned, imageBufferOrg, imageExposureCorrected, MPI_pixel, matrixTypesArray, displs, sendcounts);
            if (verbose)
                std::cout << std::endl << "Saving Image" << std::endl;
            // Guardado de la imagen
            int result = stbi_write_png(saveStep, width, height, channels, imRes->mappedData, width * channels);
            if (result) {
                if (verbose)
                    std::cout << std::endl << "Completed image transformation" << std::endl;
            }
            else {
                std::cerr << "Error al guardar la imagen" << std::endl;
            }
            // Liberación del buffer intermedio
            imRes = deleteBuffer(imRes);
        }
        else {
            // Recolección de submatrices del proceso 0
            recolect_matrix(NULL, blocksAssigned, NULL, imageExposureCorrected, MPI_pixel, NULL, displs, sendcounts);
        }

    }

    if (verbose && rank == 0)
        std::cout << std::endl << "Starting color correction" << std::endl;
    // Procesamiento del color de la imagen
    ImageBufferSet* imageColorCorrected = processColorCorrection(imageExposureCorrected, imageBlockSet);
    if (imageExposureCorrected != imageBuffer) { // Si el proceso requirio crar nuevas 
        deleteBuffers(imageExposureCorrected);
    }
    return imageColorCorrected;
}


#define PROCESS_COMPLETE    0
#define PROCESS_COLOR       1
#define PROCESS_EXPOSITION  2

#define CORRECT 0
#define HELP    1
#define ERROR   2

int main(int argc, char* argv[]) {
    const char* fileInput1 = NULL;
    const char* fileInput2 = NULL;
    const char* fileStepOutput = NULL;
    const char* fileOutput = "output.png";
    verbose = 0;
    
    double ti, tf;
    int xDimensions = 0;
    int yDimensions = 0;

    int first = 1;
    int type = PROCESS_COMPLETE; // Rechange in all places
    
    int retv = CORRECT;
    
    bool test = false;
    int test_x = 0;
    int test_y = 0;

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    int fusion = 0;
    int stepFlag = 0;
    int force_pad = 0;

    
    // Inicialización de MPI
    MPI_Init(&argc, &argv);
    MPI_Get_processor_name(name, &len);
    MPI_Comm_size(MPI_COMM_WORLD, &processNumber);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    uint8_t* image1 = NULL;
    uint8_t* image2 = NULL;
    
    int numThreads = (omp_get_max_threads() + processNumber - 1) / processNumber;

    // Procesamiento de las variables de entrada, realizado solo por el proceso 0
    if (rank == 0) { // Proceso raiz se encargará de gestionar 
        // Gestión de parametros
        for (int i = 1; i < argc && retv == 0; i++) {
            if (argv[i][0] == '-') {
                if (argv[i] == "-h") {
                    retv = HELP;
                }
                else if (strcmp(argv[i], "-o") == 0) { // Nombre del fichero de salida
                    if (i + 1 < argc) {
                        fileOutput = argv[i + 1];
                        i++;
                    }
                    else {
                        std::cerr << "-o Expected file after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-s") == 0) { // Nombre del segundo fichero de entrada
                    if (i + 1 < argc) {
                        fileInput2 = argv[i + 1];
                        fusion = 1;
                        i++;
                    }
                    else {
                        std::cerr << "-s Expected file after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-v") == 0) { // Activación del modo verboso
                    verbose = 1;
                }
                else if (strcmp(argv[i], "-c") == 0) { // Activación del modo de corrección de color única
                    if (type == PROCESS_COMPLETE)
                        type = PROCESS_COLOR;
                    else {
                        std::cerr << "-c is incompatible with -e" << std::endl;
                        retv = ERROR;
                    }

                }
                else if (strcmp(argv[i], "-e") == 0) { // Activación del modo de corrección de exposición única
                    if (type == PROCESS_COMPLETE)
                        type = PROCESS_EXPOSITION;
                    else {
                        std::cerr << "-e is incompatible with -c" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-f") == 0) { // Activación del guardado intermedio
                    if (i + 1 < argc) {
                        fileStepOutput = argv[i + 1];
                        stepFlag = 1;
                        i++;
                    }
                    else {
                        std::cerr << "-f Expected file after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-x") == 0) { // Número de divisiones de X
                    if (i + 1 < argc) {
                        xDimensions = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-x Expected int after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-t") == 0) { // Número de hilos a crear
                    if (i + 1 < argc) {
                        numThreads = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-t Expected int after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i], "-y") == 0) { // Número de divisiones de Y
                    if (i + 1 < argc) {
                        yDimensions = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-y Expected int after it" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i],"--force-padding") == 0) { // Número de divisiones de Y
                    if (i + 1 < argc) {
                        force_pad = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-Expected int after force-padding" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i],"--test-x") == 0) { // Número de divisiones de Y
                    if (i + 1 < argc) {
                        test_x = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-Expected int after test-x" << std::endl;
                        retv = ERROR;
                    }
                }
                else if (strcmp(argv[i],"--test-y") == 0) { // Número de divisiones de Y
                    if (i + 1 < argc) {
                        test_y = atoi(argv[i + 1]);
                        i++;
                    }
                    else {
                        std::cerr << "-Expected int after test-y" << std::endl;
                        retv = ERROR;
                    }
                }
                
                else if (strcmp(argv[i], "--test") == 0) { // Nombre del fichero de salida
                    test = true;
                }
                else {
                    retv = ERROR;
                }
            }
            else {
                if (first) {
                    first = 0;
                    fileInput1 = argv[i];
                }
                else {
                    std::cerr << "Only one file is expected to be pass without a specific option" << std::endl;
                    retv = ERROR;
                }
            }
        }

        // Manejo de errores
        if(!test){
            if (first) { 
                std::cerr << "It is expected that the first input file is passed" << std::endl;
                retv = ERROR;
            }

            if (type == PROCESS_EXPOSITION && fileInput2 == NULL){
                std::cerr << "For exposure correction only second input is needed" << std::endl;
                retv = ERROR;
            }

            if (type == PROCESS_COLOR && fileInput2 != NULL) {
                std::cerr << "For color correction second input is not allowed" << std::endl;
                retv = ERROR;
            }
            if (type != PROCESS_COMPLETE && fileStepOutput != NULL) {
                std::cerr << "Saving Image between corrections is incompatible with single correction modes" << std::endl;
                retv = ERROR;
            }
        } else {
            if (test_x < 0 || test_y < 0) {
                std::cerr << "Dimensions of test matrix must be a positive number greater than 0" << std::endl;
                retv = ERROR;
            }
            
            if ((xDimensions >= test_x) || (yDimensions >= test_y)) {
                std::cerr <<  "Width and Height of test matrix must be bigger than the dimensions specified" << std::endl;
                retv = ERROR;
            }
        }

        
        if(numThreads <= 0){
            std::cerr << "Number of threads must be greater than 0" << std::endl;
            retv = ERROR;
        }
        if (xDimensions <= 0 || yDimensions <= 0) {
            std::cerr << "Dimensions must be positive number greater than 0" << std::endl;
            retv = ERROR;
        }
        if(force_pad < 0){
            std::cerr << "Force padding must be equal or greater than 0" << std::endl;
            retv = ERROR;
        }
        if (processNumber > xDimensions * yDimensions) {
            std::cerr << "Number of processes must be equal or lower than the number of blocks" << std::endl;
            retv = ERROR;
        }

        if (retv != CORRECT) {
            std::cout << argv[0] << " <input> -x <int> -y <int>" << std::endl
                << "\t-h : Help" << std::endl
                << "\t-o <string>: Output file name" << std::endl
                << "\t-s <string>: Second input file name" << std::endl
                << "\t-v : Verbose mode" << std::endl
                << "\t-f <string>: Step Output" << std::endl
                << "\t-x <int>: Dimension X divisions" << std::endl
                << "\t-y <int>: Dimension Y divisions" << std::endl
                << "\t-c : Color correction only (-s option must not be included)" << std::endl
                << "\t-e : Exposure correction only (-s option must be included to)" << std::endl
                << "\t-t <int> : Number of threads to be launch" << std::endl
                << "\t--force-padding <int> : Number of Pixels which that are padding" << std::endl;

            goto result_param; // ALL PROCESS EXIT !!!
        }
        if(!test){
            // Carga de la primera imagen
            image1 = stbi_load(fileInput1, &width, &height, &channels, 0);
            if (image1 == NULL) {
                std::cerr << "Could not load image 1" << std::endl;
                
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }
            else {
                if (verbose) {
                    std::cout << "File 1 Name : " << fileInput1 << std::endl;
                    std::cout << "Width : " << width << std::endl;
                    std::cout << "Height : " << height << std::endl;
                    std::cout << "Channels : " << channels << std::endl << std::endl;
                }
            }

            if (height < yDimensions || width < xDimensions) {
                stbi_image_free(image1);
                std::cerr << "Width and Height must be bigger than the dimensions specified" << std::endl;
                
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }

            if(width - force_pad <= 0){
                stbi_image_free(image1);
                std::cerr << "Force_pad needs to be lower than the width" << std::endl;
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }

            if ((width - force_pad) * height < 2 * xDimensions * yDimensions)  {
                stbi_image_free(image1);
                std::cerr << "Pixels per block must be at least 2" << std::endl;
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }
        
            // Carga de la segunda imagen
            
            if (fileInput2 != NULL) {
                int width2, height2, channels2;
                image2 = stbi_load(fileInput2, &width2, &height2, &channels2, 0);
                if (image2 == NULL) {
                    stbi_image_free(image1);
                    std::cerr << "Could not load image 2" << std::endl;
                    
                    retv = ERROR;
                    goto result_param; // ALL PROCESS EXIT !!!
                }
                else {
                    if (verbose) {
                        std::cout << "File 2 Name : " << fileInput2 << std::endl;
                        std::cout << "Width : " << width2 << std::endl;
                        std::cout << "Height : " << height2 << std::endl;
                        std::cout << "Channels : " << channels2 << std::endl << std::endl;
                    }
                }

                // Comprobación que las imágenes tienen las mismas dimensiones
                if (width != width2 || height != height2 || channels != channels2 || channels != CHANNEL_NUMBER) {
                    stbi_image_free(image1);
                    stbi_image_free(image2);
                    std::cerr << "Images´ format didnt have four channels" << std::endl;
                    
                    retv = ERROR;
                    goto result_param; // ALL PROCESS EXIT !!!
                }
            }
            width = width - force_pad;

        } else {
            image1 = (uint8_t*) malloc( test_x * test_y * sizeof(struct Pixel));
            image2 = (uint8_t*) malloc( test_x * test_y * sizeof(struct Pixel));

            if(image1 == NULL || image2 == NULL){
                std::cerr << "Could not load image 2" << std::endl;
                    
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }
            
            channels = CHANNEL_NUMBER;
            width = test_x;
            height = test_y;
            fileStepOutput = NULL;
            type = PROCESS_COMPLETE;
            stepFlag = 0;
            
            if(width - force_pad <= 0){
                free(image1);
                free(image2);
                std::cerr << "Force_pad needs to be lower than the width" << std::endl;
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }

            if ((width - force_pad) * height < 2 * xDimensions * yDimensions)  {
                free(image1);
                free(image2);
                std::cerr << "Pixels per block must be at least 2" << std::endl;
                retv = ERROR;
                goto result_param; // ALL PROCESS EXIT !!!
            }

            width = width - force_pad;

            fill_buffer(image1, width, height);
            fill_buffer(image2, width, height);

        }
    }

 
result_param:
    // Redistribución de resultado de gestionar los parametros de entrada y finalización correcta
    MPI_Bcast(&retv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (retv != CORRECT)
    {
        MPI_Finalize();
        if (retv == HELP)
            return EXIT_SUCCESS;
        else
            return EXIT_FAILURE;
    }

    MPI_Bcast(&xDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&verbose, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&force_pad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fusion, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stepFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(verbose)
        printf("< %s >: process %d of %d\n", name, rank, processNumber);

    MPI_Barrier(MPI_COMM_WORLD);
    ti = MPI_Wtime();
    
    ImageBuffer* im1 = NULL;
    ImageBuffer* im2 = NULL;
    ImageBlockSet* ibs = NULL;
    if (rank == 0) {

        // Creación de los buffers de memoria
        im1 = createBufferNoAlloc(height, width, PIXEL_SIZE * force_pad);
        im1->mappedData = image1;

        
        if (fileInput2 != NULL) {
            im2 = createBufferNoAlloc(height, width, PIXEL_SIZE * force_pad);
            im2->mappedData = image2;
        }

        // División del tamaño en bloques
        ibs = createBlocks(im1, xDimensions, yDimensions);

        if (verbose)
            std::cout << std::endl << "Starting Sending Images Blocks" << std::endl;

    }

    if (verbose && rank == 0)
        printf("Dividing blocks between process\n");
    // Cálculo del número de bloques asignados al proceso
    int number_blocks = xDimensions * yDimensions;
    int assigned_blocks = number_blocks / processNumber + (number_blocks % processNumber > rank ? 1 : 0);
    
    // Reseva de memoria para guardar los bloques asignados
    ImageBlockSet* imageBlockAssigned = new ImageBlockSet;
    imageBlockAssigned->blockNumber = assigned_blocks;
    imageBlockAssigned->imageBlocks = new ImageBlock[assigned_blocks];

    // Cálculo del número de bloques asignados a todos los procesos
    int* sendcounts, * displs;
    sendcounts = new int[processNumber];
    displs = new int[processNumber];

    int sum = 0;
    for (int i = 0; i < processNumber; i++) {
        sendcounts[i] = number_blocks / processNumber + (number_blocks % processNumber > i ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }
    if (verbose && rank == 0)
        printf("Preparing to be send\n");

    // Creación de los tipos de MPI
    MPI_Datatype MPI_block_info, MPI_block, MPI_pixel;
    create_block_info_MeanDeviation_type(&MPI_block_info);
    create_block_type(&MPI_block, MPI_block_info);
    create_pixel_type(&MPI_pixel);
    
    // Asignación de bloques a los procesos
    if (rank == 0) {
        MPI_Scatterv(ibs->imageBlocks, sendcounts, displs, MPI_block, imageBlockAssigned->imageBlocks, sendcounts[rank], MPI_block, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(NULL, sendcounts, displs, MPI_block, imageBlockAssigned->imageBlocks, sendcounts[rank], MPI_block, 0, MPI_COMM_WORLD);
    }

    // Liberación de tipos de MPi no usados
    MPI_Type_free(&MPI_block_info);
    MPI_Type_free(&MPI_block);

    for (size_t i = 0; i < imageBlockAssigned->blockNumber && verbose; i++)
    {
        printf("I am process %d, I have been assigned the block %d, %d, %d, %d \n", rank, imageBlockAssigned->imageBlocks[i].blockLength, imageBlockAssigned->imageBlocks[i].blockWidth, imageBlockAssigned->imageBlocks[i].pixelStartX, imageBlockAssigned->imageBlocks[i].pixelStartY);
    }


    // Repartición de Submatrices de los bloques de la imagen asignados
    ImageBufferSet* imageBufferSet = createBuffers(imageBlockAssigned);
    ImageBufferSet* imageBufferSet_image2 = NULL;
    if (fusion) {
        imageBufferSet_image2 = createBuffers(imageBlockAssigned);
    }

    MPI_Datatype* matrixTypesArray = NULL;
    MPI_Datatype* matrixTypesArray_paddless = NULL;
    if (rank == 0) {
        // Creación de los tipos especificos para el envío de la submatriz
        matrixTypesArray = new MPI_Datatype[ibs->blockNumber];
        matrixTypesArray_paddless = new MPI_Datatype[ibs->blockNumber];
        for (int i = 0; i < ibs->blockNumber; i++) {
            create_submatrix_type(im1->lineNumber, im1->lineWidth + im1->lineStride / PIXEL_SIZE, 
                ibs->imageBlocks[i].blockLength, ibs->imageBlocks[i].blockWidth,
                ibs->imageBlocks[i].pixelStartY, ibs->imageBlocks[i].pixelStartX,
                &(matrixTypesArray[i]), MPI_pixel);
            create_submatrix_type(im1->lineNumber, im1->lineWidth, 
                ibs->imageBlocks[i].blockLength, ibs->imageBlocks[i].blockWidth,
                ibs->imageBlocks[i].pixelStartY, ibs->imageBlocks[i].pixelStartX,
                &(matrixTypesArray_paddless[i]), MPI_pixel);
        }
	    // Dispersión de la matriz a los procesos
        dispers_matrix(ibs, imageBlockAssigned, im1, imageBufferSet, MPI_pixel, matrixTypesArray, displs, sendcounts);
        if (fusion) dispers_matrix(ibs, imageBlockAssigned, im2, imageBufferSet_image2, MPI_pixel, matrixTypesArray, displs, sendcounts);
    }
    else {
        // Recolección de la matriz para los procesos mayores que 0
        dispers_matrix(NULL, imageBlockAssigned, NULL, imageBufferSet, MPI_pixel, NULL, displs, sendcounts);
        if (fusion) dispers_matrix(NULL, imageBlockAssigned, NULL, imageBufferSet_image2, MPI_pixel, NULL, displs, sendcounts);
    }

    if (verbose && rank == 0)
        printf("Preparing block sets\n");

    // Creación de los bloques para el movimiento por las submatrices recibidas
    ImageBlockSet* imageblockSet = new ImageBlockSet;
    imageblockSet->blockNumber = imageBufferSet->bufferNumber;
    imageblockSet->imageBlocks = new ImageBlock[imageBufferSet->bufferNumber];
    for (int i = 0; i < imageBufferSet->bufferNumber; i++) {
        imageblockSet->imageBlocks[i].blockLength = imageBufferSet->imageBuffers[i].lineNumber;
        imageblockSet->imageBlocks[i].blockWidth = imageBufferSet->imageBuffers[i].lineWidth;
        imageblockSet->imageBlocks[i].pixelStartX = 0;
        imageblockSet->imageBlocks[i].pixelStartY = 0;
    }

    if (verbose && rank == 0)
        printf("Starting processing\n");
    // Proceso de aplicación del algoritmo
    ImageBufferSet* assigned_blocks_result = NULL;
    omp_set_num_threads(numThreads);

    if (type == PROCESS_COMPLETE) {
        if(rank == 0)
            assigned_blocks_result = processImageBuffer(imageBufferSet, imageBufferSet_image2, imageblockSet, stepFlag, fileStepOutput,im1, imageBlockAssigned, ibs, MPI_pixel, matrixTypesArray_paddless, displs, sendcounts, width, height);
        else
            assigned_blocks_result = processImageBuffer(imageBufferSet, imageBufferSet_image2, imageblockSet, stepFlag, fileStepOutput, NULL, imageBlockAssigned, NULL, MPI_pixel, NULL, displs, sendcounts, width, height);
    }
    else if (type == PROCESS_COLOR) {
        assigned_blocks_result = processColorCorrection(imageBufferSet, imageblockSet);
    }
    else if (type == PROCESS_EXPOSITION) {
        assigned_blocks_result = processExposureCorrection(imageBufferSet, imageBufferSet_image2, imageblockSet, width, height);
    }
    if (verbose && rank == 0)
        printf("Finished processing\n");
    // Liberación de Buffers y Bloques tras el proceso
    imageblockSet = deleteBlocks(imageblockSet);
    imageBufferSet = deleteBuffers(imageBufferSet);
    if (fusion) {
        imageBufferSet_image2 = deleteBuffers(imageBufferSet_image2);
    }

    if (verbose && rank == 0)
        printf("Recolecting Matrix\n");
    ImageBuffer* imRes = NULL;
    if (rank == 0) {
        // Recolección de los datos resultado
        imRes = recolect_matrix(ibs, imageBlockAssigned, im1, assigned_blocks_result, MPI_pixel, matrixTypesArray_paddless, displs, sendcounts);
    }
    else {
        // Envío de los resultados para los procesos mayores que 0
        recolect_matrix(NULL, imageBlockAssigned, NULL, assigned_blocks_result, MPI_pixel, NULL, displs, sendcounts);
    }

    if (verbose && rank == 0)
        printf("Matrix Recolected\n");

    MPI_Barrier(MPI_COMM_WORLD);
    tf = MPI_Wtime();

    if(rank == 0){
        // Guardado de la imagen resultado
        printf("\nTime = %.4lf seconds\n", tf - ti);
        if(!test){
            if (verbose)
                std::cout << std::endl << "Saving Image" << std::endl;
            int result = stbi_write_png(fileOutput, width, height, channels, imRes->mappedData, width * channels);
            if (result) {
                if (verbose)
                    std::cout << std::endl << "Completed image transformation" << std::endl;
            }
            else {
                std::cerr << "Error when saving image" << std::endl;
            }
            // Liberación de la memoria del resultado
            imRes = deleteBuffer(imRes);
        }
    }
    

    // Liberación de tipos y de memoria restante
    assigned_blocks_result = deleteBuffers(assigned_blocks_result);
    imageBlockAssigned = deleteBlocks(imageBlockAssigned);
    delete [] sendcounts;
    delete [] displs;

    if (rank == 0) {
        for (int i = 0; i < ibs->blockNumber; i++) {
            MPI_Type_free(&(matrixTypesArray[i]));
            MPI_Type_free(&(matrixTypesArray_paddless[i]));
        }
        delete [] matrixTypesArray;
        delete [] matrixTypesArray_paddless;

        if(!test){
            stbi_image_free(image1);
            im1 = deleteBufferNoAlloc(im1);
            if (im2 != NULL) {
                stbi_image_free(image2);
                im2 = deleteBufferNoAlloc(im2);
            }
        } else {
            free(image1);
            free(image2);
            im1 = deleteBufferNoAlloc(im1);
            im2 = deleteBufferNoAlloc(im1);
        }

        ibs = deleteBlocks(ibs);
    }
    MPI_Type_free(&MPI_pixel);
    // Finalización de MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}


