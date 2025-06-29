

// ---------------------------- Estructuras, Variables gobales y Constantes ----------------------------

// Usado por STB_IMAGE para definir las funciones que se van a usar
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <omp.h>

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
    delete imageBuffer->mappedData;
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
    delete imageBlocks->imageBlocks;
    delete imageBlocks;
    return NULL;
}


// -------------------------------------- Sección Auto Exposición --------------------------------------

// Operación para el calculo de la luminosidad de un pixel
#define RGB_TO_Y(R, G, B) ((0.299f * ((double)R)) + (0.587f * ((double)G)) + (0.114f * ((double)B)))
// Cálculo de exposición de bloque
double calculateExposition(ImageBuffer* imageBuffer, ImageBlock* imageBlock) {
    double blockLuminance = 0.0f;
    for (int i = imageBlock->pixelStartY; i < imageBlock->blockLength + imageBlock->pixelStartY; i++) {
        uint8_t* line_pointer = imageBuffer->mappedData + i * (imageBuffer->lineWidth * PIXEL_SIZE + imageBuffer->lineStride);
        for (int j = imageBlock->pixelStartX; j < imageBlock->blockWidth + imageBlock->pixelStartX; j++) {
            Pixel* pixel_p = (Pixel*) (line_pointer + j * PIXEL_SIZE);
            blockLuminance += RGB_TO_Y(pixel_p->r, pixel_p->g, pixel_p->b) / MAX_COLOR_CHANNEL_VALUE;
        }
    }
    // Media de la luminancia del bloque
    return blockLuminance / (imageBuffer->lineNumber * imageBuffer->lineWidth);
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
ImageBuffer* processExposureCorrection(ImageBuffer* imageBuffer, ImageBlockSet * imageBlockSet, ImageBuffer* imageFusionTest) {
    ImageBuffer* imageRes = NULL;
    double luminance = 0.0f;
    int iam;
    if (verbose)
        std::cout << "Calculating total exposition" << std::endl;

    #pragma omp parallel private(iam) shared(imageBuffer, imageBlockSet, imageFusionTest, imageRes)
    {

        iam = omp_get_thread_num();
        // Calculo de la exposición en la imagen haciendo uso de la luminancia
        #pragma omp for reduction(+:luminance) schedule(dynamic,1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating exposure: I am %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }
            // Se suma la luminancia de los bloques
            luminance += calculateExposition(imageBuffer, &(imageBlockSet->imageBlocks[i]));
        }

        if (verbose && iam == 0) {
           printf("Result of exposure calculation: %lf\n", luminance);
        }

        // Gestion de la fusión de imagenes
        if (imageFusionTest != NULL) { // Si se quiere realizar fusión
            #pragma omp single
            {
                if (verbose) {
                    printf("\nImage fusion starting\n");
                }
                imageRes = createBuffer(imageBuffer->lineNumber, imageBuffer->lineWidth, 0);
            } // Barrera implícita

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < imageBlockSet->blockNumber; i++) {
                if (verbose) {
                    printf("Applying fusion: I am %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                        iam, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                        imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
                }
                applyFusion(imageBuffer, imageFusionTest, imageRes, &(imageBlockSet->imageBlocks[i]));
            }

        }
        else {
            
            #pragma omp single
            {
                if (verbose) {
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
ImageBuffer* processColorCorrection(ImageBuffer* imageBuffer, ImageBlockSet* imageBlockSet) {
    
    ImageBuffer* imageBufferRes;
    if (verbose)
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

    imageBufferRes = createBuffer(imageBuffer->lineNumber, imageBuffer->lineWidth, 0);

    int iam = 0;
    #pragma omp parallel private(iam,corrR,corrG,corrB) shared(imageBuffer, imageBlockSet, imageBufferRes)
    {
        iam = omp_get_thread_num(); // Obtiene el número del thread
        // Cálculo de la suma de las desviaciones de color 
        #pragma omp for reduction(+:deviationSumR, deviationSumG, deviationSumB) schedule(dynamic, 1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating mean and deviation: I am %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }

            calculateMeanChanelValue(imageBuffer, &(imageBlockSet->imageBlocks[i]));
            calculateDeviationChanelValue(imageBuffer, &(imageBlockSet->imageBlocks[i]));
            deviationSumR += imageBlockSet->imageBlocks[i].blockinfo.deviationRed;
            deviationSumG += imageBlockSet->imageBlocks[i].blockinfo.deviationGreen;
            deviationSumB += imageBlockSet->imageBlocks[i].blockinfo.deviationBlue;
        }

        if (verbose && iam == 0) {
            printf("\nCalculating weighted average and corrections\n");
        }


        // Cálculo de la media ponderada según las desviaciones
        #pragma omp for reduction(+:stdAvgR, stdAvgG, stdAvgB) schedule(guided,5)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Calculating weighted avarage: I am %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);

            }

            stdAvgR += imageBlockSet->imageBlocks[i].blockinfo.meanRed * (imageBlockSet->imageBlocks[i].blockinfo.deviationRed / deviationSumR);
            stdAvgG += imageBlockSet->imageBlocks[i].blockinfo.meanGreen * (imageBlockSet->imageBlocks[i].blockinfo.deviationGreen / deviationSumG);
            stdAvgB += imageBlockSet->imageBlocks[i].blockinfo.meanBlue * (imageBlockSet->imageBlocks[i].blockinfo.deviationBlue / deviationSumB);
        }

        //  Cálculo de las variables de corrección de color
        if (stdAvgR != 0)
            corrR = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgR);
        if (stdAvgG != 0)
            corrG = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgG);
        if (stdAvgB != 0)
            corrB = (stdAvgR + stdAvgG + stdAvgB) / (3 * stdAvgB);
        
        if (verbose && iam == 0) {
            printf("Deviation Sum for channel Red : % f\n", deviationSumR);
            printf("Weighted Avarege for channel Red  : %f\n", stdAvgR);
            printf("Correction Value for channel Red  : %f\n", corrR);
            printf("Deviation Sum for channel Green  : %f\n", deviationSumG);
            printf("Weighted Avarege for channel Green: %f\n", stdAvgG);
            printf("Correction Value for channel Green: %f\n", corrG);
            printf("Deviation Sum for channel Blue  : %f\n", deviationSumB);
            printf("Weighted Avarege for channel Blue : %f\n", stdAvgB);
            printf("Correction Value for channel Blue : %f\n", corrB);
            printf("\nApplying color corrections\n\n");
        }
        
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < imageBlockSet->blockNumber; i++) {
            if (verbose) {
                printf("Applying Corrections: I am %d working with block : startX= %d, startY= %d, length=%d, width=%d \n",
                    iam, imageBlockSet->imageBlocks[i].pixelStartX, imageBlockSet->imageBlocks[i].pixelStartY,
                    imageBlockSet->imageBlocks[i].blockLength, imageBlockSet->imageBlocks[i].blockWidth);
            }

            applyCorrection(imageBuffer, imageBufferRes, &(imageBlockSet->imageBlocks[i]), corrR, corrG, corrB);
        }

    }

    return imageBufferRes;
}


// ------------------------------------------- Programa Main -------------------------------------------
// Gestion para ambos algoritmos
ImageBuffer* processImageBuffer(ImageBuffer* imageBuffer, ImageBlockSet* imageBlockSet, ImageBuffer* imageFusionTest, const char* saveStep) {
    if (verbose)
        std::cout << std::endl << "Starting exposure correction" << std::endl;
    ImageBuffer* imageExposureCorrected = processExposureCorrection(imageBuffer, imageBlockSet, imageFusionTest);
    if (saveStep != NULL) { // Si se pedía se guarda la imagen intermedia
        if (verbose) 
            std::cout << "Saving image" << std::endl;
        stbi_write_png(saveStep, width, height, channels, imageExposureCorrected->mappedData, width * channels);
    }
    
    if (verbose)
        std::cout << std::endl <<  "Starting color correction" << std::endl;
   
    ImageBuffer* imageColorCorrected = processColorCorrection(imageExposureCorrected, imageBlockSet);
    if (imageExposureCorrected != imageBuffer) {
        deleteBuffer(imageExposureCorrected);
    }

    if (verbose)
        std::cout << std::endl << "Finishing process" << std::endl;

    return imageColorCorrected;
}


#define PROCESS_COMPLETE    0
#define PROCESS_COLOR       1
#define PROCESS_EXPOSITION  2

int main(int argc, char* argv[]) {
    const char* fileInput1 = NULL;
    const char* fileInput2 = NULL;
    const char* fileStepOutput = NULL;
    const char* fileOutput = "output.png";
    verbose = 0;
    
    int xDimensions = 0;
    int yDimensions = 0;

    int help = 0;
    int error = 0;
    int first = 1;
    int type = PROCESS_COMPLETE;
    int numThreads = omp_get_max_threads();
    int force_pad = 0;
    double ti, tf;

    // Gestión de parametros
    for (int i = 1; i < argc && !help && !error; i++) {
        if (argv[i][0] == '-') {
            if (argv[i] == "-h") {
                help = 1;
            }
            else if (strcmp(argv[i], "-o") == 0) { // Nombre del fichero de salida
                if (i + 1 < argc) {
                    fileOutput = argv[i + 1];
                    i++;
                }
                else {
                    std::cerr << "-o Expected file after it" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i], "-s") == 0) { // Nombre del segundo fichero de entrada
                if (i + 1 < argc) {
                    fileInput2 = argv[i + 1];
                    i++;
                }
                else {
                    std::cerr << "-s Expected file after it" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i], "-v") == 0) { // Activación del modo verboso
                verbose = 1;
            }
            else if (strcmp(argv[i], "-c") == 0) { // Activación del modo de corrección de color única
                if(type == PROCESS_COMPLETE) 
                    type = PROCESS_COLOR;
                else {
                    std::cerr << "-c is incompatible with -e" << std::endl;
                    error = 1;
                }

            }
            else if (strcmp(argv[i], "-e") == 0) { // Activación del modo de corrección de exposición única
                if (type == PROCESS_COMPLETE)
                    type = PROCESS_EXPOSITION;
                else {
                    std::cerr << "-e is incompatible with -c" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i], "-f") == 0) { // Activación del guardado intermedio
                if (i + 1 < argc) {
                    fileStepOutput = argv[i + 1];
                    i++;
                }
                else {
                    std::cerr << "-f Expected file after it" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i], "-x") == 0) { // Número de divisiones de X
                if (i + 1 < argc) {
                    xDimensions = atoi(argv[i + 1]);
                    i++;
                }
                else {
                    std::cerr << "-x Expected int after it" << std::endl;
                    error = 1;
                }
            } 
            else if (strcmp(argv[i], "-t") == 0) { // Número de hilos a crear
                if (i + 1 < argc) {
                    numThreads = atoi(argv[i + 1]);
                    i++;
                }
                else {
                    std::cerr << "-x Expected int after it" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i],"-y") == 0) { // Número de divisiones de Y
                if (i + 1 < argc) {
                    yDimensions = atoi(argv[i + 1]);
                    i++;
                }
                else {
                    std::cerr << "-y Expected int after it" << std::endl;
                    error = 1;
                }
            }
            else if (strcmp(argv[i],"--force-padding") == 0) { // Número de divisiones de Y
                if (i + 1 < argc) {
                    force_pad = atoi(argv[i + 1]);
                    i++;
                }
                else {
                    std::cerr << "-Expected int after force-padding" << std::endl;
                    error = 1;
                }
            }
            else {
                error = 1;
            }
        }
        else {
            if (first) {
                first = 0;
                fileInput1 = argv[i];
            }
            else {
                std::cerr << "Only one file is expected to be pass without a specific option" << std::endl;
                error = 0;
            }
        }
        
        
    }

    // Manejo de errores
    if (first) {
        std::cerr << "It is expected that the first input file is passed" << std::endl;
        error = 1;
    }
    if (xDimensions <= 0 || yDimensions <= 0) {
        std::cerr << "Dimensions must be positive number greater than 0" << std::endl;
        error = 1;
    }
    if(force_pad < 0){
        std::cerr << "Force padding must be equal or greater than 0" << std::endl;
        error = 1;
    }
    if(numThreads <= 0){
        std::cerr << "Number of threads must be greater than 0" << std::endl;
        error = 1;
    }
    if (type == 2 && fileInput2 == NULL) {
        std::cerr << "For exposure correction only second input is needed" << std::endl;
        error = 1;
    }
    if (type == 1 && fileInput2 != NULL) {
        std::cerr << "For color correction second input is not allowed" << std::endl;
        error = 1;
    }
    if (type != 0 && fileStepOutput != NULL) {
        std::cerr << "Saving Image between corrections is incompatible with single correction modes" << std::endl;
        error = 1;
    }

    if (help || error) {
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


        if (error)
            return EXIT_FAILURE;
        else
            return EXIT_SUCCESS;
    }

    // Carga de la primera imagen
    uint8_t* image1 = stbi_load(fileInput1, &width, &height, &channels, 0);
    if (image1 == NULL) {
        std::cerr << "Could not load image 1" << std::endl;
        return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }

    if(width - force_pad <= 0){
        stbi_image_free(image1);
        std::cerr << "Force_pad needs to be lower than the width" << std::endl;
        return EXIT_FAILURE;
    }

    if ((width - force_pad) * height < 2 * xDimensions * yDimensions)  {
        stbi_image_free(image1);
        std::cerr << "Pixels per block must be at least 2" << std::endl;
        return EXIT_FAILURE;
    }

    // Carga de la segunda imagen
    uint8_t* image2 = NULL;
    if (fileInput2 != NULL) {
        int width2, height2, channels2;
        image2 = stbi_load(fileInput2, &width2, &height2, &channels2, 0);
        if (image2 == NULL) {
            stbi_image_free(image1);
            std::cerr << "Could not load image 2" << std::endl;
            return EXIT_FAILURE;
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
            return EXIT_FAILURE;
        }
    }

    width = width - force_pad;

    ti = omp_get_wtime();

    // Creación de los buffers de memoria
    ImageBuffer* im1 = createBufferNoAlloc(height, width, PIXEL_SIZE * force_pad);
    im1->mappedData = image1;
    
    ImageBuffer* im2 = NULL;
    if (fileInput2 != NULL) {
        im2 = createBufferNoAlloc(height, width, PIXEL_SIZE * force_pad);
        im2->mappedData = image2;
    }

    // División del tamaño en bloques
    ImageBlockSet* ibs = createBlocks(im1, xDimensions, yDimensions);
   
    if (verbose)
        std::cout << std::endl << "Strarting image transformation" << std::endl;

    // Procesamiento de imagenes
    ImageBuffer* imRes = NULL;
    omp_set_num_threads(numThreads);
    if (type == PROCESS_COMPLETE) {
        imRes = processImageBuffer(im1, ibs, im2, fileStepOutput);
    }
    else if (type == PROCESS_COLOR) {
        imRes = processColorCorrection(im1, ibs);
    }
    else if (type == PROCESS_EXPOSITION) {
        imRes = processExposureCorrection(im1, ibs, im2);
    }

    tf = omp_get_wtime();
    printf("\nTiempo OMP = %lf seconds\n", (tf - ti));
    // Guardado de resultados
    if (verbose)
        std::cout << std::endl << "Saving Image" << std::endl;

    int result = stbi_write_png(fileOutput, width, height, channels, imRes->mappedData, width * channels);
    if (result) {
        if (verbose)
            std::cout << std::endl << "Completed image transformation" << std::endl;
    }
    else {
        std::cerr << "Error al guardar la imagen" << std::endl;
    }
    
    // Liberación de memoria usada
    stbi_image_free(image1);
    im1 = deleteBufferNoAlloc(im1);
    if (im2 != NULL) {
        stbi_image_free(image2);
        im2 = deleteBufferNoAlloc(im2);
    }

    ibs = deleteBlocks(ibs);
    imRes = deleteBuffer(imRes);
    
    return EXIT_SUCCESS;
}


