////////////////////////////////////////////////////////////////////////////////
// Tile multiplication
////////////////////////////////////////////////////////////////////////////////


// Valor máximo del canal de color - 2^8-1
#define MAX_COLOR_CHANNEL_VALUE 255

struct Pixel {
    uint8_t r; // Canal de color rojo
    uint8_t g; // Canal de color verde
    uint8_t b; // Canal de color azul
    uint8_t stride; // Byte de desplazamiento entre pixeles según XRGB8888
};

#define RGB_Y(R,G,B) 0.299f * (double)R + 0.587f * (double)G + 0.114f * (double) B

__global__ void calculate_sum(Pixel* image, double * sum_y, int width, int height, int pixel_padding){
	extern __shared__ double sdata[];

	int tidb = threadIdx.y * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	Pixel *data = (row < height && col < width) ? &(image[row * (width + pixel_padding) + col]) : NULL;
	sdata[tidb] = (data != NULL) ? RGB_Y(data->r, data->g, data->b) : 0;

	__syncthreads();
	
	if(tidb == 0 && (blockDim.x * blockDim.y) % 2 == 1 && (blockDim.x * blockDim.y) != 1)
    	sdata[tidb] += sdata[blockDim.x * blockDim.y - 1];

    for(unsigned int s = (blockDim.x * blockDim.y)/2; s > 0; s >>= 1) {
        if (tidb < s) 
            sdata[tidb] += sdata[tidb + s];
		
        __syncthreads();
    
        if(tidb == 0 && s % 2 == 1 && s != 1)
    	    sdata[tidb] += sdata[s - 1];
    }

    if (tidb == 0)
        atomicAdd(sum_y, (sdata[0] / (width * height)));
}

__global__ void apply_fusion(Pixel* image1, Pixel * image2, Pixel* imageRes, int width, int height, int pixel_padding){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < height && col < width){
        Pixel pixel1 = image1[row * (width + pixel_padding) + col];
	    Pixel pixel2 = image2[row * (width + pixel_padding) + col];
        Pixel * pixelRes = &(imageRes[row * width + col]);
        pixelRes->r = (pixel1.r + pixel2.r) >> 1;
        pixelRes->g = (pixel1.g + pixel2.g) >> 1;
        pixelRes->b = (pixel1.b + pixel2.b) >> 1;
        pixelRes->stride = pixel1.stride;
    }
}

__global__ void calculate_mean(Pixel* image, int width, int height, int pixel_padding,
                                         double * mean_red, double * mean_green, double * mean_blue) {
	extern __shared__ double sdata[];

    double * sred = sdata;
    double * sgreen = &(sdata[blockDim.x * blockDim.y]);
    double * sblue = &(sdata[blockDim.x * blockDim.y * 2]);

    int bid =   blockIdx.y * gridDim.x + blockIdx.x;
	int tidb = threadIdx.y * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	Pixel *data = (row < height && col < width) ? &(image[row * (width + pixel_padding) + col]) : NULL;
	sred[tidb]   = (data != NULL) ? data->r : 0;
    sgreen[tidb] = (data != NULL) ? data->g : 0;
    sblue[tidb]  = (data != NULL) ? data->b : 0; 

	__syncthreads();
    if((tidb == 0) && ((( blockDim.x * blockDim.y) % 2) == 1) && (blockDim.x*blockDim.y != 1)) {
        sred[  tidb] += sred[   blockDim.x * blockDim.y - 1];
        sgreen[tidb] += sgreen[ blockDim.x * blockDim.y - 1];
        sblue[ tidb] += sblue[  blockDim.x * blockDim.y - 1];
    }

    for(unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tidb < s){ 
            sred[  tidb] += sred[  tidb + s];
            sgreen[tidb] += sgreen[tidb + s];
            sblue[ tidb] += sblue[ tidb + s];
        }
        __syncthreads();
    
        if(tidb == 0 && s % 2 == 1 && s != 1){
    	    sred[  tidb] += sred[  s - 1];
            sgreen[tidb] += sgreen[s - 1];
            sblue[ tidb] += sblue[ s - 1];
        }
    }
        
    if (tidb == 0){
        int x = (blockDim.x <= (width  - blockIdx.x * blockDim.x)) ? blockDim.x : (width  - blockIdx.x * blockDim.x) ;
        int y = (blockDim.y <= (height - blockIdx.y * blockDim.y)) ? blockDim.y : (height - blockIdx.y * blockDim.y) ;
        mean_red[bid] = sred[tidb]     / (x*y);
        mean_green[bid] = sgreen[tidb] / (x*y);
        mean_blue[bid] = sblue[tidb]   / (x*y);
    }
}

__global__ void calculate_dev(Pixel* image, int width, int height, int pixel_padding,
                              double * mean_red, double * mean_green, double * mean_blue,
                              double * dev_red, double * dev_green, double * dev_blue,
                              double * dev_red_t, double * dev_green_t, double * dev_blue_t
                             ){
    extern __shared__ double sdata[];

    double * sred = sdata;
    double * sgreen = &(sdata[blockDim.x * blockDim.y]);
    double * sblue = &(sdata[blockDim.x * blockDim.y * 2]);

    int bid  = blockIdx.y  * gridDim.x  + blockIdx.x;
    int tidb = threadIdx.y * blockDim.x + threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double mean_red_b   = mean_red[  bid];
    double mean_green_b = mean_green[bid];
    double mean_blue_b  = mean_blue[ bid];

    Pixel *data = (row < height && col < width) ? &(image[row * (width + pixel_padding) + col]) : NULL;
    sred[  tidb] = (data != NULL) ? pow(data->r - mean_red_b  ,2) : 0;
    sgreen[tidb] = (data != NULL) ? pow(data->g - mean_green_b,2) : 0;
    sblue[ tidb] = (data != NULL) ? pow(data->b - mean_blue_b ,2) : 0; 

    __syncthreads();
    if((tidb == 0) && ((( blockDim.x * blockDim.y) % 2) == 1) && (blockDim.x*blockDim.y != 1)) {
        sred[  tidb]   += sred[ blockDim.x * blockDim.y - 1];
        sgreen[tidb] += sgreen[ blockDim.x * blockDim.y - 1];
        sblue[ tidb]  += sblue[ blockDim.x * blockDim.y - 1];
    }

    for(unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tidb < s){ 
            sred[  tidb] += sred[  tidb + s];
            sgreen[tidb] += sgreen[tidb + s];
            sblue[ tidb] += sblue[ tidb + s];
        }
        __syncthreads();

        if(tidb == 0 && s % 2 == 1 && s != 1){
            sred[  tidb] += sred[  s - 1];
            sgreen[tidb] += sgreen[s - 1];
            sblue[ tidb] += sblue[ s - 1];
        }
    }

    if (tidb == 0){
        int x = (blockDim.x <= (width  - blockIdx.x * blockDim.x)) ? blockDim.x : (width  - blockIdx.x * blockDim.x) ;
        int y = (blockDim.y <= (height - blockIdx.y * blockDim.y)) ? blockDim.y : (height - blockIdx.y * blockDim.y) ;
        dev_red[  bid] = sqrt(sred[  tidb] / (x*y));
        dev_green[bid] = sqrt(sgreen[tidb] / (x*y));
        dev_blue[ bid] = sqrt(sblue[ tidb] / (x*y));
        atomicAdd(dev_red_t,   dev_red[   bid]);
        atomicAdd(dev_green_t, dev_green[ bid]);
        atomicAdd(dev_blue_t,  dev_blue[  bid]);
    }
}


__global__ void calculate_weighted_mean(double * mean_red, double * mean_green, double * mean_blue,
                                       double * dev_red, double * dev_green, double * dev_blue,
                                       double dev_red_t, double dev_green_t, double dev_blue_t,
                                       double * std_avg_r, double * std_avg_g, double * std_avg_b,  int block_number) {
    int tidg = blockIdx.x * blockDim.x + threadIdx.x;
    if(tidg < block_number){
        double local_mean_red   = mean_red[tidg];
        double local_mean_green = mean_green[tidg];
        double local_mean_blue  = mean_blue[tidg];

        double local_dev_red   = dev_red[tidg];
        double local_dev_green = dev_green[tidg];
        double local_dev_blue  = dev_blue[tidg]; 

        atomicAdd(std_avg_r, (local_mean_red   * (local_dev_red   / dev_red_t)) );
        atomicAdd(std_avg_g, (local_mean_green * (local_dev_green / dev_green_t)) );
        atomicAdd(std_avg_b, (local_mean_blue  * (local_dev_blue  / dev_blue_t)) );
    }
}

__global__ void apply_correction(Pixel * imageOrg, Pixel * imageRes, int height, int width, int pixel_padding,  double corrR, double corrG, double corrB){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width){
        Pixel *dataOrg = &(imageOrg[row * (width + pixel_padding) + col]);
        Pixel *dataRes = &(imageRes[row * width + col]);
        
        double red = dataOrg->r * corrR;
        double green = dataOrg->g * corrG;
        double blue = dataOrg->b * corrB;
            
        dataRes->r = red < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) red : MAX_COLOR_CHANNEL_VALUE;
        dataRes->g = green < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) green : MAX_COLOR_CHANNEL_VALUE;
        dataRes->b = blue < MAX_COLOR_CHANNEL_VALUE ? (uint8_t) blue : MAX_COLOR_CHANNEL_VALUE;
        dataRes->stride = dataOrg->stride;  
    }
}
