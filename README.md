The following repository contains various programs capable of making an image look better through the use of two simple algorithms.

An Auto Exposure Algorithm has been implemented based on [2], it will allow to fuse two images to create a single image that contains most of the source images´ details. The resulting image will have the mean luminosity per pixel of the source images.

The output of AE or if only one image is passed will go to the second Algorithm. A Standard Deviation-Weighted Gray World (SDWGW) from [1] was choosen. This AWB will correct the color channel differences in the images. 

5 different programs have been designed to accelerate the execution of these algorithms:

- Sequential: Applies the algorithms sequentially.
- OpenMP: Divides the calculation over several threads. The image´s pixel matrix is divided into blocks which are calculated by the threads. Shared memory is used to reduce compute time.
- OpenMP-MPI: Divides the calculation over several processes, that contain threads. A number of blocks are assigned to each process, which are then assigned to the thread. Process 0 loads and sends the blocks to each respective process. After completion the process sends their information back to the process 0. Communication between processes is reduced to the minimum to reduce compute time.
- CUDA: Uses a NVIDIA GPU to accelarate the compute through the use of CUDA. Memory is send from and to the GPU only once and there is compute-memory overlaping to reduce compute times.
- CUDA-OpenMP: Uses the GPU and CPU to its maximum. Thread 0 on the CPU will communicate and manage with the GPU, while the rest of threads will compute the number of blocks left to the CPU.

Dependencies: 

	https://github.com/libigl/libigl-stb/blob/cd0fa3fcd90325c83be4d697b00214e029f94ca3/stb_image.h
	https://github.com/libigl/libigl-stb/blob/cd0fa3fcd90325c83be4d697b00214e029f94ca3/stb_image_write.h

  For CUDA versions
	  https://github.com/kashif/cuda-workshop/tree/82228c49be5670cbf29f91bb5928090581ad05ca/cutil/inc


[1] G Zapryanov, D Ivanova and I Nikolova, “Automatic white balance algorithms fordigital stillcameras–a comparative study,” Inf. Technol. Control., 2012. 
[2] Q. K. Vuong, S.-H. Yun and S. Kim, “A new auto exposure and auto white-balance algorithm to detect high dynamic range conditions using cmos technology,” in Proceedings of the world congress on engineering and computer science San Francisco, USA: IEEE, 2008, pages 22–24. 


