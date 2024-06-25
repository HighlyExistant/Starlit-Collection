glslc src\algorithms\morton\shaders\morton_ds.comp -o src\algorithms\morton\build\morton_ds.comp.spv
glslc src\algorithms\morton\shaders\morton_da.comp -o src\algorithms\morton\build\morton_da.comp.spv
glslc src\algorithms\morton\shaders\m32tom64_ds.comp -o src\algorithms\morton\build\m32tom64_ds.comp.spv
glslc src\algorithms\morton\shaders\m32tom64_da.comp -o src\algorithms\morton\build\m32tom64_da.comp.spv

glslc prefix_sum.comp -o src\algorithms\prefix_sum\shaders\prefix_sum.comp.spv

glslc src\algorithms\onesweep\shaders\histogram_radix256.comp -o src\algorithms\onesweep\build\histogram_radix256.comp.spv
glslc src\algorithms\onesweep\shaders\scan_radix256_sg.comp --target-env=vulkan1.1 -o src\algorithms\onesweep\build\scan_radix256_sg.comp.spv 
glslc src\algorithms\onesweep\shaders\digit_binning_pass_sg.comp --target-env=vulkan1.1 -o src\algorithms\onesweep\build\digit_binning_pass_sg.comp.spv 
glslc src\algorithms\onesweep\shaders\onesweep_reset.comp -o src\algorithms\onesweep\build\onesweep_reset.comp.spv 

glslc src\algorithms\device_sort\shaders\reset.comp -o src\algorithms\device_sort\build\reset.comp.spv 
glslc src\algorithms\device_sort\shaders\upsweep.comp --target-env=vulkan1.1 -o src\algorithms\device_sort\build\upsweep.comp.spv 
glslc src\algorithms\device_sort\shaders\scan.comp --target-env=vulkan1.1 -o src\algorithms\device_sort\build\scan.comp.spv 
glslc src\algorithms\device_sort\shaders\downsweep.comp --target-env=vulkan1.1 -o src\algorithms\device_sort\build\downsweep.comp.spv 

glslc src\algorithms\device_sort_pairs\shaders\reset.comp -o src\algorithms\device_sort_pairs\build\reset.comp.spv 
glslc src\algorithms\device_sort_pairs\shaders\upsweep.comp --target-env=vulkan1.1 -o src\algorithms\device_sort_pairs\build\upsweep.comp.spv 
glslc src\algorithms\device_sort_pairs\shaders\scan.comp --target-env=vulkan1.1 -o src\algorithms\device_sort_pairs\build\scan.comp.spv 
glslc src\algorithms\device_sort_pairs\shaders\downsweep.comp --target-env=vulkan1.1 -o src\algorithms\device_sort_pairs\build\downsweep.comp.spv 

glslc src\algorithms\range\shaders\range.comp -o src\algorithms\range\build\range.comp.spv

glslc src\algorithms\csdlfe\shaders\csdlfe.comp --target-env=vulkan1.1 -o src\algorithms\csdlfe\build\csdlfe.comp.spv 

