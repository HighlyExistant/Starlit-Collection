glslc onesweep\shaders\histogram_radix256.comp -o onesweep\build\histogram_radix256.comp.spv
glslc onesweep\shaders\scan_radix256_sg.comp --target-env=vulkan1.1 -o onesweep\build\scan_radix256_sg.comp.spv 
glslc onesweep\shaders\digit_binning_pass_sg.comp --target-env=vulkan1.1 -o onesweep\build\digit_binning_pass_sg.comp.spv 
glslc onesweep\shaders\onesweep_reset.comp -o onesweep\build\onesweep_reset.comp.spv 

glslc onesweep\shaders\keys7\histogram_radix256.comp -o                             onesweep\build\keys7\histogram_radix256.comp.spv
glslc onesweep\shaders\keys7\scan_radix256_sg.comp --target-env=vulkan1.1 -o        onesweep\build\keys7\scan_radix256_sg.comp.spv 
glslc onesweep\shaders\keys7\digit_binning_pass_sg.comp --target-env=vulkan1.1 -o   onesweep\build\keys7\digit_binning_pass_sg.comp.spv 
glslc onesweep\shaders\keys7\onesweep_reset.comp -o                                 onesweep\build\keys7\onesweep_reset.comp.spv 