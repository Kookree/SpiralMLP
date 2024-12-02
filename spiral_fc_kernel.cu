#include <torch/extension.h>
#include <vector>
#include <cmath>

__global__ void gen_offset_cuda_kernel(
    float* offset,
    const int input_dim,
    const int num_dims,
    const float R) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_dim) return;

    int k = 2;
    int group = idx / (input_dim / k);
    int i = idx % (input_dim / k);
    
    if (i <= num_dims / 2) {
        offset[2 * idx + 0] = round(i * (R / num_dims) * cos(M_PI * i / 16));
        offset[2 * idx + 1] = round(i * (R / num_dims) * sin(M_PI * i / 16));
    } else {
        offset[2 * idx + 0] = round((R - i * (R / num_dims)) * cos(M_PI * i / 16));
        offset[2 * idx + 1] = round((R - i * (R / num_dims)) * sin(M_PI * i / 16));
    }
}

__global__ void deform_conv2d_cuda_kernel(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int input_channels,
    const int input_height,
    const int input_width,
    const int output_channels,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int offset_groups) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z / input_width;
    int w = blockIdx.z % input_width;

    int input_index = ((b * input_channels + c) * input_height + h) * input_width + w;
    float value = 0.0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int offset_h = h + offset[2 * (c * kernel_h + kh) + 0];
            int offset_w = w + offset[2 * (c * kernel_w + kw) + 1];

            if (offset_h >= 0 && offset_h < input_height && offset_w >= 0 && offset_w < input_width) {
                int offset_index = ((b * offset_groups + c / (input_channels / offset_groups)) * kernel_h + kh) * kernel_w + kw;
                int weight_index = (c * kernel_h + kh) * kernel_w + kw;
                int input_offset_index = ((b * input_channels + c) * input_height + offset_h) * input_width + offset_w;

                value += input[input_offset_index] * weight[weight_index];
            }
        }
    }

    output[input_index] = value;
}

std::vector<torch::Tensor> gen_offset_cuda(
    torch::Tensor offset,
    const int input_dim,
    const int num_dims,
    const float R) {

    const int threads = 256;
    const int blocks = (input_dim + threads - 1) / threads;

    gen_offset_cuda_kernel<<<blocks, threads>>>(
        offset.data_ptr<float>(),
        input_dim,
        num_dims,
        R);

    return {offset};
}

std::vector<torch::Tensor> deform_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor offset,
    torch::Tensor weight,
    const int batch_size,
    const int input_channels,
    const int input_height,
    const int input_width,
    const int output_channels,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int offset_groups) {
    
    auto output = torch::zeros({batch_size, output_channels, input_height, input_width}, input.options());

    const dim3 threads(1, 1, 256);
    const dim3 blocks(batch_size, output_channels, input_height * input_width);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "deform_conv2d_forward_cuda", ([&] {
        deform_conv2d_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            input_channels,
            input_height,
            input_width,
            output_channels,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            offset_groups);
    }));

    return {output};
}
