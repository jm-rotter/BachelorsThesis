#include <luisa-compute.h>
#include <dsl/sugar.h>
#include "stb_image_write.h"

using namespace luisa;
using namespace luisa::compute;

// Save RGBA 8-bit image using stb_image_write
void save_image(const char *filename, const std::vector<std::byte> &data, uint width, uint height) {
    stbi_write_png(filename, width, height, 4, data.data(), width * 4);
}

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("cuda"); // Or "cpu"
    Stream stream = device.create_stream();

    constexpr uint width = 1024u;
    constexpr uint height = 1024u;

    // Create a FLOAT4 image on device
    Image<float> device_image = device.create_image<float>(PixelStorage::FLOAT4, width, height, 0u);

    // Convert linear RGB to sRGB (for xyz channels), alpha unchanged
    Callable linear_to_srgb = [](Float4 linear) noexcept {
        auto x = linear.xyz();
        auto srgb_xyz = select(
            1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
            12.92f * x,
            x <= 0.0031308f);
        return make_float4(srgb_xyz, linear.w);
    };

    // Kernel: fill image with gradient in sRGB space
    Kernel2D fill_image_kernel = [&linear_to_srgb](ImageFloat image) noexcept {
        Var coord = dispatch_id().xy();
        Float2 uv = make_float2(coord) / make_float2(dispatch_size().xy());
        image->write(coord, linear_to_srgb(make_float4(uv, 1.0f, 1.0f)));
    };

    auto fill_image = device.compile(fill_image_kernel);

    // Buffer to download FLOAT4 data from device
    std::vector<float> download_image_float(width * height * 4u);

    // Dispatch kernel, copy image data back, synchronize
    stream << fill_image(device_image.view(0)).dispatch(width, height)
           << device_image.copy_to(download_image_float.data())
           << synchronize();

    // Convert float [0,1] to uint8_t [0,255]
    std::vector<std::byte> download_image(width * height * 4u);
    for (size_t i = 0; i < width * height * 4u; i++) {
        float f = download_image_float[i];
        f = f < 0.f ? 0.f : (f > 1.f ? 1.f : f); // clamp
        uint8_t val = static_cast<uint8_t>(f * 255.f);
        download_image[i] = std::byte(val);
    }

    save_image("color.png", download_image, width, height);

    return 0;
}

