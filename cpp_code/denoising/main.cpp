#include <OpenImageDenoise/oidn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <FreeImage.h>

// Функция для загрузки изображения с диска
std::vector<float> loadImage(const std::string& filename, int& width, int& height)
{
    FIBITMAP* bitmap = FreeImage_Load(FreeImage_GetFileType(filename.c_str(), 0), filename.c_str());
    if (!bitmap)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    width = FreeImage_GetWidth(bitmap);
    height = FreeImage_GetHeight(bitmap);

    std::vector<float> imageData(width * height * 3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            RGBQUAD color;
            FreeImage_GetPixelColor(bitmap, x, y, &color);
            imageData[(y * width + x) * 3 + 0] = color.rgbRed / 255.0f;
            imageData[(y * width + x) * 3 + 1] = color.rgbGreen / 255.0f;
            imageData[(y * width + x) * 3 + 2] = color.rgbBlue / 255.0f;
        }
    }

    FreeImage_Unload(bitmap);

    return imageData;
}

// Функция для сохранения изображения на диск
void saveImage(const std::string& filename, const std::vector<float>& imageData, int width, int height)
{
    FIBITMAP* bitmap = FreeImage_Allocate(width, height, 24);
    if (!bitmap)
    {
        std::cerr << "Ошибка создания изображения: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = (y * width + x) * 3;
            RGBQUAD color;
            color.rgbRed = static_cast<BYTE>(imageData[index + 0] * 255);
            color.rgbGreen = static_cast<BYTE>(imageData[index + 1] * 255);
            color.rgbBlue = static_cast<BYTE>(imageData[index + 2] * 255);
            FreeImage_SetPixelColor(bitmap, x, y, &color);
        }
    }

    if (!FreeImage_Save(FIF_JPEG, bitmap, filename.c_str(), 0))
    {
        std::cerr << "Не верный путь до входного изображения: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    FreeImage_Unload(bitmap);
}

int main()
{
    // Пути к исходному и обработанному изображениям
    std::string inputImagePath = "/your_way/sampleInput.jpg";
    std::string outputImagePath = "/your_way/sampleOutput.jpg";

    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Загрузка изображения с диска
    int width, height;
    std::vector<float> inputImage = loadImage(inputImagePath, width, height);

    float* inputBuffer = inputImage.data();

    std::vector<float> outputImage(width * height * 3);
    float* outputBuffer = outputImage.data();

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", inputBuffer, oidn::Format::Float3, width, height);
    filter.setImage("output", outputBuffer, oidn::Format::Float3, width, height);

    // Установка параметров
    filter.set("hdr", false); // Если изображение не HDR
    filter.set("srgb", false); // Если изображение в формате sRGB
    filter.set("radius", 1); // Увеличение радиуса фильтрации
    filter.set("count", 1); // Увеличение количества итераций
    filter.commit();

    filter.execute();
    
    saveImage(outputImagePath, outputImage, width, height);

    std::cout << "Результат создан по указанному пути" << std::endl;

    return 0;
}

