import taichi as tai
from taichi.math import *
from PIL import Image
import pygame
import oidn
import numpy as np


tai.init(arch=tai.gpu, default_ip=tai.i32, default_fp=tai.f32)

pix_resolution = (700, 700)
pix_buffer = tai.Vector.field(4, float, pix_resolution)
pixels = tai.Vector.field(3, float, pix_resolution)

ray = tai.types.struct(origin=vec3, direction=vec3, color=vec3)
material = tai.types.struct(albedo=vec3, emission=vec3)
moving = tai.types.struct(position=vec3, rotation=vec3, scale=vec3, matrix=mat3)
obj = tai.types.struct(distance=float, transform=moving, material=material)

walls = obj.field(shape=8)
walls[0] = obj(
    transform=moving(vec3(0, 0.809, 0), vec3(90, 0, 0), vec3(0.2, 0.2, 0.01)),
    material=material(vec3(1, 1, 1) * 1, vec3(100)),
)
walls[1] = obj(
    transform=moving(vec3(0, 0, -1), vec3(0, 0, 0), vec3(1, 1, 0.2)),
    material=material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
walls[2] = obj(
    transform=moving(vec3(0.275, -0.55, 0.2), vec3(0, -197, 0), vec3(0.25, 0.25, 0.25)),
    material=material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
walls[3] = obj(
    transform=moving(vec3(-0.275, -0.3, -0.2), vec3(0, 112, 0), vec3(0.25, 0.5, 0.25)),
    material=material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
walls[4] = obj(
    transform=moving(vec3(1, 0, 0), vec3(0, 90, 0), vec3(1, 1, 0.2)),
    material=material(vec3(0, 1, 0) * 0.5, vec3(1)),
)
walls[5] = obj(
    transform=moving(vec3(-1, 0, 0), vec3(0, 90, 0), vec3(1, 1, 0.2)),
    material=material(vec3(1, 0, 0) * 0.5, vec3(1)),
)
walls[6] = obj(
    transform=moving(vec3(0, 1, 0), vec3(90, 0, 0), vec3(1, 1, 0.2)),
    material=material(vec3(1, 1, 1) * 0.4, vec3(1)),
)
walls[7] = obj(
    transform=moving(vec3(0, -1, 0), vec3(90, 0, 0), vec3(1, 1, 0.2)),
    material=material(vec3(1, 1, 1) * 0.4, vec3(1)),
)



@tai.func
def turn(a: vec3) -> mat3:
    sin_a, cos_a = sin(a), cos(a)
    return (
        mat3(cos_a.z, sin_a.z, 0, -sin_a.z, cos_a.z, 0, 0, 0, 1)
        @ mat3(1, 0, 0, 0, cos_a.x, sin_a.x, 0, -sin_a.x, cos_a.x)
        @ mat3(cos_a.y, 0, -sin_a.y, 0, 1, 0, sin_a.y, 0, cos_a.y)

    )


@tai.func
def dist(obj: obj, pos: vec3) -> float:
    position = obj.transform.matrix @ (pos - obj.transform.position)
    q = abs(position) - obj.transform.scale
    return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0)


@tai.func
def near_obj(p: vec3):
    index, min_dis = 0, 1e32
    for i in tai.static(range(8)):
        dis = dist(walls[i], p)
        if dis < min_dis:
            min_dis, index = dis, i
    return index, min_dis


@tai.func
def normals(obj: obj, p: vec3) -> vec3:
    e = vec2(1, -1) * 0.5773 * 0.005
    return normalize(
        e.xyy * dist(obj, p + e.xyy)
        + e.yyx * dist(obj, p + e.yyx)
        + e.yxy * dist(obj, p + e.yxy)
        + e.xxx * dist(obj, p + e.xxx)
    )


@tai.func
def ray_cast(ray: ray):
    w, s, d, cerr = 1.6, 0.0, 0.0, 1e32
    index, t, position, hit = 0, 0.005, vec3(0), False
    for _ in range(64):
        position = ray.origin + ray.direction * t
        index, distance = near_obj(position)

        ld, d = d, distance
        if ld + d < s:
            s -= w * s
            t += s
            w *= 0.5
            w += 0.5
            continue
        err = d / t
        if err < cerr:
            cerr = err

        s = w * d
        t += s
        hit = err < 0.001
        if t > 5.0 or hit:
            break
    return walls[index], position, hit


@tai.func
def sampling(normal: vec3) -> vec3:
    z = 2.0 * tai.random() - 1.0
    a = tai.random() * 2.0 * pi
    xy = sqrt(1.0 - z * z) * vec2(sin(a), cos(a))
    return normalize(normal + vec3(xy, z))


# local estimate

# оценка освещённости в заданной точке с учетом освещения от источника света, модель Блинна-Фонга
@tai.func
def estimate_radiance(source_position: vec3, direction: vec3, normal: vec3) -> vec3:
    light_position = vec3(0.0, 1.0, 0.0)
    light_color = vec3(1.0, 1.0, 1.0)
    light_intensity = 1.0

    light_direction = normalize(light_position - source_position)
    diffuse_term = max(dot(light_direction, normal), 0.0) * light_intensity * light_color

    view_direction = normalize(-direction)
    half_vector = normalize(light_direction + view_direction)
    specular_term = pow(max(dot(half_vector, normal), 0.0), 32.0) * light_intensity * light_color

    return diffuse_term + specular_term



@tai.func
def double_local_estimate(source_position: vec3, target_position: vec3, normal: vec3, num_samples: int, threshold: float) -> vec3:
    total_radiance = vec3(0.0)
    total_weight = 0.0

    # for _ in range(num_samples):
        # Генерация случайного направления луча
        # direction = hemispheric_sampling(normal) # вместо direction - target_position

    # Вычисление излучения от источника в этом направлении
    radiance = estimate_radiance(source_position, target_position, normal)

    # Вычисление веса на основе косинусного фактора
    weight = dot(target_position, normal)

    # Накопление излучения с весом
    total_radiance += radiance * weight
    total_weight += weight

    # Проверка, падает ли вес ниже порога
    # if total_weight < threshold:
    #     break

    # Нормализация излучения по общему весу
    normalized_radiance = vec3(0.0)
    if total_weight > 0:
        normalized_radiance = total_radiance / total_weight

    return normalized_radiance


@tai.func
def raytrace(ray: ray) -> ray:
    for _ in range(3):
        object, position, hit = ray_cast(ray)
        if not hit:
            ray.color = vec3(0)
            break

        normal = normals(object, position)
        ray.direction = sampling(normal)
        ray.color *= object.material.albedo
        ray.origin = position

        # вызов локальной оценки
        ray.color += double_local_estimate(position, ray.origin, normal, num_samples=1000, threshold=0.01)
        ray.direction = sampling(normal)
        ray.color *= object.material.albedo
        ray.origin = position

        intensity = dot(ray.color, vec3(0.3, 0.6, 0.1))
        ray.color *= object.material.emission
        visible = dot(ray.color, vec3(0.3, 0.6, 0.1))
        if intensity < visible or visible < 0.00001:
            break
    return ray


@tai.kernel
def level_create():
    for i in walls:
        r = radians(walls[i].transform.rotation)
        walls[i].transform.matrix = turn(r)


@tai.kernel
def render(camera_position: vec3, camera_lookat: vec3, camera_up: vec3):
    for i, j in pixels:
        z = normalize(camera_position - camera_lookat)
        x = normalize(cross(camera_up, z))
        y = cross(z, x)

        half_width = half_height = tan(radians(35) * 0.5)
        lower_left_corner = camera_position - half_width * x - half_height * y - z
        horizontal = 2.0 * half_width * x
        vertical = 2.0 * half_height * y

        uv = (vec2(i, j) + vec2(tai.random(), tai.random())) / vec2(pix_resolution)
        po = lower_left_corner + uv.x * horizontal + uv.y * vertical
        rd = normalize(po - camera_position)

        ray = raytrace(ray(camera_position, rd, vec3(1)))
        buffer = pix_buffer[i, j]
        buffer += vec4(ray.color, 1.0)
        pix_buffer[i, j] = buffer

        color = buffer.rgb / buffer.a
        color = pow(color, vec3(1.0 / 2.2))
        color = (
            mat3(0.6, 0.4, 0.05, 0.07, 1, 0.01, 0.03, 0.1, 0.8)
            @ color
        )
        color = (color * (color + 0.02) - 0.0001) / (color * (1 * color + 0.4) + 0.2)
        color = (
            mat3(1.6,  -0.5, -0.07, -0.1,  1.1, -0.006, -0.003, -0.07, 1)
            @ color
        )
        pixels[i, j] = clamp(color, 0, 1)


def save_image(image, filename):
    Image.fromarray(image).save(filename)


def denoise_image(img_noised):
    result = np.zeros_like(img_noised, dtype=np.float32)

    device = oidn.NewDevice()
    oidn.CommitDevice(device)

    denoise_filter = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(
        denoise_filter, "color", img_noised, oidn.FORMAT_FLOAT3, img_noised.shape[1], img_noised.shape[0]
    )
    oidn.SetSharedFilterImage(
        denoise_filter, "output", result, oidn.FORMAT_FLOAT3, img_noised.shape[1], img_noised.shape[0]
    )
    oidn.CommitFilter(denoise_filter)
    oidn.ExecuteFilter(denoise_filter)
    result = np.array(np.clip(result * 255, 0, 255), dtype=np.uint8)

    oidn.ReleaseFilter(denoise_filter)
    oidn.ReleaseDevice(device)

    return result


# Попиксельное сравнение
def compare_images(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Need same dimensions.")
    num_similar_pixels = np.sum(image1 == image2)
    total_pixels = image1.shape[0] * image1.shape[1] * image1.shape[2]

    similarity_percentage = (num_similar_pixels / total_pixels) * 100

    return similarity_percentage


# Сравнение расстоянием Хеллингера
def hellinger_distance(hist1, hist2):
    sqrt_diff = np.sqrt(hist1) - np.sqrt(hist2)
    distance = np.sqrt(np.sum(sqrt_diff ** 2)) / np.sqrt(2)
    return distance


def main():

    rendering_iterations = 1
    window = tai.ui.Window("rendering", pix_resolution)
    canvas = window.get_canvas()
    level_create()
    pygame.init()

    while window.running:
        render(vec3(0, 0, 3.5), vec3(0, 0, -1), vec3(0, 1, 0))
        canvas.set_image(pixels)
        window.show()
        pygame.display.set_caption(f"Cornell Box denoise {rendering_iterations}")

        if rendering_iterations % 100 == 0:
            filename = f'CornellBoxNoised.png'
            tai.tools.imwrite(pixels.to_numpy(), filename)
            img_noised = np.array(Image.open(filename), dtype=np.float32) / 255.0
            result = denoise_image(img_noised)
            save_image(result, "CornellBoxDenoised.png")
            img = Image.open("CornellBoxDenoised.png")

            screen = pygame.display.set_mode(pix_resolution)
            pygame_img = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
            screen.blit(pygame_img, (0, 0))
            pygame.display.update()

        rendering_iterations += 1


if __name__ == "__main__":
    main()
