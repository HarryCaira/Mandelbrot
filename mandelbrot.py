from typing import Optional
from PIL import Image
from tqdm import tqdm
import numpy as np

# constants
CONST_D = 4
CONST_N = 200
CONST_MIN_WIDTH = -2
CONST_MAX_WIDTH = 1
CONST_MIN_HEIGHT = -1
CONST_MAX_HEIGHT = 1
CONST_WIDTH_RESOLUTION = 1500
CONST_HEIGHT_RESOLUTION = 1000
CONST_WIDTH_SPACING = abs((CONST_MAX_WIDTH - CONST_MIN_WIDTH) / CONST_WIDTH_RESOLUTION)
CONST_HEIGHT_SPACING = abs(
    (CONST_MAX_HEIGHT - CONST_MIN_HEIGHT) / CONST_HEIGHT_RESOLUTION
)
CONST_DIVERGENCE_LIMIT = 2


def calc_zeta(
    zeta_n_minus_one: complex, c: complex, d: int = CONST_D
) -> Optional[complex]:
    """
    Calculates a single iteration of a single point in a multibrot set, Zn = Z(n-1)^d + c for a given point on the complex plane, c.
    :param zeta_n_minus_one: equivalent to Z(n-1) in the above equation
    :type zeta_n_minus_one: complex
    :param c: a point on a complex plane
    :type c: complex
    :param d: the generalised multibrot power
    :type d: int
    :default d: 2 - this gives the mandelbrot equation. Change d to calculate a different set
    :returns: Zn if doesn't overflow else None
    :type: Optional[complex]
    """
    zeta_n = zeta_n_minus_one ** d + c
    return zeta_n


def calc_zeta_n(
    c: complex,
    d: int = CONST_D,
    n: int = CONST_N,
    divergence_limit: float = CONST_DIVERGENCE_LIMIT,
) -> tuple[complex, int]:
    """
    Calculates the nth iteration in a multibrot set, Zn = Z(n-1)^d + c for a given point on the complex plane, c.
    Tests for divergence. If |Zn| > constant, returns (None, iteration_num)
    :param c: a point on a complex plane
    :type c: complex
    :param n: the number of iterations to compute
    :type n: int
    :returns: (Zn, n) - the resulting nth iteration (if convergent) of the computation else (Zi, i)
    :type: tuple[complex, int]
    """
    zeta = 0
    for i in range(1, n + 1):
        zeta = calc_zeta(zeta, c, d)

        # test for divergence
        if abs(zeta) > divergence_limit:
            return zeta, i
    return zeta, n


def coord_to_complex(coord: tuple[float, float]) -> complex:
    """
    Converts a coordinate into a complex number
    :param coord: the coordinate to convert
    :type coord: tuple[float, float]
    :returns: the complex conversion of the coord
    :type: complex
    """
    c = complex(*coord)
    return c


def coord_to_pixel(coord: tuple[float, float]) -> tuple[int, int]:
    x, y = coord
    pixel_x = int(abs((x - CONST_MIN_WIDTH) / CONST_WIDTH_SPACING))
    pixel_y = int(abs((y - CONST_MIN_HEIGHT) / CONST_HEIGHT_SPACING))
    return (pixel_x, pixel_y)


def map_angle_to_colour(angle: float) -> tuple[int, int, int]:
    if angle < 0:
        angle = abs(angle) + 180

    colour_val = int((765 / 360) * angle)
    num = int(colour_val / 255)
    return tuple(num * [255] + [colour_val % 255] + (2 - num) * [1])


def get_pixel_colour(zeta: complex, iteration: int) -> tuple[int, int, int]:
    angle = np.angle(zeta, deg=True)

    # converges
    if iteration == CONST_N:
        brightness_multiplier = 0

    # diverges
    else:
        brightness_multiplier = -(2 ** (-iteration - 1 / 255)) + 1

    red, green, blue = map_angle_to_colour(angle)

    colour = (
        int(brightness_multiplier * red * 1),
        int(brightness_multiplier * green * 1),
        int(brightness_multiplier * blue * 1),
    )
    return colour


def main():
    # define points
    x_points = np.linspace(CONST_MIN_WIDTH, CONST_MAX_WIDTH, CONST_WIDTH_RESOLUTION)
    y_points = np.linspace(CONST_MIN_HEIGHT, CONST_MAX_HEIGHT, CONST_HEIGHT_RESOLUTION)

    # create image
    img = Image.new(
        "RGB",
        (CONST_WIDTH_RESOLUTION + 1, CONST_HEIGHT_RESOLUTION + 1),
        color=(255, 255, 255),  # white background
    )

    # calculate and draw pixels
    for x in tqdm(x_points, total=len(x_points), desc="Plotting x points", position=0):
        for y in tqdm(
            y_points,
            total=len(y_points),
            desc="Plotting y points - happens quite fast",
            position=1,
            leave=False,
        ):
            coord = (x, y)

            c = coord_to_complex(coord)
            zeta, iteration = calc_zeta_n(c)

            colour = get_pixel_colour(zeta, iteration)
            pixel = coord_to_pixel(coord)
            img.putpixel(pixel, colour)
    img.save(f"multibrot_{CONST_D}_{CONST_N}.png")
    img.show()


if __name__ == "__main__":
    main()
