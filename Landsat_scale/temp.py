from PIL import Image

def extract_red(input_image_path, output_image_path, red_threshold=150, other_threshold=100):
    """
    Extracts red parts of an image and saves it with a transparent background.

    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the output PNG image.
    :param red_threshold: The minimum value for the red channel to be considered "red".
    :param other_threshold: The maximum value for green and blue channels.
    """
    try:
        img = Image.open(input_image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: The file '{input_image_path}' was not found.")
        return

    width, height = img.size
    new_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for x in range(width):
        for y in range(height):
            r, g, b, a = img.getpixel((x, y))
            if r > red_threshold and g < other_threshold and b < other_threshold:
                new_img.putpixel((x, y), (r, g, b, a))

    new_img.save(output_image_path, "PNG")
    print(f"Image saved to {output_image_path}")

if __name__ == '__main__':
    # Please replace 'input.jpg' with the path to your image file.
    # The output will be 'output.png'.
    input_path = "/Users/zhengfei/Desktop/china_map.png" #<--- IMPORTANT: Change this to your image file
    output_path = "/Users/zhengfei/Desktop/output.png"
    extract_red(input_path, output_path)