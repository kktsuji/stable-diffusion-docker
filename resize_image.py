import os
import glob
from PIL import Image


def resize_image_pil(input_path, output_path, width, height, maintain_aspect=True):
    """
    Resize image using PIL/Pillow

    Args:
        input_path (str): Path to input image
        output_path (str): Path to save resized image
        width (int): Target width
        height (int): Target height
        maintain_aspect (bool): Whether to maintain aspect ratio
    """
    try:
        with Image.open(input_path) as img:
            if maintain_aspect:
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
            else:
                img = img.resize((width, height), Image.Resampling.LANCZOS)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save with original format or convert to RGB if needed
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(output_path, quality=100, optimize=True)
            print(f"Image resized and saved to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    ori_dir = "./data/process/ctcs_ori/"
    out_dir = "./data/process/ctcs/"

    img_list = glob.glob(ori_dir + "*")

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        resize_image_pil(
            input_path=img_path,
            output_path=os.path.join(out_dir, img_name),
            width=40,
            height=40,
            maintain_aspect=True,
        )
