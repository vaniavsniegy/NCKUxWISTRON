import sys
import time
from pathlib import Path
import pypdfium2 as pdfium
from tqdm import tqdm

print("Initializing pipeline...\n")
from ultralytics import RTDETR

# Configuration
MODEL_PATH = "model.pt"
CONF = 0.5
PDF_INPUT_DIR = Path("input_pdfs")
IMAGE_WORKING_DIR = Path("working_images")
DPI_VALUE = 300
VERSION = "v2"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def convert_pdfs_to_images(input_dir, output_dir, dpi=300):
    """Convert all PDFs found recursively in input_dir to JPEG images in output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    pdf_files = list(input_path.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{input_dir}'. Skipping conversion step...")
        return

    print(f"Found {len(pdf_files)} PDF files. Converting to images...")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            parent_folder = pdf_file.parent.name.replace(" ", "_")
            pdf_name = pdf_file.stem

            pdf = pdfium.PdfDocument(str(pdf_file))
            n_pages = len(pdf)

            for page_number in range(n_pages):
                page = pdf[page_number]
                scale = dpi / 72.0
                bitmap = page.render(scale=scale, rotation=0)
                pil_image = bitmap.to_pil()

                output_filename = f"{VERSION}_{parent_folder}_{pdf_name}_{page_number + 1}.jpg"
                output_file_path = output_path / output_filename
                pil_image.save(output_file_path, "JPEG")

            pdf.close()

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")


def main():
    start_time = time.time()

    # Validate model file exists
    if not Path(MODEL_PATH).exists():
        print(f"\nError: Could not find model file '{MODEL_PATH}'")
        input("Press Enter to exit...\n")
        sys.exit(1)

    # Create required directories if they don't exist
    PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # Convert PDFs to images
    # convert_pdfs_to_images(PDF_INPUT_DIR, IMAGE_WORKING_DIR, DPI_VALUE)

    # Collect all images for inference
    image_paths = [
        str(p) for p in IMAGE_WORKING_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print("\nError: No images to process.")
        print(f"Place PDFs in '{PDF_INPUT_DIR}' or images in '{IMAGE_WORKING_DIR}'.")
        input("Press Enter to exit...\n")
        sys.exit(0)

    # Load model and run batch inference
    print(f"\nModel loaded. Found {len(image_paths)} images. Processing...")
    model = RTDETR(MODEL_PATH)

    results = model(
        source=image_paths,
        conf=CONF,
        imgsz=1280,
        save=True,
        project="",
        name="",
        exist_ok=True,
        verbose=False
    )

    # Print per-image detection summary
    print("\n--- Summary ---")
    for r in results:
        img_name = Path(r.path).name
        print(f"{img_name}: {len(r.boxes)} detection(s)")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nDone in {elapsed} seconds.")
    print(f"Results saved in: {'runs/detect/predict'}")


if __name__ == "__main__":
    main()
