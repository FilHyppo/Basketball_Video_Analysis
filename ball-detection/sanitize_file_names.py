import os

def sanitize_filename(filename):
    # Sostituisci i caratteri problematici con underscore
    sanitized_filename = ''.join(c if c.isalnum() or c in (' ', '.', '-', '_') else '_' for c in filename)
    return sanitized_filename

def main():
    images_dir = "C:\\Users\\marco\\OneDrive\\Desktop\\my-dataset\\all_images"
    labels_dir = "C:\\Users\\marco\\OneDrive\\Desktop\\my-dataset\\all_labels"

    for filename in os.listdir(images_dir):
        sanitized_filename = sanitize_filename(filename)
        if sanitized_filename != filename:
            old_image_path = os.path.join(images_dir, filename)
            new_image_path = os.path.join(images_dir, sanitized_filename)
            os.rename(old_image_path, new_image_path)
            print(f"Renamed {filename} to {sanitized_filename}")

        label_filename = filename.replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(os.path.join(labels_dir, label_filename)):
            old_label_path = os.path.join(labels_dir, label_filename)
            new_label_path = os.path.join(labels_dir, sanitize_filename(label_filename))
            if old_label_path != new_label_path:
                os.rename(old_label_path, new_label_path)
                print(f"Renamed {label_filename} to {sanitize_filename(label_filename)}")

if __name__ == "__main__":
    main()
