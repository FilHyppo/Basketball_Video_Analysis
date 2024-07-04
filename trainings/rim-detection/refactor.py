import os

def process_labels(directory):
    for subset in ["test"]:
        label_dir = os.path.join(directory, subset, 'labels')
        image_dir = os.path.join(directory, subset, 'images')
        
        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # Filter lines for class 2 and replace with class 0
            new_lines = []
            for line in lines:
                parts = line.split()
                class_id = parts[0]
                if class_id == '4':
                    parts[0] = '0'
                    new_lines.append(' '.join(parts) + '\n')

            # If there are no lines with class 2, delete the label and the image
            if not new_lines:
                os.remove(label_path)
                image_path = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))
                if os.path.exists(image_path):
                    os.remove(image_path)
            else:
                # Write the new label file
                with open(label_path, 'w') as file:
                    file.writelines(new_lines)

# Path to the main directory
main_directory = './datasets/Ball_and_rim-1'
process_labels(main_directory)
