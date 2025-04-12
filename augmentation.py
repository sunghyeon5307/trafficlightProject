import cv2
import os
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

input_dir = '/Users/bagseonghyeon/Documents/traffic_light/color/obj_train_data'
output_dir = '/Users/bagseonghyeon/Documents/traffic_light/augmented'
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpeg'):
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(input_dir, filename)
        txt_path = os.path.join(input_dir, base + '.txt')

        image = cv2.imread(img_path)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        bboxes = []
        class_labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            class_labels.append(class_id)

        for i in range(5): 
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            out_img_path = os.path.join(output_dir, 'images', f'{base}_aug{i}.jpg')
            out_txt_path = os.path.join(output_dir, 'labels', f'{base}_aug{i}.txt')

            cv2.imwrite(out_img_path, aug_img)
            with open(out_txt_path, 'w') as f:
                for label, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{label} {' '.join(map(str, bbox))}\n")
