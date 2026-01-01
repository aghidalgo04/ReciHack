import os
from time import time
import cv2
import torch
from PIL import Image
from groundingdino.util.inference import load_model, predict, annotate, load_image
import groundingdino.datasets.transforms as T
import supervision as sv

# Device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_images_from_folder(folder_path):
    images = []
    clases = []
    lista = os.listdir(folder_path)

    for lis in lista:
        img_path = os.path.join(folder_path, lis)
        img = cv2.imread(img_path)
        images.append(img)
        clases.append(os.path.splitext(lis)[0])

    return images, clases

def save_results(image, boxes, class_id, out_folder):
    # Norm
    xc, yc, an, al = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

    xc, yc, an, al = max(0, min(1, xc)), max(0, min(1, yc)), max(0, min(1, an)), max(0, min(1, al))

    list_info = [f"{class_id} {xc} {yc} {an} {al}"]

    time_now = str(time()).replace('.', '')

    cv2.imwrite(f"{out_folder}/{time_now}.jpg", image)

    for info in list_info:
        with open(f"{out_folder}/{time_now}.txt", 'a') as f:
            f.write(info)

def main():
    img_folder_path = "./img/organico"
    out_folder_path = './DINO/annotations'
    class_id = 0
    save_results_flag = True

    images, classes = read_images_from_folder(img_folder_path)
    num_images = len(images)

    print(f"Imagenes: {num_images}")
    print(f'Nombres: {classes}')

    home = os.getcwd()

    # Config Path
    config_path = os.path.join(home, "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

    # CheckPoint Weights
    check_point_path = './pesos/groundingdino_swint_ogc.pth'

    # Model
    model = load_model(config_path, check_point_path)

    # Prompt
    text_prompt = 'organic waste'
    box_threshold = 0.40
    text_threshold = 0.25

    for con in range(num_images):
      img = images[con]
      print("------------------//--------------------")
      print(f"Image: {classes[con]}")

      img_copy = img.copy()

      transform = T.Compose([
          T.RandomResize([800], max_size=1333),
          T.ToTensor(),
          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

      img_source = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      img_transform, _ = transform(img_source, None)
      # image_source, img_transform = load_image(img_copy)

      boxes, logits, phrases = predict(
          model=model,
          image=img_transform,
          caption=text_prompt,
          box_threshold=box_threshold,
          text_threshold=text_threshold,
          device=DEVICE)
      
      annotated_img = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
      out_frame = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

      if len(boxes) != 0:
          if save_results_flag:
              save_results(out_frame, boxes, class_id, out_folder_path)

if __name__ == "__main__":
    main()