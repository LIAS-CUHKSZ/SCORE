import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram, inference_ram_openset
from ram import get_transform
import torchvision.transforms as transform

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ram.utils import build_openset_llm_label_embedding


import cv2
import torchvision
import hydra
from matplotlib import pyplot as plt

import glob
import os
import json

import argparse
import os
from PIL import Image
import numpy as np


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        help="path to dataset",
        default="images",
    )
    parser.add_argument(
        "--des_file",
        help="path to description file for ram++"
    )
    parser.add_argument(
        "--output-dir",
        help="path to dataset",
        default="output",
    )
    parser.add_argument(
        "--ram-ck",
        help="path to pretrained model",
        default="pretrain_models/ram_plus_swin_large_14m.pth",
    )
    parser.add_argument(
        "--sam-ck",
        help="path to pretrained model",
        default="pretrain_models/sam2.1_hq_hiera_large.pt",
    )
    parser.add_argument(
        "-grounding-ck",
        help="path to pretrained model",
        default="pretrain_models/groundingdino_swinb_cogcoor.pth",
    )
    parser.add_argument(
        "--image-size",
        default=384,
        type=int,
        help="input image size",
    )
    parser.add_argument(
        "--dino-api",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # with hydra.initialize(version_base="1.2", config_path="."):
    #     cfg = hydra.compose(config_name="pretrain_models/sam2.1_hq_hiera_l")

    ram_normalize = transform.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    ram_transform = transform.Compose(
        [transform.Resize((384, 384)), transform.ToTensor(), ram_normalize]
    )

    ram_model = ram_plus(
        pretrained=args.ram_ck, image_size=args.image_size, vit="swin_l"
    )

    with open(args.des_file, "r") as f:
        llm_tag_des = json.load(f)

    tags_index = {}
    with open(args.des_file.replace(".json", "_index.txt"), "r") as f:
        tags = f.readlines()
        for _, tag in enumerate(tags):
            index = int(tag.strip().split(" ")[0])
            tag = tag.strip().split(" ", maxsplit=1)[1]
            tags_index[tag] = index

    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(
        llm_tag_des
    )
    ram_model.tag_list = np.array(openset_categories)
    ram_model.label_embed = torch.nn.Parameter(openset_label_embedding.float())
    ram_model.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    ram_model.class_threshold = torch.ones(ram_model.num_class) * 0.5

    ram_model.eval()
    ram_model = ram_model.to(device)
    if not args.dino_api:
        grounding_model = load_model(
            "pretrain_models/GroundingDINO_SwinB_cfg.py",
            args.grounding_ck,
            device,
        )
    else:
        pass

    sam_predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hq_hiera_l.yaml", args.sam_ck))


    images = glob.glob(os.path.join(args.image_dir, "*"))
    images = sorted(images)

    for img_path in images:
        print(f"Processing {img_path}")
        image_pil, image = load_image(img_path)
        ram_image = image_pil.resize((args.image_size, args.image_size))
        ram_image = ram_transform(ram_image).unsqueeze(0).to(device)

        ram_res = inference_ram_openset(ram_image, ram_model)

        grounding_box_threshold = 0.35
        grounding_text_threshold = 0.3
        ram_tags = ram_res.replace(" | ", ".")

        if not args.dino_api:
            boxes_filt, scores, pred_phrases = get_grounding_output(
                grounding_model,
                image,
                ram_tags,
                grounding_box_threshold,
                grounding_text_threshold,
                device=device,
            )
        else:
            prompts = dict(image=img_path, prompt=ram_tags)
            while True:
                try:
                    results = grounding_model.inference(prompts)
                    break
                except Exception as e:
                    print(e)
                    print("Retrying...")
                    continue

            boxes_filt = torch.Tensor(results["boxes"])
            scores = torch.Tensor(results["scores"])
            pred_phrases = results["categorys"]

        iou_threshold = 0.8

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        if not args.dino_api:
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
        else:
            boxes_filt[:, 0] = boxes_filt[:, 0].clamp(0, W)
            boxes_filt[:, 1] = boxes_filt[:, 1].clamp(0, H)
            boxes_filt[:, 2] = boxes_filt[:, 2].clamp(0, W)
            boxes_filt[:, 3] = boxes_filt[:, 3].clamp(0, H)

        boxes_filt = boxes_filt.cpu()

        bbox_size = (boxes_filt[:, 2:] - boxes_filt[:, :2]).prod(1)
        bbox_size_index = torch.argsort(bbox_size, descending=True)
        boxes_filt = boxes_filt[bbox_size_index]
        pred_phrases = [pred_phrases[idx] for idx in bbox_size_index]

        if boxes_filt.shape[0] == 0:
            continue

        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_filt,
            multimask_output=False,
        )

        alpha = 0.5

        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ori_image = cv2.imread(img_path)

        if not os.path.exists(args.output_dir + "_info"):
            os.makedirs(args.output_dir + "_info")

        if not os.path.exists(args.output_dir + "_mask"):
            os.makedirs(args.output_dir + "_mask")

        building_index = 0

        meta_info = {}
        meta_info["bbox"] = boxes_filt.numpy()
        meta_info["phrase"] = pred_phrases
        meta_info["mask"] = masks

        mask_output = np.zeros((H, W), dtype=np.uint8)

        # draw imaeg with masks, bbox and text using opencv
        for i in range(boxes_filt.shape[0]):
            box = boxes_filt[i]
            mask = masks[i][0] if len(masks.shape) == 4 else masks[i]
            mask_color = np.random.randint(0, 255, 3)
            mask_color = mask.reshape(
                mask.shape[0], mask.shape[1], 1
            ) * mask_color.reshape(1, 1, 3)
            img[mask > 0] = img[mask > 0] * (1 - alpha) + mask_color[mask > 0] * alpha

            cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (
                    int(box[2]),
                    int(box[3]),
                ),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                img,
                pred_phrases[i],
                (int(box[0]) + 20, int(box[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if tags_index.get(pred_phrases[i].split("(")[0].strip()) is not None:
                mask_output[mask > 0] = tags_index[
                    pred_phrases[i].split("(")[0].strip()
                ]
            else:
                print(f"New tag: {pred_phrases[i].split('(')[0].strip()}")

        np.save(
            os.path.join(
                args.output_dir + "_info", os.path.basename(img_path)[:-4] + ".npy"
            ),
            meta_info,
        )
        np.save(
            os.path.join(
                args.output_dir + "_mask", os.path.basename(img_path)[:-4] + ".npy"
            ),
            mask_output,
        )
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(img_path)), img)
