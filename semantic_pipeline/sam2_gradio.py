import gradio as gr
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import os
import torchvision
from openai import AzureOpenAI
import base64
from mimetypes import guess_type
from torchvision.ops import masks_to_boxes

sam2_checkpoint = "pretrain_models/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = torch.device("cuda:1")

if device.type == "cuda:1":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda:1", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.75,
    stability_score_offset=0.8,
    crop_n_layers=1,
    box_nms_thresh=0.1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25,
    use_m2m=True,
)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

sam2_img = build_sam2(model_cfg, sam2_checkpoint, device=device)
img_predictor = SAM2ImagePredictor(sam2_img)

api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("OPENAI_API_VERSION")

azure_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}",
)


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps)
    return output_path


def get_frames_from_video(video_input, sam_state):
    video_path = video_input
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    sam_state["initial_images"] = frames
    sam_state["original_images"] = frames
    sam_state["select_frame_start_index"] = 1
    sam_state["fps"] = fps

    print(f"Extracted {len(frames)} frames with shape {frames[0].shape}")

    return (
        sam_state,
        gr.update(value=sam_state["original_images"][0]),
        gr.update(maximum=len(frames), value=1),
        gr.update(maximum=len(frames), value=len(frames), minimum=2),
    )


def select_start_frame(image_start_slider, sam_state):
    sam_state["select_frame_start_index"] = image_start_slider
    sam_state["original_images"] = sam_state["initial_images"][image_start_slider - 1 :]
    return (
        sam_state,
        sam_state["original_images"][0],
        gr.update(
            maximum=len(sam_state["initial_images"]),
            value=image_start_slider + 1,
            minimum=image_start_slider + 1,
        ),
    )


def select_end_frame(image_end_slider, sam_state):
    sam_state["select_frame_end_index"] = image_end_slider
    sam_state["original_images"] = sam_state["initial_images"][
        sam_state["select_frame_start_index"] - 1 : image_end_slider - 1
    ]
    return sam_state


def auto_mask_button_click(template_frame, sam_state):
    masks = mask_generator.generate(template_frame)
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    frame = template_frame.copy()
    mask_filter = []
    for mask in sorted_masks:
        if mask["area"] < 2500:
            break
        mask_filter.append(mask)
        m = mask["segmentation"]
        color_mask = np.random.random(3) * 255
        frame[m] = color_mask * 0.5 + frame[m] * 0.5

    sam_state["current_masks"] = mask_filter.copy()
    sam_state["current_mask_frame"] = frame.copy()
    sam_state["current_mask_index"] = 1

    cv2.rectangle(
        frame,
        (int(sorted_masks[0]["bbox"][0]), int(sorted_masks[0]["bbox"][1])),
        (
            int(sorted_masks[0]["bbox"][0] + sorted_masks[0]["bbox"][2]),
            int(sorted_masks[0]["bbox"][1] + sorted_masks[0]["bbox"][3]),
        ),
        (255, 0, 0),
        2,
    )

    return sam_state, frame, gr.update(maximum=len(mask_filter), value=1)


def sam_refine(mode_selection, sam_state, point_prompt, evt: gr.SelectData):
    if mode_selection == "AutoMask":
        return (
            sam_state,
            sam_state["original_images"][0],
        )

    coordinate = [int(evt.index[0]), int(evt.index[1])]

    if sam_state["click_points"] is None:
        sam_state["click_points"] = []
        sam_state["click_points_label"] = []
        sam_state["click_mask_colors"] = np.random.random(3) * 255
        sam_state["click_mask"] = None
        sam_state["click_logit"] = None

    sam_state["click_points"].append(coordinate)
    sam_state["click_points_label"].append(1 if point_prompt == "Positive" else 0)

    template_image = sam_state["original_images"][0].copy()
    img_predictor.set_image(template_image)
    masks, scores, logits = img_predictor.predict(
        point_coords=np.array(sam_state["click_points"]),
        point_labels=np.array(sam_state["click_points_label"]),
        multimask_output=True,
        mask_input=(
            sam_state["click_logit"][None]
            if sam_state["click_logit"] is not None
            else None
        ),
    )

    max_index = np.argmax(scores)
    sam_state["click_logit"] = logits[max_index]
    sam_state["click_mask"] = masks[max_index].astype(np.bool_)

    frame = sam_state["original_images"][0].copy()
    mask = sam_state["click_mask"]
    for i in sam_state["objects"].keys():
        obj_mask = sam_state["objects"][i]["click_mask"]
        obj_color = sam_state["objects"][i]["click_mask_colors"]
        frame[obj_mask] = obj_color * 0.5 + frame[obj_mask] * 0.5

        cv2.putText(
            frame,
            sam_state["objects"][i]["obj_name"],
            (
                (
                    int(sam_state["objects"][i]["bbox"][0])
                    + int(sam_state["objects"][i]["bbox"][2])
                )
                // 2
                - 10,
                (
                    int(sam_state["objects"][i]["bbox"][1])
                    + int(sam_state["objects"][i]["bbox"][3])
                )
                // 2
                - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 36, 35),
            2,
            cv2.LINE_AA,
        )

    frame[mask] = sam_state["click_mask_colors"] * 0.5 + frame[mask] * 0.5

    return sam_state, frame


def clear_click(sam_state):
    sam_state["click_points"] = None
    sam_state["click_points_label"] = None
    sam_state["click_mask_colors"] = None
    sam_state["click_mask"] = None
    sam_state["click_logit"] = None

    frame = sam_state["original_images"][0].copy()
    for i in sam_state["objects"].keys():
        obj_mask = sam_state["objects"][i]["click_mask"]
        obj_color = sam_state["objects"][i]["click_mask_colors"]
        frame[obj_mask] = obj_color * 0.5 + frame[obj_mask] * 0.5

        cv2.putText(
            frame,
            sam_state["objects"][i]["obj_name"],
            (
                (
                    int(sam_state["objects"][i]["bbox"][0])
                    + int(sam_state["objects"][i]["bbox"][2])
                )
                // 2
                - 10,
                (
                    int(sam_state["objects"][i]["bbox"][1])
                    + int(sam_state["objects"][i]["bbox"][3])
                )
                // 2
                - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 36, 35),
            2,
            cv2.LINE_AA,
        )

    return sam_state, frame


def add_object(sam_state, objects_dropdown, objects_name, prompt_for_gpt):
    if sam_state["click_points"] is None:
        return (
            sam_state,
            objects_dropdown,
            objects_name,
            gr.update(minimum=1, maximum=len(objects_name), value=1),
        )

    obj_index = sam_state["object_index"] + 1
    bbox = (
        masks_to_boxes(torch.from_numpy(sam_state["click_mask"][None]))[0]
        .numpy()
        .astype(np.int32)
    )
    obj = sam_state["original_images"][0][
        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
    ]
    cv2.imwrite(
        f"./results/object_{obj_index:0>2}.jpg", cv2.cvtColor(obj, cv2.COLOR_RGB2BGR)
    )

    if prompt_for_gpt != "":
        tags = prompt_for_gpt.strip().split(".")
        tags = [tag.strip() for tag in tags]
        system_prompt = f"You are a very precise image annotator. Identify the part in the center of the image, and select only one tag as the label for this part: [{', '.join(tags)}], return only tag name."
    else:
        system_prompt = "You are a very precise image annotator. Identify the part in the center of the image, and select only one tag as the label for this part, return only tag name."

    print(f"Processing object {obj_index} with prompt {system_prompt}")

    try:
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": local_image_to_data_url(
                                    f"./results/object_{obj_index:0>2}.jpg"
                                )
                            },
                        }
                    ],
                },
            ],
            max_tokens=2000,
        )
        objects_name.append(response.choices[0].message.content.lower())
    except:
        objects_name.append("unknown")

    sam_state["objects"][obj_index] = {
        "click_points": sam_state["click_points"].copy(),
        "click_points_label": sam_state["click_points_label"].copy(),
        "click_mask_colors": sam_state["click_mask_colors"].copy(),
        "click_mask": sam_state["click_mask"].copy(),
        "click_logit": sam_state["click_logit"].copy(),
        "obj_name": objects_name[-1],
        "bbox": bbox,
    }

    frame = sam_state["original_images"][0].copy()
    for i in sam_state["objects"].keys():
        obj_mask = sam_state["objects"][i]["click_mask"]
        obj_color = sam_state["objects"][i]["click_mask_colors"]
        frame[obj_mask] = obj_color * 0.5 + frame[obj_mask] * 0.5

        cv2.putText(
            frame,
            sam_state["objects"][i]["obj_name"],
            (
                (
                    int(sam_state["objects"][i]["bbox"][0])
                    + int(sam_state["objects"][i]["bbox"][2])
                )
                // 2
                - 10,
                (
                    int(sam_state["objects"][i]["bbox"][1])
                    + int(sam_state["objects"][i]["bbox"][3])
                )
                // 2
                - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 36, 35),
            2,
            cv2.LINE_AA,
        )

    objects_dropdown.append(f"Object_{obj_index:0>2}")

    sam_state["object_index"] += 1
    sam_state["click_points"] = None
    sam_state["click_points_label"] = None
    sam_state["click_mask_colors"] = None
    sam_state["click_mask"] = None
    sam_state["click_logit"] = None

    return (
        sam_state,
        objects_dropdown,
        objects_name,
        gr.update(minimum=1, maximum=len(objects_name), value=1),
        frame,
    )


def remove_objects(sam_state, frame, objects_dropdown, objects_name):
    remove_objs = []

    # not remove
    if (
        len(objects_dropdown)
        == len(sam_state["objects"]) + sam_state["forward_obj_nums"]
    ):
        return (
            sam_state,
            frame,
            objects_name,
            gr.update(minimum=1, maximum=len(objects_name), value=1),
        )

    print(sam_state)
    if sam_state["forward"]:
        objects_name.clear()
    else:
        objects_name = objects_name[: sam_state["forward_obj_nums"]]

    for i in sam_state["objects"].keys():
        if f"Object_{i:0>2}" not in objects_dropdown:
            remove_objs.append(i)

    for i in range(len(remove_objs)):
        sam_state["objects"].pop(remove_objs[i])

    for i in range(sam_state["object_index"] + 1):
        if i in sam_state["objects"]:
            objects_name.append(sam_state["objects"][i]["obj_name"])

    frame = sam_state["original_images"][0].copy()
    for i in sam_state["objects"].keys():
        obj_mask = sam_state["objects"][i]["click_mask"]
        obj_color = sam_state["objects"][i]["click_mask_colors"]
        frame[obj_mask] = obj_color * 0.5 + frame[obj_mask] * 0.5

    return (
        sam_state,
        frame,
        objects_name,
        gr.update(minimum=1, maximum=len(objects_name), value=1),
    )


def change_object_name(sam_state, objects_name_slider, objects_name, replace_name):
    index = sam_state["forward_obj_nums"]
    for i in range(sam_state["object_index"] + 1):
        if i in sam_state["objects"]:
            index += 1
            if index == objects_name_slider:
                sam_state["objects"][i]["obj_name"] = replace_name
                objects_name[objects_name_slider - 1] = replace_name
                break

    frame = sam_state["original_images"][0].copy()
    for i in sam_state["objects"].keys():
        obj_mask = sam_state["objects"][i]["click_mask"]
        obj_color = sam_state["objects"][i]["click_mask_colors"]
        frame[obj_mask] = obj_color * 0.5 + frame[obj_mask] * 0.5

    return sam_state, objects_name, gr.update(value=""), frame


def clear_objects(sam_state, objects_dropdown, objects_name):
    sam_state["objects"].clear()
    objects_name.clear()
    sam_state["object_index"] = -1
    objects_dropdown.clear()
    sam_state, frame = clear_click(sam_state)

    return (
        sam_state,
        objects_dropdown,
        frame,
        objects_name,
        gr.update(minimum=0, maximum=0, value=0),
    )


def select_mask(mask_selection_slider, sam_state):
    mask = sam_state["current_masks"][mask_selection_slider - 1]
    current_mask_frame = sam_state["current_mask_frame"].copy()
    sam_state["current_mask_index"] = mask_selection_slider

    cv2.rectangle(
        current_mask_frame,
        (int(mask["bbox"][0]), int(mask["bbox"][1])),
        (
            int(mask["bbox"][0] + mask["bbox"][2]),
            int(mask["bbox"][1] + mask["bbox"][3]),
        ),
        (255, 0, 0),
        2,
    )

    return (
        sam_state,
        gr.update(value=current_mask_frame),
    )


# need to update
def predict_video_automask(videp_input, sam_state):
    basename = os.path.basename(videp_input)[:-4]

    # generate new video
    generate_video_from_frames(
        sam_state["original_images"], f"./results/{basename}.mp4", fps=sam_state["fps"]
    )

    inference_state = video_predictor.init_state(video_path=f"./results/{basename}.mp4")
    video_predictor.reset_state(inference_state)

    ann_frame_idx = 0
    ann_obj_id = 0

    bbox = sam_state["current_masks"][sam_state["current_mask_index"] - 1]["bbox"]
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]),
    )

    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    output_frames = []
    mask_color = np.random.random(3) * 255
    for frame_idx, frame in enumerate(sam_state["original_images"]):
        if frame_idx in video_segments:
            frame_mask = frame.copy()
            for obj_id, mask in video_segments[frame_idx].items():
                for index in range(mask.shape[0]):
                    frame_mask[mask[index]] = (
                        mask_color * 0.5 + frame_mask[mask[index]] * 0.5
                    )
            output_frames.append(frame_mask)

    print(f"Generated {len(output_frames)} frames with shape {output_frames[0].shape}")
    basename = os.path.basename(videp_input)
    output_video_path = generate_video_from_frames(
        output_frames, f"./results/{basename}", sam_state["fps"]
    )
    return output_video_path


def predict_video_click(videp_input, sam_state):
    basename = os.path.basename(videp_input)[:-4]
    # generate new video
    generate_video_from_frames(
        sam_state["original_images"],
        f"./results/{basename}_slice.mp4",
        fps=sam_state["fps"],
    )
    inference_state = video_predictor.init_state(
        video_path=f"./results/{basename}_slice.mp4"
    )
    video_predictor.reset_state(inference_state)

    ann_frame_idx = 0

    out_obj_ids = None
    out_mask_logits = None
    obj_colors = {}
    for i in sam_state["objects"].keys():
        ann_obj_id = i
        obj = sam_state["objects"][i]
        points = obj["click_points"]
        labels = obj["click_points_label"]
        print(
            f"Adding object {i} with {len(points)} points, its tag is {obj['obj_name']}"
        )

        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        obj_colors[i] = np.random.random(3) * 255

    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    output_frames = []
    for frame_idx, frame in enumerate(sam_state["original_images"]):
        if frame_idx in video_segments:
            frame_mask = frame.copy()
            for obj_id, mask in video_segments[frame_idx].items():
                obj_color = obj_colors[obj_id]
                for index in range(mask.shape[0]):
                    frame_mask[mask[index]] = (
                        obj_color * 0.5 + frame_mask[mask[index]] * 0.5
                    )

                    mask_tensor = torch.from_numpy(mask)
                    y, x = torch.where(mask_tensor[0] != 0)

                    if len(y) == 0:
                        continue

                    bbox = masks_to_boxes(mask_tensor)[0].numpy().astype(np.int32)
                    cv2.putText(
                        frame_mask,
                        sam_state["objects"][obj_id]["obj_name"],
                        (
                            (bbox[0] + bbox[2]) // 2 - 10,
                            (bbox[1] + bbox[3]) // 2 - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (200, 36, 35),
                        2,
                        cv2.LINE_AA,
                    )
            output_frames.append(frame_mask)

    print(f"Generated {len(output_frames)} frames with shape {output_frames[0].shape}")

    sam_state["initial_images"] = output_frames[::-1]
    sam_state["original_images"] = output_frames[::-1]
    sam_state["select_frame_start_index"] = 1
    sam_state["forward_obj_nums"] = len(sam_state["objects"])

    np.save(f"./results/{basename[:-4]}_objects.npy", sam_state["objects"])

    sam_state["objects"].clear()
    sam_state["click_points"] = None
    sam_state["click_points_label"] = None
    sam_state["click_mask_colors"] = None
    sam_state["click_mask"] = None
    sam_state["click_logit"] = None
    sam_state["forward"] = False

    frame = sam_state["original_images"][0].copy()

    output_video_path = generate_video_from_frames(
        output_frames, f"./results/{basename}", sam_state["fps"]
    )
    return (
        output_video_path,
        sam_state,
        frame,
        f"/data1/home/lucky/ram-sam/results/{basename[:-4]}_objects.npy",
    )


def update_mode_selection(mode_selection):
    if mode_selection == "AutoMask":
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )


with gr.Blocks() as iface:
    sam_state = gr.State(
        {
            "initial_images": None,
            "original_images": None,
            "current_masks": None,
            "current_mask_frame": None,
            "current_mask_index": None,
            "select_frame_start_index": None,
            "select_frame_end_index": None,
            "click_points": None,
            "click_points_label": None,
            "click_mask_colors": None,
            "objects": {},
            "object_index": -1,
            "forward": True,
            "forward_obj_nums": 0,
        }
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            mode_selection = gr.Radio(
                choices=["AutoMask", "Click"],
                value="Click",
                label="Mode",
                interactive=True,
                scale=1,
            )
            extract_frames_button = gr.Button(value="Extract Frames")
        with gr.Column():
            template_frame = gr.Image(
                label="Template Frame",
                type="numpy",
            )
            with gr.Row():
                image_start_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label="Track start frame",
                )
                image_end_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label="Track end frame",
                )
            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="Point prompt",
                    interactive=True,
                    scale=1,
                    visible=True,
                )
                with gr.Column():
                    with gr.Row():
                        clear_click_button = gr.Button(
                            value="Clear Click", visible=True
                        )
                        clear_obj_button = gr.Button(
                            value="Clear Objects", visible=True
                        )
                    with gr.Row():
                        add_obj_button = gr.Button(value="Add Object", visible=True)
                        predict_video_button_click = gr.Button(
                            value="Predict Video", visible=True
                        )

            prompt_for_gpt = gr.Textbox(
                label="Prompt Tags for GPT",
                info="Please provide the tags for GPT, like tree . dog . wall",
                visible=True,
            )
            objects_dropdown = gr.Dropdown(
                multiselect=True,
                value=[],
                allow_custom_value=True,
                label="Object selection",
                visible=True,
            )
            objects_name = gr.Dropdown(
                multiselect=True,
                info="Please do not change the order of objects name",
                value=[],
                interactive=False,
                allow_custom_value=True,
                label="Object Name",
                visible=True,
            )
            with gr.Row():
                objects_name_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    label="Select Object Index",
                    interactive=True,
                )
                replace_name = gr.Textbox(
                    label="Replace Name",
                )

            auto_mask_button = gr.Button(value="Auto Mask", visible=False)
    with gr.Row():
        with gr.Column():
            mask_frame = gr.Image(label="Mask Frame", type="numpy", visible=False)
            mask_selection_slider = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=1,
                label="Select Mask",
                visible=False,
            )
            predict_video_button_automask = gr.Button(
                value="Predict Video", visible=False
            )

        with gr.Column():
            output_video = gr.Video(label="Output Video")
            output_file = gr.File(label="segmentation file", type="filepath")

    extract_frames_button.click(
        get_frames_from_video,
        inputs=[video_input, sam_state],
        outputs=[sam_state, template_frame, image_start_slider, image_end_slider],
    )

    mode_selection.change(
        update_mode_selection,
        inputs=[mode_selection],
        outputs=[
            mask_selection_slider,
            mask_frame,
            predict_video_button_automask,
            point_prompt,
            auto_mask_button,
            clear_click_button,
            clear_obj_button,
            add_obj_button,
            predict_video_button_click,
            objects_dropdown,
            objects_name,
            prompt_for_gpt,
        ],
    )
    image_start_slider.release(
        select_start_frame,
        inputs=[image_start_slider, sam_state],
        outputs=[sam_state, template_frame, image_end_slider],
    )
    image_end_slider.release(
        select_end_frame,
        inputs=[image_end_slider, sam_state],
        outputs=[sam_state],
    )
    clear_click_button.click(
        clear_click, inputs=[sam_state], outputs=[sam_state, template_frame]
    )
    clear_obj_button.click(
        clear_objects,
        inputs=[sam_state, objects_dropdown, objects_name],
        outputs=[
            sam_state,
            objects_dropdown,
            template_frame,
            objects_name,
            objects_name_slider,
        ],
    )
    add_obj_button.click(
        add_object,
        inputs=[sam_state, objects_dropdown, objects_name, prompt_for_gpt],
        outputs=[
            sam_state,
            objects_dropdown,
            objects_name,
            objects_name_slider,
            template_frame,
        ],
    )
    objects_dropdown.change(
        remove_objects,
        inputs=[sam_state, template_frame, objects_dropdown, objects_name],
        outputs=[sam_state, template_frame, objects_name, objects_name_slider],
    )
    replace_name.submit(
        change_object_name,
        inputs=[sam_state, objects_name_slider, objects_name, replace_name],
        outputs=[sam_state, objects_name, replace_name, template_frame],
    )
    auto_mask_button.click(
        auto_mask_button_click,
        inputs=[template_frame, sam_state],
        outputs=[sam_state, mask_frame, mask_selection_slider],
    )
    template_frame.select(
        sam_refine,
        inputs=[mode_selection, sam_state, point_prompt],
        outputs=[sam_state, template_frame],
    )
    mask_selection_slider.release(
        select_mask,
        inputs=[mask_selection_slider, sam_state],
        outputs=[sam_state, mask_frame],
    )
    predict_video_button_click.click(
        predict_video_click,
        inputs=[video_input, sam_state],
        outputs=[output_video, sam_state, template_frame, output_file],
    )
    predict_video_button_automask.click(
        predict_video_automask,
        inputs=[video_input, sam_state],
        outputs=[output_video],
    )
iface.launch(server_name="0.0.0.0", server_port=6009)
