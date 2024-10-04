# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm
import json
from worker_pool import WorkerPool
torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32

class_dict = {
    0: 'Background', 1: 'Apparel', 2: 'Face_Neck', 3: 'Hair', 4: 'Left_Foot', 5: 'Left_Hand', 
    6: 'Left_Lower_Arm', 7: 'Left_Lower_Leg', 8: 'Left_Shoe', 9: 'Left_Sock', 
    10: 'Left_Upper_Arm', 11: 'Left_Upper_Leg', 12: 'Lower_Clothing', 13: 'Right_Foot', 
    14: 'Right_Hand', 15: 'Right_Lower_Arm', 16: 'Right_Lower_Leg', 17: 'Right_Shoe', 
    18: 'Right_Sock', 19: 'Right_Upper_Arm', 20: 'Right_Upper_Leg', 21: 'Torso', 
    22: 'Upper_Clothing', 23: 'Lower_Lip', 24: 'Upper_Lip', 25: 'Lower_Teeth', 
    26: 'Upper_Teeth', 27: 'Tongue'
}


def _demo_mm_inputs(batch_size, input_shape):
    (C, H, W) = input_shape
    N = batch_size
    rng = np.random.RandomState(0)
    imgs = rng.rand(batch_size, C, H, W)
    if torch.cuda.is_available():
        imgs = torch.Tensor(imgs).cuda()
    return imgs


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=model.dtype).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=model.dtype
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s

def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def img_save_and_viz(
    image, result, output_path, classes, palette, title=None, opacity=0.5, threshold=0.3, 
):
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )
    output_seg_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", "_seg.npy")
    )

    image = image.data.numpy() ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()

    mask = pred_sem_seg > 0
    np.save(output_file, mask)
    np.save(output_seg_file, pred_sem_seg)

    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    vis_image = np.concatenate([image, vis_image], axis=1)
    cv2.imwrite(output_path, vis_image)





def img_save_and_viz_per_class(
    image, result, output_path, classes,palette, name="sapiens1b", 
    opacity=0.5, threshold=0.3, frame_idx=None
):    

    frame_idx = os.path.basename(output_path).split('.')[0]  # Assumes frame number is in the name
    output_dir = os.path.dirname(output_path)
    # Ensure output directories exist
    os.makedirs(f"{output_dir}/blend_mask/visualize", exist_ok=True)
    os.makedirs(f"{output_dir}/blend_mask", exist_ok=True)

    # Example of class label to name mapping
    class_dict = {
        0: 'Background', 1: 'Apparel', 2: 'Face_Neck', 3: 'Hair', 4: 'Left_Foot', 5: 'Left_Hand', 
        6: 'Left_Lower_Arm', 7: 'Left_Lower_Leg', 8: 'Left_Shoe', 9: 'Left_Sock', 
        10: 'Left_Upper_Arm', 11: 'Left_Upper_Leg', 12: 'Lower_Clothing', 13: 'Right_Foot', 
        14: 'Right_Hand', 15: 'Right_Lower_Arm', 16: 'Right_Lower_Leg', 17: 'Right_Shoe', 
        18: 'Right_Sock', 19: 'Right_Upper_Arm', 20: 'Right_Upper_Leg', 21: 'Torso', 
        22: 'Upper_Clothing', 23: 'Lower_Lip', 24: 'Upper_Lip', 25: 'Lower_Teeth', 
        26: 'Upper_Teeth', 27: 'Tongue'
    }

    # print pallete and its length 
    # print (f"palette: {palette}")
    # print (f"palette length: {len(palette)}")

    # Convert image tensor to numpy (assuming BGR format)
    image = image.data.numpy()  # BGR image

    # Resizing segmentation result to match the input image size
    seg_logits = F.interpolate(result.unsqueeze(0), size=image.shape[:2], mode="bilinear").squeeze(0)

    # Determine if it's binary or multi-class segmentation
    if seg_logits.shape[0] > 1:
        # Multi-class segmentation: Use argmax to get the predicted class for each pixel
        pred_sem_seg = seg_logits.argmax(dim=0).cpu().numpy()  # Get predicted class (argmax over channels)
    else:
        # Binary segmentation: Use sigmoid and threshold
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits).cpu().numpy()

    # Now, create and save the argmax-colored image (palette-colored segmentation)
    num_classes = len(class_dict) 
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    # Get the colors from the palette based on labels
    colors = [palette[label] for label in labels]

    # Create a mask for visualization
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # Initialize the mask

    # Assign colors from the palette to the mask based on predicted segmentation
    for label, color in zip(labels, colors):
        mask[sem_seg == label] = np.array(color, dtype=np.uint8)  # Ensure color is an array

    # Save the colored argmax image (segmentation) and overlay with original image
    colored_output_path = f"{output_dir}/blend_mask/visualize/{str(frame_idx).zfill(5)}_{name}_argmax_colored.jpg"
    overlay_output_path = f"{output_dir}/blend_mask/visualize/{str(frame_idx).zfill(5)}_{name}_overlay.jpg"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)
    vis_concat_image = np.concatenate([image, vis_image], axis=1)

    cv2.imwrite(colored_output_path, mask)  # Save segmentation
    cv2.imwrite(overlay_output_path, vis_concat_image)  # Save overlayed image

    # Save confidence heatmap (for argmax) as both RGB and grayscale
    confidence_argmax = seg_logits.softmax(dim=0).max(dim=0).values.cpu().numpy()
    confidence_rgb = cv2.applyColorMap((confidence_argmax * 255).astype(np.uint8), cv2.COLORMAP_JET)
    confidence_gray = (confidence_argmax * 255).astype(np.uint8)

    confidence_rgb_output_path = f"{output_dir}/blend_mask/visualize/{str(frame_idx).zfill(5)}_{name}_confidence_rgb.jpg"
    confidence_gray_output_path = f"{output_dir}/blend_mask/{str(frame_idx).zfill(5)}_{name}_confidence_gray.jpg"
    cv2.imwrite(confidence_rgb_output_path, confidence_rgb)
    cv2.imwrite(confidence_gray_output_path, confidence_gray)

    # Use the class index values directly for the grayscale image. Each pixel is the id of predicted vlass.
    grayscale_image_predictions = pred_sem_seg.astype(np.float32)
    grayscale_output_path = f"{output_dir}/blend_mask/{str(frame_idx).zfill(5)}_{name}_argmax_grayscale.png"
    cv2.imwrite(grayscale_output_path, grayscale_image_predictions.astype(np.uint8))

    # Save the class dictionary to a JSON file
    class_dict_output_path = f"{output_dir}/blend_mask/class_dict.json"
    with open(class_dict_output_path, 'w') as json_file:
        json.dump(class_dict, json_file, indent=4)

    seg_logits = seg_logits.softmax(dim=0)
    # Process each class individually for binary masks and confidence maps
    for class_idx, class_name in class_dict.items():
        # Create a binary mask based on whether the argmax class matches the current class
        class_mask = (pred_sem_seg == class_idx).astype(np.uint8)  # Mark as 1 if argmax matches class_idx

        # Save binary mask (grayscale image)
        mask_output_path = os.path.join(output_dir,"blend_mask", f"{str(frame_idx).zfill(5)}_{name}_{class_name}_mask.jpg")
        cv2.imwrite(mask_output_path, class_mask * 255)  # Save as binary (0 or 255)

        # Save confidence map (grayscale image, raw logit values)
        class_prob = seg_logits[class_idx]  # Get logits for the current class
        confidence_map = class_prob.cpu().numpy()
        confidence_map_rescaled = (confidence_map * 255).astype(np.uint8)  # Rescale for visualization
        confidence_output_path = os.path.join(output_dir,"blend_mask", f"{str(frame_idx).zfill(5)}_{name}_{class_name}_confidence.jpg")
        cv2.imwrite(confidence_output_path, confidence_map_rescaled)



def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=4,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument("--title", default="result", help="The image identifier.")
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.to(args.device)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [
            os.path.basename(path) for path in image_paths
        ]  # Extract base names for image processing
        input_dir = (
            os.path.dirname(image_paths[0]) if image_paths else ""
        )  # Use the directory of the first image path
    else:
        raise ValueError("Invalid input, must be a directory or a text file")

    if len(image_names) == 0:
        raise ValueError("No images found in the input directory")

    # If left unspecified, create an output folder relative to this script.
    if args.output_root is None:
        args.output_root = os.path.join(input_dir, "output")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size
    print (f"Running inference on {len(image_names)} images in {n_batches} batches")
    print (f"image sizes {input_shape}")
    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        (input_shape[1], input_shape[2]),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()) // 2, 4),
    )
    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz_per_class, processes=max(min(args.batch_size, cpu_count()) // 2, 4)
    )
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype)

        args_list = [
            (
                i,
                r,
                os.path.join(args.output_root, os.path.basename(img_name)),
                GOLIATH_CLASSES,
                GOLIATH_PALETTE,
                args.title,
                args.opacity,
            )
            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                result[:valid_images_len],
                batch_image_name,
            )
        ]
        img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
