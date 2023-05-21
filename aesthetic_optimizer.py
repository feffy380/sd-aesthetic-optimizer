import argparse
import base64
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, PngImagePlugin
from transformers import logging

from aesthetic_predictor.aesthetic_predictor import AestheticPredictor

logging.set_verbosity_error()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def get_best_image(resp_json, models, model_weights):
    info = json.loads(resp_json["info"])
    infotexts = info["infotexts"]
    images = [
        Image.open(io.BytesIO(base64.b64decode(image))) for image in resp_json["images"]
    ]
    embeds = models[0].get_embeds(images)
    scores = [sigmoid(np.array(model.predict(embeds=embeds))) for model in models]
    scores = [np.prod(np.array(s) ** model_weights) for s in zip(*scores)]
    batch_best = max(scores)
    i = scores.index(batch_best)
    best_image = images[i]
    best_score = scores[i]
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", infotexts[i])

    return {
        "image": best_image,
        "image_b64": resp_json["images"][i],
        "score": best_score,
        "metadata": metadata,
        "job_timestamp": info["job_timestamp"],
        "seed": info["all_seeds"][i],
    }


def encode_pil_to_base64(pil_image):
    with io.BytesIO() as output_bytes:
        pil_image.save(output_bytes, "PNG")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return f"data:image/png;base64,{base64_str}"


def format_decimal(num):
    return f"{num:.2f}"


def generate_patches(image, patch_size, overlap=None):
    if overlap is None:
        overlap = math.ceil(patch_size / 2)
    width, height = image.size
    patches = []
    for y in range(
        0,
        math.ceil(height / patch_size) * patch_size - patch_size + 1,
        patch_size - overlap,
    ):
        for x in range(
            0,
            math.ceil(width / patch_size) * patch_size - patch_size + 1,
            patch_size - overlap,
        ):
            patch = (x, y, x + patch_size, y + patch_size)
            patches.append(patch)
    return patches


def create_mask(image, patch_set):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for patch in patch_set:
        draw.rectangle(patch, fill=255)
    return mask


def txt2img_stage(url, config, models, model_weights, outdir):
    best = None
    p = 0
    print("Starting txt2img seed search")
    while p < config.get("seed_search_patience", 64):
        response = requests.post(
            url=f"{url}/sdapi/v1/txt2img", json=config["parameters"]
        )
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        batch_best = get_best_image(r, models, model_weights)
        if best is not None and batch_best["score"] <= best["score"]:
            p += len(r["images"])
            continue
        p = 0
        best = batch_best
        # save new best
        filename = f"{best['job_timestamp']}-{best['seed']}.png"
        best["image"].save(outdir / filename, pnginfo=best["metadata"])
        print("New best:", best["score"])
    return best


def img2img_stage(url, config, models, model_weights, outdir, best):
    p = 0
    params = config["parameters"].copy()
    denoising_strength = config.get("initial_denoising_strength", 0.75)
    # TODO: make these configurable
    denoise_step = 0.1
    denoise_min = 0.1
    print(
        f"Starting img2img random search with denoising strength {denoising_strength}"
    )
    while denoising_strength >= denoise_min:
        if p >= config.get("img2img_patience", 16):
            denoising_strength -= denoise_step
            p = 0
            print(f"Reduced denoising strength to {denoising_strength}")
            continue

        params["init_images"] = [best["image_b64"]]
        params["denoising_strength"] = format_decimal(denoising_strength)
        response = requests.post(url=f"{url}/sdapi/v1/img2img", json=params)
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        batch_best = get_best_image(r, models, model_weights)
        if batch_best["score"] <= best["score"]:
            p += len(r["images"])
            continue
        p = 0
        best = batch_best
        # save new best
        filename = f"{best['job_timestamp']}-{best['seed']}.png"
        best["image"].save(outdir / filename, pnginfo=best["metadata"])
        print("New best:", best["score"])
    return best


def inpainting_stage(url, config, models, model_weights, outdir, best):
    patch_ratio = 5
    patch_size = math.ceil(
        min(best["image"].size[0], best["image"].size[1]) / patch_ratio
    )
    patches = generate_patches(best["image"], patch_size)
    # TODO: smarter patch selection
    params = config["parameters"].copy()
    params["mask_blur"] = patch_size // 10
    params["inpainting_fill"] = 1  # original
    params["inpaint_full_res"] = False  # whole image, not just masked region
    denoising_strength = config.get("initial_inpaint_denoising_strength", 0.5)
    denoise_min = 0.1
    denoise_step = 0.1
    p = 0
    print(f"Inpainting with denoising strength {denoising_strength}")
    print(f"Number of tiles: {len(patches)}")
    while denoising_strength >= denoise_min:
        if p >= len(patches) / 2:  # TODO: threshold is arbitrary
            denoising_strength -= denoise_step
            print(f"Reduced denoising strength to {denoising_strength}")
            p = 0
            continue

        params["init_images"] = [best["image_b64"]]
        params["denoising_strength"] = format_decimal(denoising_strength)

        num_patches = 1
        patch_set = random.choices(patches, k=num_patches)
        mask = create_mask(best["image"], patch_set)
        params["mask"] = encode_pil_to_base64(mask)

        response = requests.post(url=f"{url}/sdapi/v1/img2img", json=params)
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        batch_best = get_best_image(r, models, model_weights)
        if batch_best["score"] <= best["score"]:
            p += len(r["images"])
            continue
        p = 0
        best = batch_best
        # save new best
        filename = f"{best['job_timestamp']}-{best['seed']}.png"
        best["image"].save(outdir / filename, pnginfo=best["metadata"])
        print("New best:", best["score"])
    return best


def main(
    url, config, outdir, init_image=None, skip_img2img=False, skip_inpaint=False
):
    # TODO: make configurable
    model_paths = [
        "aesthetic_predictor/models/e621-l14-rhoLoss.ckpt",
        "aesthetic_predictor/models/e621a-l14-rhoLoss.ckpt",
        "aesthetic_predictor/models/starboard_cursed-l14-rhoLoss.ckpt",
    ]
    model_weights = [0.5, 0.5, 1]
    models = [AestheticPredictor(model_path=path) for path in model_paths]

    # create output directory if it doesn't exist
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    requests.post(url=f"{url}/sdapi/v1/options", json=config["options"])

    best = None
    if init_image is not None:
        embeds = models[0].get_embeds([init_image])
        scores = [sigmoid(np.array(model.predict(embeds=embeds))) for model in models]
        scores = [np.prod(np.array(s) ** model_weights) for s in zip(*scores)]
        best = {
            "image": init_image,
            "image_b64": encode_pil_to_base64(init_image),
            "score": scores[0],
        }
        print("Init image score:", best["score"])

    # txt2img seed search
    if init_image is None:
        best = txt2img_stage(url, config, models, model_weights, outdir)
    # img2img refinement
    if not skip_img2img:
        best = img2img_stage(url, config, models, model_weights, outdir, best)
    # inpainting refinement
    best = inpainting_stage(url, config, models, model_weights, outdir, best)
    return best


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively improve Stable Diffusion outputs using aesthetic scores via"
            " AUTOMATIC1111's webui API."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        help="json file containing generation parameters",
        default="config.json",
    )
    parser.add_argument(
        "--outdir", type=str, help="directory to save images", default="outputs"
    )
    parser.add_argument(
        "--url", type=str, help="URL for A1111 WebUI", default="http://localhost:7860"
    )
    parser.add_argument(
        "--init_image",
        type=str,
        help="use starting image instead of performing random seed search. skips txt2img",
    )
    parser.add_argument(
        "--skip_img2img",
        action=argparse.BooleanOptionalAction,
        help="skip the img2img phase",
    )
    parser.add_argument(
        "--skip_inpaint",
        action=argparse.BooleanOptionalAction,
        help="skip the inpainting phase",
    )
    args = parser.parse_args()

    with open(args.config, "rt") as f:
        config = json.load(f)

    init_image = None
    if args.init_image is not None:
        init_image = Image.open(Path(args.init_image).expanduser())

    main(args.url, config, args.outdir, init_image, args.skip_img2img, args.skip_inpaint)
