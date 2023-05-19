import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image, PngImagePlugin
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


def main(url, config, outdir):
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
    patience = 10
    p = 0

    print("Starting txt2img seed search")
    while p < patience:
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
            p += 1
            continue
        p = 0
        best = batch_best
        # save new best
        filename = f"{best['job_timestamp']}-{best['seed']}.png"
        best["image"].save(outdir / filename, pnginfo=best["metadata"])
        print("New best:", best["score"])

    p = 0
    i1 = 0
    i2 = 0
    params = config["parameters"].copy()
    step_size = 0.75
    step_inc = 0.1
    step_inc_huge = step_inc * 4
    huge_step_interval = 8
    step_min = 0.0
    print(f"Starting img2img random search with denoising strength {step_size}")
    while p < patience:
        params["init_images"] = [best["image_b64"]]

        # small step
        params["denoising_strength"] = step_size
        response = requests.post(url=f"{url}/sdapi/v1/img2img", json=params)
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        small_step = get_best_image(r, models, model_weights)

        # large step
        big_step_size = step_size + step_inc
        if i1 % huge_step_interval == 0:
            big_step_size = step_size + step_inc_huge
        big_step_size = min(1.0, big_step_size)
        params["denoising_strength"] = big_step_size
        response = requests.post(url=f"{url}/sdapi/v1/img2img", json=params)
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        large_step = get_best_image(r, models, model_weights)

        # check for improvement
        new_best = True
        if max(small_step["score"], large_step["score"]) <= best["score"]:
            i2 += 1
            new_best = False
        elif small_step["score"] > large_step["score"]:
            best = small_step
            print("New best:", best["score"])
            i2 = 0
        else:
            best = large_step
            print("New best:", best["score"])
            if big_step_size > step_size:
                print(f"Increased denoising strength to {big_step_size}")
            step_size = big_step_size
            i2 = 0
        # save new best
        if new_best:
            filename = f"{best['job_timestamp']}-{best['seed']}.png"
            best["image"].save(outdir / filename, pnginfo=best["metadata"])
        # do we need to reduce step size?
        if i2 >= patience:
            step_size -= step_inc
            print(f"Reduced denoising strength to {step_size}")
            i2 = 0
        # stop when step_size gets too low
        if step_size <= step_min:
            break
        i1 += 1


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
    args = parser.parse_args()

    with open(args.config, "rt") as f:
        config = json.load(f)

    main(args.url, config, args.outdir)
