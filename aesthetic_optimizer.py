import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image, PngImagePlugin

from aesthetic_predictor.aesthetic_predictor import AestheticPredictor


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


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
    metadata = None
    best_score = float("-inf")
    patience = 40
    p = 0

    while p < patience:
        response = requests.post(
            url=f"{url}/sdapi/v1/txt2img", json=config["parameters"]
        )
        if response.status_code // 100 != 2:
            print("Request failed:", response.status_code)
            p += 1
            continue
        r = response.json()
        info = json.loads(r["info"])
        infotexts = info["infotexts"]
        job_timestamp = info["job_timestamp"]

        images = [
            Image.open(io.BytesIO(base64.b64decode(image))) for image in r["images"]
        ]
        scores = [sigmoid(np.array(model.predict(images))) for model in models]
        scores = [np.prod(np.array(s)**model_weights) for s in zip(*scores)]
        assert len(scores) == len(images)
        batch_best = max(scores)
        if batch_best < best_score:
            p += len(images)
            continue
        p = 0
        i = scores.index(batch_best)
        best = images[i]
        best_score = scores[i]
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", infotexts[i])
        filename = f"{job_timestamp}-{info['all_seeds'][i]}.png"
        best.save(outdir / filename, pnginfo=metadata)
        print("New best:", best_score)


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
