# sd-aesthetic-optimizer
Iteratively improve Stable Diffusion outputs using aesthetic scores

**This script uses the A1111 webui API, so ensure you have `--api` in the launch flags.**

## Setup
```bash
# clone with submodules
git clone --recurse-submodules https://github.com/feffy380/sd-aesthetic-optimizer
# install dependencies (you may want to do this in a venv)
pip install -r requirements.txt
```

## Usage
Copy `config.json.template` and fill in the fields:
- `seed_search_patience`: number of txt2img seeds to sample. Best result will be used as the starting point for img2img.
- `img2img_patience`: after generating this many images with img2img with no score improvement, reduce denoising strength.
- `initial_denoising_strength`: img2img start with this denoising strength.
- `options`: passed to `/sdapi/v1/options`. See webui's `/docs` page.
- `parameters`: passed to `/sdapi/v1/txt2img` and `/sdapi/v1/img2img`. See webui's `/docs` page.
```bash
# see --help for available options

# default values. same as running with no arguments
# search seeds with txt2img and iterate on the best result with img2img. save current best to "outputs" folder
python aesthetic_optimizer.py --config="config.json" --outdir="outputs"

# skip txt2img phase by providing a starting image
python aesthetic_optimizer.py --init_image="path/to/init_image.png"
```
