from fastapi import FastAPI, Body, Request

from modules import images
from modules.processing import StableDiffusionProcessing,  get_fixed_seed
from modules.paths import Paths
from modules.shared import opts
from modules.api.models import *
from modules.api import api
from modules.system_monitor import monitor_call_context
import gradio as gr

import rembg

# models = [
#     "None",
#     "u2net",
#     "u2netp",
#     "u2net_human_seg",
#     "u2net_cloth_seg",
#     "silueta",
# ]


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
        request: Request,
        task_id: str = Body(),
        input_image: str = Body("", title='rembg input image'),
        model: str = Body("u2net", title='rembg model'), 
        return_mask: bool = Body(False, title='return mask'), 
        alpha_matting: bool = Body(False, title='alpha matting'), 
        alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'), 
        alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'), 
        alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)

        with monitor_call_context(
            request,
            "modules.postprocessing.run_postprocessing",
            "modules.postprocessing.run_postprocessing",
            task_id,
            decoded_params={
                "width": input_image.width,
                "height":input_image.height,
            },
            is_intermediate=False,
        ):
            with monitor_call_context(
                request,
                "extensions.rembg",
                "extensions.rembg",
                decoded_params={
                    "width": input_image.width,
                    "height":input_image.height,
                },
            ):
                image = rembg.remove(
                    input_image,
                    session=rembg.new_session(model),
                    only_mask=return_mask,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                )


                p = StableDiffusionProcessing()
                p.set_request(request)

                parameters, existing_pnginfo = images.read_info_from_image(input_image)
                if parameters:
                    existing_pnginfo["parameters"] = parameters

                if opts.enable_pnginfo:
                    image.info = existing_pnginfo

                images.save_image(
                    image,
                    path=Paths(request).outdir_extras_samples(),
                    basename="",
                    seed=get_fixed_seed(-1),
                    extension=opts.samples_format,
                    short_filename=False,
                    no_prompt=True,
                    grid=False,
                    pnginfo_section_name="extras",
                    existing_info=existing_pnginfo,
                    p=p,
                    save_to_dirs=True,
                )

                return {"image": api.encode_pil_to_base64(image).decode("utf-8")}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
