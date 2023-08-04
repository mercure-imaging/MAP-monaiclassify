import json
from typing import Dict, Text

import torch
import numpy as np
import monai.deploy.core as md
from monai.data import DataLoader, Dataset
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    Orientation,
    Spacing,
    LoadImage,
    ScaleIntensityRange,
    DeleteItemsd
)

from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ClipBoxToImaged,
    ConvertBoxModed,
)

from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape



@md.input("image", Image, IOType.IN_MEMORY)
@md.output("result_text", Text, IOType.IN_MEMORY)
class ClassifierOperator(Operator):
    def __init__(self):
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

    def _convert_dicom_metadata_datatype(self, metadata: Dict):
        if not metadata:
            return metadata

        # Try to convert data type for the well knowned attributes. Add more as needed.
        if metadata.get("SeriesInstanceUID", None):
            try:
                metadata["SeriesInstanceUID"] = str(metadata["SeriesInstanceUID"])
            except Exception:
                pass
        if metadata.get("row_pixel_spacing", None):
            try:
                metadata["row_pixel_spacing"] = float(metadata["row_pixel_spacing"])
            except Exception:
                pass
        if metadata.get("col_pixel_spacing", None):
            try:
                metadata["col_pixel_spacing"] = float(metadata["col_pixel_spacing"])
            except Exception:
                pass

        print("Converted Image object metadata:")
        for k, v in metadata.items():
            print(f"{k}: {v}, type {type(v)}")

        return metadata

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        input_image = op_input.get("image")
        _reader = InMemImageReader(input_image)
        input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))

        output_path = context.output.get().path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        returned_layers = [1,2]
        base_anchor_shapes = [[6,8,4],[8,6,5],[10,10,6]]
        anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes= base_anchor_shapes
    )
        
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process()
        
        model_path = context.models.get().path
        net = torch.jit.load(model_path).to(device)
        detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False)
        print("Detector Loaded...")

         # set inference components
        patch_size= [192,192,80]
        score_thresh = 0.02
        nms_thresh = 0.22
        detector.set_box_selector_parameters(
            score_thresh=score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=nms_thresh,
            detections_per_img=100
        )
        detector.set_sliding_window_inferer(
            roi_size=patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="gaussian",
            device="cpu"
        )
        detector.eval()

        dataset = Dataset(data=[{self._input_dataset_key: img_name}], transform=pre_transforms)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

        with torch.no_grad():
            for d in dataloader:
                image = d[0].to(device)
                use_inferer = not all(
                [inference_data_i[0, ...].numel() < np.prod(patch_size) for inference_data_i in d]
                )
                
                print("Preprocessing finished...")
                outputs = detector(image, use_inferer=use_inferer)
                print("Predictions made...")

                inference_data = {}
                inference_data["image"] = d[0][0]
                inference_data["pred_box"] = outputs[0][detector.target_box_key].to(torch.float32)
                inference_data["pred_label"] = outputs[0][detector.target_label_key]
                inference_data["pred_score"] = outputs[0][detector.pred_score_key].to(torch.float32)
                
                inference_data = post_transforms(inference_data)
                print(inference_data)

                result_dict_out = {
                    "label": inference_data["pred_label"].cpu().detach().numpy().tolist(),
                    "box": inference_data["pred_box"].cpu().detach().numpy().tolist(),
                    "score": inference_data["pred_score"].cpu().detach().numpy().tolist(),
                }
                
                nNodules = 0
                for score in result_dict_out["score"]:
                    if score >= 0.5:
                        nNodules += 1

                result = {}
                result["number_of_nodules"] = nNodules
                text = "Number of nodules: " + str(nNodules)
                print(text)

                # Mercure email notification parameter
                requested = True if nNodules > 0 else False
                result["__mercure_notification"] = {
                    "text": text,
                    "requested": requested
                }

                print("Post-processing finished...")
                
        output_folder = context.output.get().path
        # output_folder.mkdir(parents=True, exist_ok=True)

        output_path = output_folder / "result.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)

        output_path = output_folder / "output_raw.json"
        with open(output_path, "w") as fp:
            json.dump(result_dict_out, fp)

        op_output.set("Number of nodules detected: " + str(nNodules), "result_text")

    def pre_process(self, image_reader) -> Compose:
        intensity_transform = ScaleIntensityRange(
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
        )
        return Compose(
            [
                LoadImage(reader=image_reader, image_only=True, affine_lps_to_ras=True),
                EnsureChannelFirst(),
                EnsureType(dtype=torch.float32),
                Orientation(axcodes="RAS"),
                intensity_transform,
                Spacing(pixdim=[
                        0.703125,
                        0.703125,
                        1.25
                    ], padding_mode="border"),
                EnsureChannelFirst(),
            ]
        )

    def post_process(self) -> Compose:
        return Compose(
        [
            ClipBoxToImaged(
                box_keys="pred_box",
                label_keys=["pred_label", "pred_score"],
                box_ref_image_keys="image",
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys="pred_box",
                box_ref_image_keys="image",
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=True,
            ),
            ConvertBoxModed(box_keys="pred_box", src_mode="xyzxyz", dst_mode="cccwhd"),
            DeleteItemsd(keys="image"),
        ]
    )
