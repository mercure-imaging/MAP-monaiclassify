import json
from typing import Dict, Text
import os, shutil
import torch
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import monai.deploy.core as md
from monai.data import DataLoader, Dataset
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector

from pathlib import Path
from typing import Dict, Optional
from monai.deploy.core import AppContext, ConditionType, Fragment, Image, Operator, OperatorSpec

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
from monai.data.box_utils import convert_box_mode
from pydicom.uid import generate_uid


# @md.input("image", Image, IOType.IN_MEMORY)
# @md.output("result_text", Text, IOType.IN_MEMORY)
class ClassifierOperator(Operator):
    
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "classification_results"
    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path.cwd() / "model/model.ts"))

    def __init__(self,
                fragment: Fragment,
                *args,
                model_name: Optional[str] = "",
                app_context: AppContext,
                model_path: Path = MODEL_LOCAL_PATH,
                output_folder: Path = DEFAULT_OUTPUT_FOLDER,
                **kwargs,
                 ):
        
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"
        self.input_name_image = "image"
        self.output_name_result = "result_text"
        # The name of the optional input port for passing data to override the output folder path.
        self.input_name_output_folder = "output_folder"
        # The output folder set on the object can be overriden at each compute by data in the optional named input
        self.output_folder = output_folder
        # Need the name when there are multiple models loaded
        # self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        # Need the path to load the models when they are not loaded in the execution context
        self.model_path = app_context.model_path
        # Use AppContect object for getting the loaded models
        self.app_context = app_context
        # self.model = self._get_model(self.app_context, self.model_path, self._model_name)
        super().__init__(fragment, *args, **kwargs)


    def setup(self, spec: OperatorSpec):
        """Set up the operator named input and named output, both are in-memory objects."""

        spec.input(self.input_name_image)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional for overriding.
        spec.output(self.output_name_result).condition(ConditionType.NONE)  # Not forcing a downstream receiver.

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
        
        print("returning from here")

        return metadata

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        input_image = op_input.receive(self.input_name_image)
        if not input_image:
            raise ValueError("Input image is not found.")
        if not isinstance(input_image, Image):
            raise ValueError(f"Input is not the required type: {type(Image)!r}")

        _reader = InMemImageReader(input_image)
        input_img_metadata = self._convert_dicom_metadata_datatype(input_image.metadata())
        print("Back from printing metadata")
        img_name = str(input_img_metadata.get("SeriesInstanceUID", "Img_in_context"))


        # output_path = context.output.get().path

        print("checking device")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device being used:", device)

        returned_layers = [1,2]
        base_anchor_shapes = [[6,8,4],[8,6,5],[10,10,6]]
        anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes= base_anchor_shapes
    )
        
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process()
        
        model_path = self.model_path
        print(model_path)
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

        # Testing
        # pred_boxes = [[56.740604400634766, 84.39207458496094, -115.19475555419922, 23.67993927001953, 16.88201904296875, 18.818740844726562]]
        # series_uid = generate_uid()
        # # self.draw_bounding_boxes(pred_boxes, context.input.get().path, context.output.get().path, series_uid)
        # self.draw_bounding_boxes(pred_boxes, Path(self.app_context.input_path), Path(self.app_context.output_path), series_uid)

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
                
                nNodules, confident_boxes = 0, []
                for score, box in zip(result_dict_out["score"], result_dict_out["box"]):
                    if score >= 0.8:
                        nNodules += 1
                        confident_boxes.append(box)

                result = {}
                result["number_of_nodules"] = nNodules
                text = "Number of nodules: " + str(nNodules)
                print(text)
                print("Confident Boxes: ", confident_boxes)

                # Mercure email notification parameter
                requested = True if nNodules > 0 else False
                result["__mercure_notification"] = {
                    "text": text,
                    "requested": requested
                }

                print("Post-processing finished...")

        series_uid = generate_uid()
        self.draw_bounding_boxes(confident_boxes, Path(self.app_context.input_path), Path(self.app_context.output_path), series_uid)
                
        # output_folder = context.output.get().path
        output_folder = self.output_folder
        print(output_folder)

        output_path = output_folder / "result.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)

        output_path = output_folder / "output_raw.json"
        with open(output_path, "w") as fp:
            json.dump(result_dict_out, fp)

        op_output.emit("Number of nodules detected: " + str(nNodules), "result_text")

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

    def normalize_image_to_uint8(self, image):
        draw_img = image
        draw_img_min, draw_img_max = np.amin(draw_img), np.amax(draw_img)
        if np.amin(draw_img) < 0:
            draw_img -= np.amin(draw_img)
        if np.amax(draw_img) > 1:
            draw_img = draw_img / np.amax(draw_img)
        draw_img = (255 * draw_img).astype(np.uint8)
        return draw_img, draw_img_min, draw_img_max
    
    def denormalize_image(self, image, draw_img_min, draw_img_max):
        draw_img = image
        draw_img = draw_img/255
        if draw_img_max > 1:
            # Here, additionally multiplying with 1.25 to account for data loss.
            draw_img = draw_img * draw_img_max * 1.25
        if draw_img_min < 0:
            draw_img += draw_img_min
        draw_img = draw_img.astype(np.int16)
        return draw_img
    
    def draw_bounding_boxes(self, pred_boxes, input_path, output_path, series_uid):
        print("Drawing bounding boxes!!")
        print(input_path, output_path, series_uid)
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in sorted(os.walk(input_path)):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))

        lstFilesDCM.sort()
                    
        # Get ref file
        ds = pydicom.dcmread(lstFilesDCM[0])

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(ds.Rows), int(ds.Columns), 3, len(lstFilesDCM))

        # Getting required values from reference image tags.
        deltaX, deltaY, deltaZ = ds[0x0020,0x0032].value
        deltaXpx, deltaYpx = int(abs(deltaX)/0.703125), int(abs(deltaY)/0.703125)
        pixelSpacing = ds[0x0028,0x0030].value[0]
        sliceThickness = ds[0x0018,0x0050].value

        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)

        print("Delta x: ", deltaX)
        print("Delta y: ", deltaY)
        print("Delta z: ", deltaZ)
        print("deltaxpx: ", deltaXpx, "deltaypx: ", deltaYpx)
        print("pixel spacing: ", pixelSpacing)
        print("slice thickness: ", sliceThickness)

        print("DCM Files list: ")
        print(lstFilesDCM)

        # loop through all the DICOM files
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = pydicom.dcmread(filenameDCM)
            img = ds.pixel_array
            draw_img, draw_img_min, draw_img_max = self.normalize_image_to_uint8(img)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
            # store the raw image data
            ArrayDicom[:, :, :, lstFilesDCM.index(filenameDCM)] = draw_img

        # Processing the predicted bounding boxes.
        for pred_box in pred_boxes:
            bbox = np.round(pred_box).astype(int).tolist() 
            vec = convert_box_mode(
                    np.expand_dims(np.array(bbox), axis=0),
                    src_mode="cccwhd",
                    dst_mode="xyzxyz",
                )
            vec = vec.squeeze()
            xmin, ymin, zmin = vec[0], vec[1], vec[2]
            xmax, ymax, zmax = vec[3], vec[4], vec[5]

            # Looping over the volume and checking if that slice is falling in the bounding box depth range.
            for i in range(ArrayDicom.shape[-1]):
                slicePosMM = (-i * sliceThickness) + deltaZ
                print(slicePosMM, zmin, zmax)
                draw_img = ArrayDicom[:, :, :, i]
                draw_img = np.ascontiguousarray(draw_img, dtype=np.uint8)
                if slicePosMM <= zmax and slicePosMM >= zmin:
                    print("slice in the right range: {} with value: {}".format(i+1, slicePosMM))
                    cv2.rectangle(
                        draw_img,
                        pt1=(int((xmin)/pixelSpacing)+deltaXpx, int((ymin)/pixelSpacing)+deltaYpx),
                        pt2=(int((xmax)/pixelSpacing)+deltaXpx, int((ymax)/pixelSpacing)+deltaYpx),
                        color=(0, 0, 255),  # red for predicted box
                        thickness=1,
                    )
                    # plt.imshow(draw_img, cmap=plt.cm.gray)
                    # plt.show()
                    # plt.savefig(str(output_path) + '/slice' + str(i+1) + '.png')
                    
                # Save the modified slice back in the original array, so that another bounding box could be drawn on it- if that is the case.
                ArrayDicom[:, :, :, i] = draw_img

        # Looping over the array again, to save the final modified slices.
        for i in range(ArrayDicom.shape[-1]):
            dcm_file_in = lstFilesDCM[i]
            # Load the input slice
            ds = pydicom.dcmread(dcm_file_in)
            ds.PhotometricInterpretation = 'RGB'
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 3
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.add_new(0x00280006, 'US', 0)
            ds.is_little_endian = True
            ds.fix_meta_info()
            # Set the new series UID
            ds.SeriesInstanceUID = series_uid
            # Set a UID for this slice (every slice needs to have a unique instance UID)
            ds.SOPInstanceUID = generate_uid()
            # Compose the filename of the modified DICOM using the new series UID
            out_filename = series_uid + "#" + str(i+1)
            dcm_file_out = str(output_path) + "/" + str(out_filename) + ".dcm"
            # Store the updated pixel data of the input image.
            draw_img = np.ascontiguousarray(ArrayDicom[:, :, :, i], dtype=np.uint8)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
            ds.PixelData = draw_img.tobytes()
            ds.Modality = "OT"
            del ds.WindowCenter
            del ds.WindowWidth
            del ds.RescaleIntercept
            del ds.RescaleSlope
            
            # Write the modified DICOM file to the output folder
            ds.save_as(dcm_file_out)