import os
import time


class ObjectDetectionModel:
    """
    Object detection wrapper.

    Note: importing `torch`/`torchvision` can be problematic in some environments.
    To keep the API up, we default to a lightweight stub unless `ENABLE_TORCH=1`.
    """

    # COCO class names
    CLASSES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self):
        # Defaults chosen to keep the API running even if Torch is unavailable.
        self.enable_torch = os.getenv("ENABLE_TORCH", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.model_version = None
        self.model = None
        self.transform = None
        self._torch = None

        if self.enable_torch:
            self._init_torch_model()

    def _init_torch_model(self) -> None:
        # Import inside the method to avoid crashing on API startup.
        import torch
        from torchvision import models, transforms

        pretrained = os.getenv("MODEL_PRETRAINED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        self._torch = torch
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        self.model.eval()
        self.model_version = "fasterrcnn_resnet50_fpn"

    def predict(self, image_path: str) -> dict:
        start_time = time.time()

        # Lite mode: return empty predictions but keep API stable.
        if not self.enable_torch or self.model is None:
            return {
                "predictions": [],
                "model_version": "stub",
                "processing_time": time.time() - start_time,
                "image_path": image_path,
            }

        # Torch mode.
        torch = self._torch
        assert torch is not None

        device = next(self.model.parameters()).device
        if device.type != "cpu":
            # Keep behavior consistent even if CUDA is available; move tensors to device.
            pass
        # Import PIL only in Torch mode; keep API startup safe.
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).to(device)

        with torch.no_grad():
            prediction = self.model([image_tensor])

        predictions = []
        for i in range(len(prediction[0]["boxes"])):
            confidence = float(prediction[0]["scores"][i].item())
            if confidence > 0.5:
                box = prediction[0]["boxes"][i].tolist()
                label_id = int(prediction[0]["labels"][i].item())
                label = self.CLASSES[label_id]
                predictions.append(
                    {"class": label, "confidence": confidence, "bbox": box}
                )

        return {
            "predictions": predictions,
            "model_version": self.model_version or "fasterrcnn_resnet50_fpn",
            "processing_time": time.time() - start_time,
        }