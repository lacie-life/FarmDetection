import numpy as np
from agoro_field_boundary_detector import FieldBoundaryDetector

# Load in the model, will start GEE session
model = FieldBoundaryDetector(model_path="/home/lacie-life/Github/SmartAgriculture/field-boundaries-detection/collection/agoro-field-boundary-detector/models/mask_rcnn")

# Make the prediction
im = np.asarray("/home/lacie-life/Github/SmartAgriculture/field-boundaries-detection/images/agriculture-800x500.jpg")
single_polygon = model(im)

