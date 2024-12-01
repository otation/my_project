import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
import pytesseract

# Windows 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath('plate_detect.py'))  # 현재 스크립트 위치
MODEL_DIR = os.path.join(CURRENT_DIR, "model")  # 모델 폴더
DATA_DIR = os.path.join(CURRENT_DIR, "data")   # 데이터 폴더

# 필요한 디렉토리 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# notebook_utils.py가 필요하다면
import requests

try:
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    utils_path = os.path.join(CURRENT_DIR, "notebook_utils.py")
    with open(utils_path, "w", encoding='utf-8') as f:
        f.write(r.text)
        
    import notebook_utils as utils
    print("notebook_utils.py 다운로드 및 임포트 성공")
    
except Exception as e:
    print(f"notebook_utils.py 다운로드 중 오류 발생: {e}")

# A directory where the model will be downloaded.
base_model_dir = Path("model")



# The name of the model from Open Model Zoo.
detection_model_name = "vehicle-detection-0200"
recognition_model_name = "vehicle-attributes-recognition-barrier-0039"
# Selected precision (FP32, FP16, FP16-INT8)
precision = "FP32"

base_model_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"



# Check if the model exists.
detection_model_url = f"{base_model_url}/{detection_model_name}/{precision}/{detection_model_name}.xml"
recognition_model_url = f"{base_model_url}/{recognition_model_name}/{precision}/{recognition_model_name}.xml"
detection_model_path = (base_model_dir / detection_model_name).with_suffix(".xml")
recognition_model_path = (base_model_dir / recognition_model_name).with_suffix(".xml")

# Download the detection model.
if not detection_model_path.exists():
    utils.download_file(detection_model_url, detection_model_name + ".xml", base_model_dir)
    utils.download_file(
        detection_model_url.replace(".xml", ".bin"),
        detection_model_name + ".bin",
        base_model_dir,
    )
# Download the recognition model.
if not os.path.exists(recognition_model_path):
    utils.download_file(recognition_model_url, recognition_model_name + ".xml", base_model_dir)
    utils.download_file(
        recognition_model_url.replace(".xml", ".bin"),
        recognition_model_name + ".bin",
        base_model_dir,
    )

device = utils.device_widget()

device

# Initialize OpenVINO Runtime runtime.
core = ov.Core()



def model_init(model_path: str): #-> Tuple:
    """
    Read the network and weights from file, load the
    model on the CPU and get input and output names of nodes

    :param: model: model architecture path *.xml
    :retuns:
            input_key: Input node network
            output_key: Output node network
            exec_net: Encoder model network
            net: Model network
    """

    # Read the network and corresponding weights from a file.
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    # Get input and output names of nodes.
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model
# 번호판 인식을 위한 모델 추가
plate_detection_model_name = "license-plate-recognition-barrier-0001"
plate_model_url = f"{base_model_url}/{plate_detection_model_name}/{precision}/{plate_detection_model_name}.xml"
plate_model_path = (base_model_dir / plate_detection_model_name).with_suffix(".xml")

# 번호판 인식 모델 다운로드
if not plate_model_path.exists():
    utils.download_file(plate_model_url, plate_detection_model_name + ".xml", base_model_dir)
    utils.download_file(
        plate_model_url.replace(".xml", ".bin"),
        plate_detection_model_name + ".bin",
        base_model_dir,
    )

# 번호판 인식 모델 초기화
input_key_plate, output_keys_plate, compiled_model_plate = model_init(plate_model_path)
height_plate, width_plate = list(input_key_plate.shape)[2:]

def detect_license_plate(image, box):
    """번호판 영역 검출 및 텍스트 인식"""
    x_min, y_min, x_max, y_max = box
    car_roi = image[y_min:y_max, x_min:x_max]
    
    # 번호판 영역 전처리
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    # OCR 수행
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=config, lang='kor+eng')
    return text.strip()

def enhanced_convert_result_to_image(compiled_model_re, bgr_image, resized_image, boxes, threshold=0.6):
    """기존 함수에 번호판 인식 기능 추가"""
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    car_position = crop_images(image_de, resized_image, boxes)

    for x_min, y_min, x_max, y_max in car_position:
        # 차량 속성 인식
        attr_color, attr_type = vehicle_recognition(compiled_model_re, (72, 72), 
                                                  image_de[y_min:y_max, x_min:x_max])
        
        # 번호판 인식
        plate_text = detect_license_plate(bgr_image, (x_min, y_min, x_max, y_max))
        
        plt.close()

        # 바운딩 박스 그리기
        rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), 
                                colors["red"], 2)

        # 차량 속성 텍스트
        rgb_image = cv2.putText(
            rgb_image,
            f"{attr_color} {attr_type}",
            (x_min, y_min - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors["green"],
            2,
            cv2.LINE_AA,
        )
        
        # 번호판 텍스트
        rgb_image = cv2.putText(
            rgb_image,
            f"Plate: {plate_text}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors["blue"],
            2,
            cv2.LINE_AA,
        )

    return rgb_image

# de -> detection
# re -> recognition
# Detection model initialization.
input_key_de, output_keys_de, compiled_model_de = model_init(detection_model_path)
# Recognition model initialization.
input_key_re, output_keys_re, compiled_model_re = model_init(recognition_model_path)

# Get input size - Detection.
height_de, width_de = list(input_key_de.shape)[2:]
# Get input size - Recognition.
height_re, width_re = list(input_key_re.shape)[2:]

def plt_show(raw_image):
    """
    Use matplot to show image inline
    raw_image: input image

    :param: raw_image:image array
    """
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(raw_image)
    plt.show()

# Load an image.
# url = "https://storage.openvinotoolkit.org/data/test_data/images/person-bicycle-car-detection.bmp"
# filename = "cars.jpg"
# directory = "data"
# image_file = utils.download_file(
#     url,
#     filename=filename,
#     directory=directory,
#     show_progress=False,
#     silent=True,
#     timeout=30,
# )

image_file = "data/cars.jpg"
assert Path(image_file).exists()

# Read the image.
image_de = cv2.imread("data/cars.jpg")
# Resize it to [3, 256, 256].
resized_image_de = cv2.resize(image_de, (width_de, height_de))
# Expand the batch channel to [1, 3, 256, 256].
input_image_de = np.expand_dims(resized_image_de.transpose(2, 0, 1), 0)
# Show the image.
plt_show(cv2.cvtColor(image_de, cv2.COLOR_BGR2RGB))


#디텍팅모델의 output의 3번째 벡터는 [id, label, N, x,y,x,y]인데 추출+ 0인값 소거 
# Run inference.
boxes = compiled_model_de([input_image_de])[output_keys_de]
# Delete the dim of 0, 1.
boxes = np.squeeze(boxes, (0, 1))
# Remove zero only boxes.
boxes = boxes[~np.all(boxes == 0, axis=1)]

def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    """
    Use bounding boxes from detection model to find the absolute car position

    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: car_position: car's absolute position
    """
    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = (
        bgr_image.shape[:2],
        resized_image.shape[:2],
    )
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Find the boxes ratio
    boxes = boxes[:, 2:]
    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[0]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image,
            # upper box  bar should be positioned a little bit lower to make it visible on image
            (x_min, y_min, x_max, y_max) = [
                (int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2 else int(corner_position * ratio_x * resized_x))
                for idx, corner_position in enumerate(box[1:])
            ]

            car_position.append([x_min, y_min, x_max, y_max])

    return car_position

# Find the position of a car.
car_position = crop_images(image_de, resized_image_de, boxes)

# Select a vehicle to recognize.
pos = car_position[0]
# Crop the image with [y_min:y_max, x_min:x_max].
test_car = image_de[pos[1] : pos[3], pos[0] : pos[2]]
# Resize the image to input_size.
resized_image_re = cv2.resize(test_car, (width_re, height_re))
input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)
plt_show(cv2.cvtColor(resized_image_re, cv2.COLOR_BGR2RGB))

def vehicle_recognition(compiled_model_re, input_size, raw_image):
    """
    Vehicle attributes recognition, input a single vehicle, return attributes
    :param: compiled_model_re: recognition net
    :param: input_size: recognition input size
    :param: raw_image: single vehicle image
    :returns: attr_color: predicted color
                       attr_type: predicted type
    """
    # An attribute of a vehicle.
    colors = ["White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"]
    types = ["Car", "Bus", "Truck", "Van"]

    # Resize the image to input size.
    resized_image_re = cv2.resize(raw_image, input_size)
    input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)

    # Run inference.
    # Predict result.
    predict_colors = compiled_model_re([input_image_re])[compiled_model_re.output(1)]
    # Delete the dim of 2, 3.
    predict_colors = np.squeeze(predict_colors, (2, 3))
    predict_types = compiled_model_re([input_image_re])[compiled_model_re.output(0)]
    predict_types = np.squeeze(predict_types, (2, 3))

    attr_color, attr_type = (
        colors[np.argmax(predict_colors)],
        types[np.argmax(predict_types)],
    )
    return attr_color, attr_type    

print(f"Attributes:{vehicle_recognition(compiled_model_re, (72, 72), test_car)}")

def convert_result_to_image(compiled_model_re, bgr_image, resized_image, boxes, threshold=0.6):
    """
    Use Detection model boxes to draw rectangles and plot the result

    :param: compiled_model_re: recognition net
    :param: input_key_re: recognition input key
    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: rgb_image: processed image
    """
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Find positions of cars.
    car_position = crop_images(image_de, resized_image, boxes)

    for x_min, y_min, x_max, y_max in car_position:
        # Run vehicle recognition inference.
        attr_color, attr_type = vehicle_recognition(compiled_model_re, (72, 72), image_de[y_min:y_max, x_min:x_max])

        # Close the window with a vehicle.
        plt.close()

        # Draw a bounding box based on position.
        # Parameters in the `rectangle` function are: image, start_point, end_point, color, thickness.
        rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["red"], 2)

        # Print the attributes of a vehicle.
        # Parameters in the `putText` function are: img, text, org, fontFace, fontScale, color, thickness, lineType.
        rgb_image = cv2.putText(
            rgb_image,
            f"{attr_color} {attr_type}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            colors["green"],
            10,
            cv2.LINE_AA,
        )

    return rgb_image

# plt_show(convert_result_to_image(compiled_model_re, image_de, resized_image_de, boxes))
plt_show(enhanced_convert_result_to_image(compiled_model_re, image_de, resized_image_de, boxes))
