import numpy as np
from PIL import Image

# 1. 이미지 파일 불러오기
image_path = "path_to_image.jpg"  # 이미지 파일 경로를 입력하세요
image = Image.open(image_path)
image_array = np.array(image)  # 이미지를 Numpy 배열로 변환

# 2. 차원을 (Height, Width, Channel) -> (Batch, Height, Width, Channel)로 확장
expanded_image = np.expand_dims(image_array, axis=0)
print("Shape after expand_dims:", expanded_image.shape)

# 3. 차원의 순서를 (Batch, Height, Width, Channel) -> (Batch, Channel, Width, Height)로 변경
transposed_image = np.transpose(expanded_image, (0, 3, 2, 1))
print("Shape after transpose:", transposed_image.shape)
