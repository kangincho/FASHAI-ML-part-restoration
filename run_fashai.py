import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 여기서부터 기존 코드 실행
import cv2
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer

# 1. 모델 초기화
# ctx_id: 0은 GPU, -1은 CPU
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('checkpoints/inswapper_128.onnx', download=False)

restorer = GFPGANer(
    model_path='checkpoints/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    device='cpu' # GPU 사용 시 'cuda', CPU 사용 시 'cpu'
)

# 2. 이미지 로드
# source: 사용자 얼굴 / target: 얼굴을 바꿀 타깃 이미지 
source_img = cv2.imread('source_img/movie_source.jpg')
target_img = cv2.imread('target_img/movie_target.jpg')

# 3. 얼굴 검출 및 특징/임베딩 추출 (Buffalo_l)
source_faces = face_app.get(source_img)
target_faces = face_app.get(target_img)

if not source_faces or not target_faces:
    raise ValueError("얼굴을 검출하지 못했습니다. 이미지를 확인해주세요.")

# 가장 크게 잡힌 얼굴 1개 선택 (Bounding box 기준 정렬)
source_face = sorted(source_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]

# 4. 얼굴 교체 (Inswapper128)
# target_img의 target_face 위치에 source_face의 얼굴을 합성
swapped_img = swapper.get(target_img, target_face, source_face, paste_back=True)

# 5. 얼굴 화질 복원 및 업스케일 (GFPGAN)
# 128x128 스왑으로 인해 뭉개진 얼굴 디테일 개선
_, _, restored_img = restorer.enhance(
    swapped_img,
    has_aligned=False,
    only_center_face=False,
    paste_back=True
)

# 6. 결과 저장
cv2.imwrite('output_result.jpg', restored_img)
print("추론 완료: output_result.jpg 저장 완료")

