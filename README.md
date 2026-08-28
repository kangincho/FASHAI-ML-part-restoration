1) pre-trained 모델이 onnx로 배포되기 때문에 onnxruntime이 필요하다.

```
# CPU 환경인 경우
pip install onnxruntime

# GPU를 사용하는 경우
pip install onnxruntime-gpu
```

2) insightface의 모델은 직접 코드를 받을 필요는 없고 라이브러리와 모델 가중치 파일만 갖고 있다면 실행이 가능하다.

```
# insightface 라이브러리 다운로드
pip install insightface
```

3) 모델 가중치 파일 준비

- HuggingFace 저장소(ezioruan/inswapper_128.onnx) 등 inswapper_128.onnx라는 이름의 파일을 받아서 checkpoints/ 폴더에 넣어둘 것.
- GFPGAN Releases(https://github.com/TencentARC/GFPGAN/releases)에서 GFPGANv1.4.pth를 받아 checkpoints/ 폴더에 넣어둘 것.

4) 이미지 파일 준비

- source_img : 결과 이미지에 사용되기를 원하는 얼굴이 포함된 이미지
- target_img : 결과 이미지에 사용되기를 원하는 배경(또는 신체)이 포함된 이미지

5) 코드 실행
- 사용할 이미지 경로 수정 후 `run_fashai` 실행
