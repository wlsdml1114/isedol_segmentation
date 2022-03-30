# isedol_segmentation
이세돌 영상 matting (delete background), 즉 누끼 따는 실험
- Image matting
  - 비챤님으로 여러가지 image matting 모델을 실험
  - MODNet(pretrained weight)을 이용해 Image matting 하기로 최종 결정

<img width="1478" alt="스크린샷 2022-03-17 오후 1 00 55" src="https://user-images.githubusercontent.com/6532977/160552854-7bbba1c3-1b94-47e1-aa1e-3c2fb1931160.png">

  - 비챤님, 아이네님, 징버거님은 캐릭터 가장자리가 둥글둥글해서 matting mask가 깔끔하게 떨어짐
  - 반면에 세구님은 후드티가 파스텔톤이라 배경이랑 구분이 잘 되지 않아 후드티가 자꾸 사라짐
  - 릴파님은 포니테일 꼬랑지 머리가 몸과 너무 분리되어있어서 잘 포착되지 않는 모습
  - 주르르님은 리본이 자꾸 배경으로 인식되는 문제 발생
<img width="1387" alt="스크린샷 2022-03-17 오후 12 58 10" src="https://user-images.githubusercontent.com/6532977/160553135-785a31a0-8a66-4c35-a2a1-e1c9f8170410.png">

  - Image matting의 한계를 느끼고 semantic segmentation으로 넘어감
  - Image matting으로 얻은 데이터를 전수점검해서 깔끔하게 잘 추출된 이미지만 segmentation에서 GT로 사용해서 transfer learning 진행
***
# Semantic segmentation
  - Image matting으로 mask를 모든 프레임에 대해 추출한 다음에 잘 따진 누끼만 학습용 데이터로 사용,
  - 이세돌 개인별로 모델생성(총 6개)
  - Data augmentation 진행 (flip, crop, resize 등)
  - 세구님 후드티랑 주르르님 리본은 드디어 인식을 어느정도하기 시작했는데, 아직도 배경이 혼란스러운경우에는 인식이 잘 되지 않는 경우가 많음
  - 릴파님 포니테일은 아무리해도 잘 안따짐, 아바타를 빌려서 직접 GT데이터를 더 확실하게 만든 다음에 학습을 진행해야하나 고민중
***
# Results

[이세계아이돌] 비챤 취기를 빌려 반응영상 (딥러닝으로 아바타 누끼작업)

[![챤기를 빌려 반응영상](http://img.youtube.com/vi/wcX7fQqaIe8/0.jpg)](https://youtu.be/wcX7fQqaIe8?t=0s)

[이세계아이돌] 2집 겨울봄 반응영상 (딥러닝으로 아바타 누끼작업)

[![겨울봄 반응영상](http://img.youtube.com/vi/KH5TlW0Njvg/0.jpg)](https://youtu.be/KH5TlW0Njvg?t=0s)

[이세계아이돌] 주르르 사이언티스트 반응영상 (딥러닝으로 아바타 누끼작업)

[![사이언티스트 반응영상](http://img.youtube.com/vi/o1p1PnMw7zc/0.jpg)](https://youtu.be/o1p1PnMw7zc?t=0s)

[이세계아이돌] 남세돌 리와인드 반응영상 (딥러닝으로 아바타 누끼작업)

[![남세돌 반응영상](http://img.youtube.com/vi/4s2UVhjfpDY/0.jpg)](https://youtu.be/4s2UVhjfpDY?t=0s)

***
# Future work
  -  Train dataset을 좀 더 다양하게 준비해서 모델 학습
  -  Edge detection(cv2.canny)를 이용해서 edge를 검출한뒤에 segmentation결과와 ensemble
    -  Edge를 이용하는 방법에 대한 고찰 필요
  -  다른 최신 SOTA 모델 및 back-bone 적용 (ex// HRNet)
  -  etc.(생각나면 추가)
