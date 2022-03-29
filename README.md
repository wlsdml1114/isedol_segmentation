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
