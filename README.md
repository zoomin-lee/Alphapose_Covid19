## Stereo vision과 Alphapose를 활용한 실시간 객체 거리 탐지 연구

http://ktsde.kips.or.kr/digital-library/24795


### Opencv 함수
#### cv2.StereoSGBM_create의 파라미터
- minDisparity : 가능한 최소한의 disparity 값으로 보통 0으로 설정하지만 조정 알고리즘이 이미지를 이동시킬 수 있어서 이 파라미터는 알맞게 조정되어야 한다.
- numDisparities : 최대 disparity - 최소 disparity 값으로 항상 0보다 큰 값을 갖는다.
- blockSize : 매칭된 블록 사이즈로 1보다 큰 홀수여야한다. 보통 3~11의 값을 갖는다.
- P1 : diparity smoothness를 조절하는 첫 파라미터로 인접 픽셀의 disparity 차이를 +1 or -1로 바꿔주는 disparity 패널티이다.
- P2 : diparity smoothness를 조절하는 두 번째 파라미터이다. 값이 커지면 더 부드러워진다. P2는 인접 픽셀 disparity 사이에서 1보다 크게 변화하는 것에 대한 패널티이다. P2 > P1이어야 한다.
- disp12MAxDiff : 좌-우 disparity 체크에서 허용된 최대 차이로 비활성화 하려면 음수를 설정하면 된다.
- uniquenessRatio : 보통 5~15의 값을 갖는게 좋다. 1등 매치(cost function 비교)와 2등 매치간의 마진을 퍼센트로 나타낸 것이다.
- speckleWindowSize : 노이즈 반점(speckle) 및 무효화를 고려하는 smooth disparity 영역의 최대 크기이다. 보통 50-200의 범위를 가지며 0으로 설정하면 비활성화된다.
- speckleRange : 각 연결된 요소들 내에서 최대 disparity 변동이다. speckle 필터링을 한다면, 양수로 설정하고, 이는 16과 곱해질 것이기다. 보통 1,2면 좋다.
- preFilterCap : 필터링된 이미지 픽셀을 위한 Truncation(절단) 값이다.
- mode : cv2.STEREO_SGBM_MODE_HH, MODE_HH4, MODE_SGBM_3WAY 등의 모드가 있다.
