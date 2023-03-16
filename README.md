# SimpleTLTrade
가장 초기버전으로, 속도도 느리다.
Q-Value를 softmax 하여 가장 값이 높은(reward가 높다고 생각되는) action을 수행하도록 했다.
또한 Q-value의 값에 따라 구매 혹은 판매하는 수량을 정했다.

## 개요
+ 데이터 크롤링
+ 데이터 전처리
+ 각종 인덱스 생성
+ VAE의 latent vector를 활용한 차원 축소
+ LSTM 으로 미래 흐름 예상
+ out_data.csv 생성
+ Q-learning 사용한 강화학습 진행
+ 리워드 확인
  