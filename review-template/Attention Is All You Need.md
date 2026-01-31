---
Reading-Stutus:
  - ✅ Done
Author:
  - Ashish Vaswan
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  -  Łukasz Kaiser
  - Illia Polosukhin
Journal:
Published Year:
Topic:
Review-Date:
Comment: 트랜스포머 제안
URL:
isTargetPaper:
linked-bases:
  - "[[Architecture.base]]"
Lab:
  - Google Brain
  - Google Research
---
Note : [[Transformer]]
# Abstract
---
- 기존의 번역 모델들은 convolution, recurrence를 복잡하게 사용거나 attention을 사용해서 encoder, decoder 연결해서 사용.
- convolution, recursive를 완전 배제한 attention만을 기반으로 하는 새로운 아키텍쳐 제안 → transformer
- 특징점: 병렬화와 우수한 성능, 훈련 시간이 적게 걸림
- 다른 작업에도 잘 적용되는 일반화가 잘 됨.


# I. Introduction
---
- 기존의 SOTA 모델에 사용된 RNN 기반 방식들은 시계열로 정보를 처리(hidden cell)하여, 병렬화 불가능.
- 또한, 계속 재귀적으로 backprop을 하다보니, long-range dependencies 문제가 있었음. 
- attention은 이제 이 분야의 중요 매커니즘으로 자리 잡음 → LSTM 같은 RNN 기반의 고질적 문제인 input과 ouput 간 거리가 길어지면 추론력이 약해지는 점을 보완.
- 'attention is all you need' -> RNN/CNN 구조의 탈피를 제안. 


# 2. Background
---
1. RNN approach
	- Sequential 한 정보 처리 즉, 이전 hidden state가 계산되어야 이후가 계산 가능해서 병렬처리가 불가능했음.
	- 또한, long-range dependencies, vanishing gradient에 취약한 모습을 보였음.
2. CNN approach 
	- Convolution filter를 통해 local 한 feature를 뽑으려고 시도했으나, long-range에 대한 정보를 뽑기 위해서는 여러 번의 convolution이 필요했으니, 연산량이 증가했음.
	- 대표적으로 ConvS2S, ByteNet 등이 있음.
3. Self-Attention(from this paper)
	- context 내 attention이 한번 즉, $O(1)$ 에 계산되어 long-range dependencies가 확보.
	- 풍부한 representation을 뽑기 위해 multi-head로 사용함. 
	- 본 논문에서는 single-head로 사용하게 되면, 모델은 q에 대해 모든 단어들의 정보를 하나의 vector로 평균내어(v matrix)로 사용하게 되는데, 정보의 해상력이 너무 낮음.
		- CNN으로 치면 filter가 많을수록 다양한 feature를 뽑을 수 있는 거니까, 다다익선.(head = filter)
# 3. Model Architecture
---
> [!multi-column]
>> [!check]
>> ![[The Transformer - model Architecture.png | 250]]
>> Fig1. Transformer Model Architecture
>
>> [!check] 
>> - 대부분 성능 좋은 모델들은 encoder-decoder 구조를 차용하고 있다.
>> 
>> - encoder에서는 symbolic representation인 $(x_1, ..., x_n) → (z_1, ..., z_n)$로 맵핑
>> 
>> - decoder는 z가 주어지면, 한 번에 한 토큰씩 $(y_1, ..., y_m)$ 를 순차적으로 생성. 이 때, 한 번 출력된 token은 input으로 사용되어 auto-regressive 구조.
> 


## 3.1 **Encoder and Decoder Stacks**
---
### Encoder:
---
- _N_ = 6 즉, 6개의 독립적인 레이어로 구성.
- $d_{model} = 512$ : 모든 layer와 embedding 차원을 통일.
- 각 레이어는 하위 . 이어 2개로 구성.
	- Multi-head Attention(Self-Attention)
	- Feed Forward Network
- 이 하위 레이어 연결 간에는 residual connection을 사용 즉, $\text{ouput} = \text{LayerNorm}(x + \text{sublayer}(x))$
	- residual-connection(short-cut): ResNet에서 제안된 방법으로 layer간 정보 전달 시 layer를 통과한 결과가 통과전 결과를 sum해서 전파. 
	- →vanishing gradient 해결법 중 하나. 

### Decoder:
---
- 마찬가지로, _N_ = 6 즉, 6개의 독립적인 레이어로 구성.
- $d_{model} = 512$ : 모든 layer와 embedding 차원을 통일.
- encoder의 출력에 대해 cross-attention을 수행할 한 개의 sublayer 추가해서 각 layer는 총 3개의 sublayer로 구성.
	- Masked Multi-heade Attention
	- Mutlihead-Attention(Cross)
	- Feed Forward Network
- encoder와 마찬가지로, sublayer간 연결은 residual connection
- 또, decoder의 역할은 token 생성이기에, 이는 앞의 context에만 의존적이여야 하기 때문에 masked attention을 사용 즉, i+1번 위치의 token을 추론하기 위해서는 i 번째 까지만 참고하게 설계

## 3.2 [[Attention]]
---
- Visualization ex : [detail-Youtube Vidoe](https://www.youtube.com/watch?v=DdpOpLNKRJs)
  
> [!question] 
>  왜 Q, K, V가 변환되는 과정에서 다른 Dense Layer가 사용되어야 하는가?
>  -> 같은 걸 사용해도 되는데, representation power 확보

> [!summary] Summary
> 1. **Q, K, V 계산:** 입력 임베딩에 대해 세 개의 독립적인 Dense Layer를 사용해 각각 q, k, v 벡터 생성.
> 2. **Attention Score Matrix:** 각 토큰의 Query와 다른 토큰의 Key 간 내적을 통해 계산된 값(스케일링 및 softmax 적용)은, 해당 토큰(q)이 다른 토큰(k)에 얼마나 집중(주의)할지를 나타내는 가중치로 사용됨.
> 3. **독립적 Dense Layer의 필요성:** 각 벡터가 서로 다른 역할(유사도 계산과 정보 전달)을 수행하므로, 별도의 변환을 통해 모델이 더 풍부하고 유연한 표현을 학습할 수 있도록 합니다.
>

> [!multi-column]
>> [!check]
>> ![[scaled-dot-attention-MHA.png | 250]]
>> Scaled Dot-product Attention Multi-Head Attention
>
>> [!check] 
>> ![[MHA.png]]
>> Muli-Head-Attention
> 

### 3.2.1 **Scaled Dot-Product Attention**
---
> [!summary] 
> $$Attention(Q, K, V ) = softmax({QK^T\over √d_k})V $$
> smoothing : 차원 수 $d_k$가 커지면 내적 값인 $QK^T$도 커지는 경향성이 있음. 이대로 softmax 통과하면, logit-prob distribution이 sharp해져서 backprop시 vanishing gradient 부담이 커짐.

> [!warning] 
>  $d_k$는 query, key vector의 dimension이고, $d_v$는 value vector의 dimension 둘이 통일 되지 않아도 되는 것에 주의!

### 3.2.2 Multi-Head Attention
---
> [!multi-column]
>> [!check]
>> ![[MHA.png]]
>> Muli-Head-Attention
>
>> [!check] 
>> - self-attention 구조를 병렬적(h개)으로 여러 개 사용하여, (Q, K, V)를 여러 쌍으로 만들어 사용하는 것이 더 좋은 성능을 냈다.
>> - 각 self-attention 계층의 output은 v vector들이 weighted sum이 된 형태이므로, $d_v$ 차원이 된다.
>> - 이 (token_length x $d_v$) matrix h 개를 concat 시키고 Linear-projection으로 압축.(하나의 head
>

$$\text{MultiHead}(Q, K, V ) = \text{Concat}(\text{head}_{1}, ..., \text{head}_{\text{h}})W^O$$

where
- $\text{head}_{\text{i}} = \text{Attention}(QW^Q_i, KW^K_i, V W^V_i)$
- $W^Q_i ∈ \mathbb{R}^{d_{model}×d_k}, W^K_i∈\mathbb{R}^{d_{model}×d_k}$
	- alignment matrix가 정방행렬 : 두 projection layer shape은 동일해야 함. 
- $W^V_i ∈ \mathbb{R}^{d_{model}×d_v}$
	- value vector의 embedding은 크기다 q, k랑 달라도 되긴 하는데, multi-head 부분을 하나로 합쳐주는 $W^O$ layer shape도 같이 변함.
- $W^O ∈ \mathbb{R}^{hd_v×d_{model}}$

- 본 논문에서는 $h=8$
- $d_v=d_k=d_{model}/h=64$로 설정.
	- 연산량을 비슷하게 맞춰서 single-head vs multi-head를 비교하려고 그렇게 구성.
- 같은 computing cost라면, single-head보다 multi-head가 더 나은 performance를 보임.

### **3.2.3** **Applications of Attention in our Model**
---
모델 구조를 좀 더 펴서 보면 아래와 같다.
https://zeuskwon-ds.tistory.com/88
![[transformer-overview.png]]

Attention은 모델 내에서 총 세 군데 사용되었는데, 

### Encoder(Self-Attention) : NLU
---
- $Q, K, V$ 모두 이전 encoder-block에서 가져옴.
- transformer는 원래 machine-translation을 위해 개발된 툴로, encoder 단에서는 번역 대상 문장에 대한 natural language understanding이 기능을 수행하도록 만들어짐.
- 따라서 전체 입력에 대해 un-masked attention을 수행해서 contextualization을 함.
- 최종적으로 encoder의 출력은 번역 타겟 문장의 context-vector.

### Decoder(Masked Self-Attention) : casual-inference
---
- $Q, K, V$ 모두 decoder-block의 이전 출력 layer에서 가져옴.
-  decoder 단에서는 encoder를 통해 생성된 context-vector를 사용해서, 번역 문장을 생성할 수 있도록 만들어짐. 
- 생성에 초점을 두어, self-attention에 일부 mask를 하여, attention-score 계산 시 현재 q에 대해서 더 앞의 token들만 참조할 수 있게 함. 
- masking의 경우, alignment matrix 단에서 $-\infty$를 더해서 사용.

### Encoder-Decoder Attention(Cross Attention) : 
---
- $K, V$ 는 encoder의 최종 출력, $Q$는 decoder에서 가져옴.
- 정확하게는 $Q, K, V$ 자체를 이 layer에서 만드는 거니까, 입력으로 encoder, decoder의 latent를 받고, attention-block 내부에서 projection.
- decoder의 기능은 번역본 생성이니, 현재까지의 decoder $Q$에 대해서, encoder 단에서 학습해둔 $K, V$를 참고하여 generate.
	- decoder에서 온 $q$는 현재 번역되고 있는 context로부터 만들어진거고,
	- encoder 단에서 학습된 $K, V$는 번역 타겟 문장에서 앞으로 참고해야할, 할 수 있는 token들에 대한 정보가 있어서 여기서 연결됨.

## 3.3 Position-wise Feed-Forward Networks
---
> [!summary] 
> $$FFN(x) = \text{ReLU}(xW_1+b_1)W_2+b_2$$
> 
> 2개의 linear layer w/ ReLU activation
> dimension은 512 -> 2048 -> 512로 구성됨. (inverse-bottleneck)
> dimension을 channel로 생각하면 1x1 convolution으로도 해석가능하다고 한다.
> 
> 이걸 CNN 계열 발전과정과 엮어서 생각해본다면, Depth-wise Separable Convolution 관점에서,
> - Attention 부분은 마치 CNN이 nxn(3x3) filter로 loac_feature를 뽑고
> - FFN은 CNN의 bottleneck구조처럼 channel 수를 조절하는 테크닉처럼 해석할 수 있음.
> -> 두 과정을 분리했다.
> 
> position-wise : token단위로 연산하겠다. (token끼리 영향을 주는 연산이 아님.)
> mlp랑 구조적으로는 동일하나, flatten을 하지 않는다는 것. 
> 즉, batch n짜리이고 feature가 $d_{model}$개인 데이터 처리와 동일.

## 3.4 Embedding and Softmax
---
최종 Decoder의 출력은 $n \times d_{model}$($n: \text{sequence length}$)
Decoder 뒤 붙어있는 linear layer는 다시 VOCA로 반환하는 layer 즉, $W_{out} \in \mathbb{R}^{d_{model} \times \text{Vocab size}}$
이후 Softmax는 VOCA axis 방향으로 적용하여 probability distribution 형태로 만듦.

Train time에서는 이미 정답 레이블이 다 있기에, token level 별 ground-truth와 loss 계산 후 backprop.
loss를 계산하기 위해 Transformer output shape : $n \times d_{model}$

Inference Time에서도 동일하게 $n \times d_{model}$으로 결과는 얻을 수 있으나, 입력된 마지막 token에 해당하는 prediction인 prob-distribution만 사용해서 NTP.

> [!important] Weight Sharing
> - Input Embedding : Encoder input을 vector로 변환 | $\mathbb{R}^{\text{VOCA size} \times d_{model}}$
> - Output Embedding : Decoder input을 vector로 변환 | $\mathbb{R}^{\text{VOCA size} \times d_{model}}$
> - Pre-softmax Linear Transformation: $\mathbb{R}^{d_{model} \times \text{VOCA size}}$
> 
> 이 세 layer들은 가중치를 공유하는데, shape이 다른 하나는 transpose해서 사용한다고 한다. 
> Embedding space는 word의 semantics를 담고 있어서, 입력 단에서와 출력 단에서의 의미를 일치시켜서 빠르고 일관성 있는 학습을 촉진함. 

## 3.5 [[Positional Encoding]]
---
- Transformer 같은 경우, 입력이 parallel하게 한 번에 밀어넣으니, 입력 token간 순서 정보가 없음. 따라서 이걸 명시적으로 주기 위해 token embedding에 더해서 순서 정보를 삽입.

# 4 Why Self-Attention
---
![[transformer-table1.png]]

> [!summary] 
> 1. Recurrent Net보다 계산 복잡도 이득 : $O(n^2d)$ vs $O(nd^2)$
> 	- $n$ : seq-length, $d$: embedding dimension
> 2. parallelism
> 3. long-range dependencies
> 4. interpretability : 어디에 focus를 주어 해석했는지.

# 5 Training
---
## 5.1 Training Data and Batching
---
> [!summary] 
> 4.5M 규모의 sentence pair의 WMT 2014 English-German set.
> [[Byte-Pair Encoding(BPE)]] 사용했고, 약 37k VOCA size.
> 
> WMT 2014 English-French는 36M 규모의 sentence pair.
> Word-piece, 32k VOCA

## 5.2 Hardware and Schedules
---
> [!summary] 
> P100 8장 
> base model : 약 12시간
> Big Model: 약 3.5일 정도 소요.
> 

## 5.3 Optimizer
---
> [!summary] 
> Adam w/ $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$
> 고정 lr이 아닌, 가변 lr 사용했고, 4000 step까지 warmup.(이후 decay)
> 
>  $$\text{learning rate} = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_{num}}\cdot \text{warmup\_steps}^{-1.5})$$

## 5.4 Regularization
---
> [!summary] 
> 1. Residual Dropout
> 	- $\text{Output} = \text{LayerNorm}(x+\text{Dropout}(\text{Sublayer}(x)))$
> 		- 비율은 0.1
> 2. Label Smoothing
> 	- 특정 단어에 너무 over-confidence를 갖지 않도록 hard label 대신 soft label을 사용.
> 		- $\epsilon_{ls}=0.1$
> 	- one-hot을 살짝 변형에서 오답에도 확률을 나눠줌.
> 	- perplexity는 높아질 수 있지만(오답에도 확률을 주어서 entropy를 높이니까), 최종 정확도나 BLEU에는 좋은 영향을 주었다고 함. 

# 6 Results
---
## 6.1 Machine Translation
---
![[transformer-table2.png]]
- 경제적이고, SOTA 획득.

## 6.2 Model Variations
---
![[transformer-table3.png]]
- (A): single-head vs multi-head : multi-head가 좋은 performane를 보이지만, 너무 head를 늘리다간, embedding-dimension이 너무 줄어서 표현력 확보가 안될수도 있어, 균형 잘 잡아야 함.
- (B): key 차원은 줄이면 성능 하락.
- (C) : layer는 깊어질수록, embedding은 커질수록, FNN에서도 inverse bottleneck을 키울수록 성능 상승
- (D) : Dropout, Soft-labeling은 효과적.
- (E) : 논문에서 제안한 sine/consine 기반의 positional encoding은 성능과는 큰 관련이 없으나, extrapolation에 강해서 채택.

## 6.3 English Constituency Parsing
---
![[transformer-table4.png]]
- Constituency Parsing : 구문 분석
	- 문장 구성 단어들이 어떤 문법적인 관계를 맺고 있는지 계층적인 tree 구조로 분석하는 방법.
- 4layer, $d_{model} = 1024$, 기존 번역 모델에서 Dropout, LR, Beam size만 살짝 조절함.
- 대부분의 모델 성능을 뛰어 넘음. 모델 구조를 크게 바꾸지 않았음에도.

# 7 Conclusion
---
> [!summary] 
> 1. Attention is All You Need.
> 2. fast learning - parallelism
> 3. SOTA
