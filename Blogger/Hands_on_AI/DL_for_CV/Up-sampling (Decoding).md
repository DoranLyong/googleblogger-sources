# Up-sampling (Decoding)

![([source](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder))](./imgs2/Untitled.png)

([source](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder))

- `Up-sampling` 은 `Down-sampling`의 반대로 디코딩(decoding) 과정에서 신호를 복원하기 위해 data의 사이즈를 늘리는 처리 과정이다.

- 다음은 `Up-sampling`을 위한 대표적인 방법들이다.

  

### Unpooling

- `Maxpooling`으로 소실된 신호를 거꾸로 재현한다.
- 주변 픽셀을 동일한 값으로 채우거나 (`Nearest Neighbor Unpooling`), 0으로 채워주는 방식(`Bed of Nails Unpooling`)이 있다.

![([source](https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba))](./imgs2/Untitled%201.png)

([source](https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba))

- **(문제점)** 위의 $2 \times 2$ matrix 로 `Maxpooling` 된 data를 다시 원래 사이즈인 $4\times4$ matrix 로 `Unpooling` 할 때, Max pooled 된 값의 위치(position)을 알 수 없다. 즉, Unpooling 과정에서 본래 Max value를 어느 위치에 할당해야 할 지 알 수 없다.

### Max Unpooling

- Max value 의 위치를 알 수 없는 문제를 개선하기 위해 `Max unpooling` 방법을 사용한다. 방식은 직관적이다. `Max pooling` 과정에서 미리 선택된 값들의 **위치 인덱스(index)를 기억**해서 저장해 두고, 복원 과정에서 원래 자료와 동일한 위치에 Max value를 위치시킨다.

![([source](https://deep-learning-study.tistory.com/565))](./imgs2/Untitled%202.png)

([source](https://deep-learning-study.tistory.com/565))

- (장점) 이러한 방법은 `Bed of Nails Unpooling`에 비해 정보 손실을 방지할 수 있다. 물론, 위치 인덱스를 저장할 별도의 메모리(memory)가 필요하지만, 전체 CNN 아키텍처에서 적은 비율이므로 상관이 없다.
- (구현) [torch.nn.MaxUnpool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html)
  
    ```python
    # == 정의 == # 
    pool = nn.MaxPool2d(2, stride=2, return_indices=True) # position 값 저장 가능 
    unpool = nn.MaxUnpool2d(2, stride=2) # unpooling 
    input = torch.tensor([[[[ 1.,  2,  3,  4],
                            [ 5,  6,  7,  8],
                            [ 9, 10, 11, 12],
                            [13, 14, 15, 16]]]])
    
    # == 동작 == #
    output, indices = pool(input) # Maxpooling 과정에서 max value의 위치 인덱스 반환 
    unpool(output, indices) # Max Unpooling 
    >> tensor([[[[  0.,   0.,   0.,   0.],
                 [  0.,   6.,   0.,   8.],
                 [  0.,   0.,   0.,   0.],
                 [  0.,  14.,   0.,  16.]]]])
    
    # 입력 사이즈와 다르게 복원하기 
    unpool(output, indices, output_size=torch.Size([1, 1, 5, 5])) # (B,C, 5, 5) 사이즈로 복원
    >> tensor([[[[  0.,   0.,   0.,   0.,   0.],
                 [  6.,   0.,   8.,   0.,   0.],
                 [  0.,   0.,   0.,  14.,   0.],
                 [ 16.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.]]]])
    ```
    

### Bilinear Interpolation - ([link](https://blog.naver.com/PostView.naver?blogId=dic1224&logNo=220841161411&redirect=Dlog&widgetTypeCall=true&directAccess=false))

![([source](https://blog.naver.com/dic1224/220882679460))](./imgs2/869AF7FB-C09F-411C-86B4-A6322CB421FB.jpeg)

([source](https://blog.naver.com/dic1224/220882679460))

- 값들이 선형 관계를 갖는다고 가정하기 때문에 linear interpolation 이라고 부른다
- `bilinear interpolation` 은 linear interpolation은 x축과 y축에 대해 두 번 적용하는 것을 의미한다
    - P, Q 픽셀을 x축으로 linear interpolation
    - ★ 픽셀은 P, Q 픽셀에 대해 y축으로 linear interpolation
- (구현) [torch.nn.Upsample(scale_factor=2, mode=’bilinear’)](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
  
    ```python
    input = torch.Tensor([6,10,9,12]).view(1,1,2,2)
    >> tensor([[[[ 6., 10.],
    	         [ 9., 12.]]]])
    
    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    m(input)
    >> tensor([[[[ 6.0000,  7.3333,  8.6667, 10.0000],
                 [ 7.0000,  8.2222,  9.4444, 10.6667],
                 [ 8.0000,  9.1111, 10.2222, 11.3333],
                 [ 9.0000, 10.0000, 11.0000, 12.0000]]]])
    ```
    

### Deconvolution

- 많은 논문에서 `Deconvolution`과 `Transposed convolution`을 혼용하여 사용하지만 두 개념은 엄연히 다르다.
- `Deconvolution`은 convolution의 역연산(inverse operation)으로 수식으로 표현하면 다음과 같다:
  
    $\mathbf{Conv} : f,g \rarr f*g= h$  
    
    $f$ 는 filter(또는 kernel), $*$ 은 convolution 연산, $g$는 input, $h$ 는 output 이다. kernel과 output 을 알고 있는 상태에서 input 이 무엇이었는지를 구하는 것이 `Deconvonlution` 이다. 즉, 
    
    $\mathbf{Deconv}: \text{inverse matrix of }f,h \rarr  \text{inverse matrix of }f * h=g$
    

![([soucre](https://arxiv.org/abs/1505.04366))](./imgs2/Untitled%203.png)

([soucre](https://arxiv.org/abs/1505.04366))

- **ex)** 예를 들어 어떤 이미지를 convolution layer에 통과시켜서 얻은 feature map을 `Deconvolution` 하면 원래의 이미지로 복원할 수도 있다 (사실 convolution은 적분을 사용하기 때문에 $g$ 를 완벽하게 찾을 수는 없고 근사시킬 수는 있음).
- (구현) `nn.ConvTranspose2d(in_dim, out_dim, ...)` - ([link](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

### Transposed Convolution (= Backward Strided Convolution)

- `Deconvolution` 이 $f$ 의 역행렬을 구하는 것이라면, `Transposed Convolution` 은 학습을 통해 새로운 $f$ 를 구하는 것이 차이점이다. 수식으로 표현하면:
  
    $\mathbf{convTrans}: f',h \rarr f'* ^{T}h=g$
    
    $f'$ 는 학습을 통해 얻어진 새로운 filter 이다. 
    
- (유래) 왜 ‘transposed(전치, 행과 열이 바뀜)’ 라는 단어가 붙었느지를 이해하기 위해 Standard convolution이 실제로 어떻게 구현되는지 알 필요가 있다.

![Matrix multiplication for convolution: from a Large input image (4 x 4) to a Small output image (2 x 2) - ([source](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215))](./imgs2/Untitled%204.png)

Matrix multiplication for convolution: from a Large input image (4 x 4) to a Small output image (2 x 2) - ([source](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215))

- 위 그림은 $(3\times3 \text{ kernel}) * (4\times4 \text{ input})=(2\times 2 \text{ output})$ 을 도출하기 위한 **`Standard convolution`** 연산을 Matrix multiplication 으로 구현한 것이다.
    - $(3\times 3 \text{ kernel})$을 $(4\times 16 \text{ sparse matrix})$로 변환
    - $(4\times 4 \text{ input})$을 $(16\times 1 \text{ vector})$로 변환
    - $(2\times 2 \text{ output})$을 $(4\times 1 \text{ vector})$로 변환
- **`Transposed convolution`**은 $(3\times3 \text{ kernel}) * (2\times2 \text{ input})=(4\times 4 \text{ output})$ 을 도출하는 것을 목표로 아래와 같이 연산한다.
    - $(3\times 3 \text{ kernel})$을 $(16\times 4 \text{ sparse matrix})$로 변환
    - $(2\times 2 \text{ input})$을 $(4\times 1 \text{ vector})$로 변환
    - $(4\times 4 \text{ output})$을 $(16\times 1 \text{ vector})$로 변환
    
    ![Matrix multiplication for convolution: from a Small input image (2 x 2) to a Large output image (4 x 4) - ([source](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215))](./imgs2/Untitled%205.png)
    
    Matrix multiplication for convolution: from a Small input image (2 x 2) to a Large output image (4 x 4) - ([source](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215))
    
- 위의 두 그림을 비교하면 `**Transposed convolution**`을 수행할 때 Sparse matrix $C$ 가 전치됨(transposed)을 볼 수 있다. 그림에서 가중치 $\mathbf{w}$은 유지하고 위치만 전치되게 묘사 됐지만, 실제로는 가중치의 값도 학습되어 변한다 (learnable parameters). 사전에 input($g$), output($h$) 값을 알고 있으므로 학습을 통해 최적화된 $f'$ 값을 구할 수 있다.
- **(계산 방법)**
    - 1) 정석적인 방법
    - 2) 공백을 추가해서 standard convolution으로 구하는 방법

![([reference](https://realblack0.github.io/2020/05/11/transpose-convolution.html))](./imgs2/9669DEBD-7323-4EB0-9D18-803C62CA69C3.jpeg)

([reference](https://realblack0.github.io/2020/05/11/transpose-convolution.html))

![Transposed Convolution, kernel 3x3, stride 2 - ([source](https://www.reddit.com/r/learnmachinelearning/comments/e7iwk2/confused_about_transposed_convolution_help_please/))](./imgs2/Untitled%206.png)

Transposed Convolution, kernel 3x3, stride 2 - ([source](https://www.reddit.com/r/learnmachinelearning/comments/e7iwk2/confused_about_transposed_convolution_help_please/))

![([referece](https://realblack0.github.io/2020/05/11/transpose-convolution.html))](./imgs2/78410299-5F95-4A95-986A-FCE5D2F81DD8.jpeg)

([referece](https://realblack0.github.io/2020/05/11/transpose-convolution.html))

- (구현) `[nn.ConvTransposed2d(in_dim, out_dim, ... , dilation = 1, ...)](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d)`
  
    ```python
    input = torch.ones(1,1,2,2)
    m = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    output = m(input)
    
    output.size() 
    >> torch.Size([1, 1, 5, 5])
    ```
    

### 마치며

- Upsampling 에서 Max Unpooling, Bilinear Interpolation, Deconvolution, 그리고 Transposed Convolution 방식의 장단점 비교.
- 최적화된 Upsampling 을 위해서 학습이 가능한 Transposed Convolution 방식이 필요하다

---

**Reference**

1. [https://wikidocs.net/149326](https://wikidocs.net/149326)
2. [https://deep-learning-study.tistory.com/565](https://deep-learning-study.tistory.com/565)
3. [https://www.machinecurve.com/index.php/2021/12/28/how-to-use-upsample-for-upsampling-with-pytorch/](https://www.machinecurve.com/index.php/2021/12/28/how-to-use-upsample-for-upsampling-with-pytorch/)
4. [https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder#kl-divergence-term](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder#kl-divergence-term)
5. [https://d2l.ai/chapter_computer-vision/transposed-conv.html#connection-to-matrix-transposition](https://d2l.ai/chapter_computer-vision/transposed-conv.html#connection-to-matrix-transposition)
6. [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)
7. [https://www.cosmos.esa.int/web/machine-learning-group/convolutional-neural-networks-introduction](https://www.cosmos.esa.int/web/machine-learning-group/convolutional-neural-networks-introduction)
8. [https://simonjisu.github.io/deeplearning/2019/10/27/convtranspose2d.html](https://simonjisu.github.io/deeplearning/2019/10/27/convtranspose2d.html)