# Down-sampling (Encoding)

![([source](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder))](./imgs/Untitled.png)

([source](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder))

- `Downsampling`은 신호처리(signal processing)에서 사용되는 용어로 sample의 개수를 줄이는 처리 과정을 말한다. 딥러닝에서는 [인코딩(Encoding)과정에서 feature의 개수를 줄이는 처리과정](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/#downsampling-your-inputs)이라 볼 수 있다.
- feature 개수를 줄이면 이를 다룰 파라미터도 줄어들기 때문에 차원이 축소되는 효과를 얻을 수 있다. 이는 곳 overfitting을 방지하는 효과로 이어진다.
- 딥러닝 모델 아키텍처에서 사용되는 몇 가지 대표적인 `Downsampling` 방식을 정리한다.

### Pooling - 대표값 구하기

- 커널 영역(kernel scope)에 포함된 data 중에서 대표 값 하나를 샘플링하는 처리 방식이다. 대표 값을 얻는 방식은 따르는 규칙에 따라 다르다; Max Pooling, Average Pooling, Global Average Pooling, and etc.

![Untitled](./imgs/Untitled%201.png)

- 보통 Average pooling 보다는 Max pooling을 더 많이 사용한다. Average pooling의 경우 **산술평균의 특성**을 가지기 때문에 **필연적으로 발생하는 문제**가 있다. 즉, 신호의 분산이 클 경우 [**대표 값과 실제 두드러지는 값과의 차이가 커진다**](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/#downsampling-your-inputs).
- 반면, Max pooling 연산이 좀더 간단하고 두드러지는 신호를 잘 전달할 수 있기 때문에 더 자주 사용된다. 때문에 논문과 교재에서 별도의 언급이 없이 Pooling layer를 언급한다면 Max pooling 이라 생각하면 된다.
- (장점) Pooling은 **학습 가능한 파라미터 (learnable parameters)가 없기** 때문에 kernel size를 키워도 차원의 저주를 피할 수 있는 방법중 하나다.
- `torch.nn` [pooling layers](https://pytorch.org/docs/stable/nn.html#pooling-layers)

### Dilated (Atrous) Convolution - 확장된 (구멍이 있는) 합성곱

- Convolution operation 을 활용해서도 `Downsampling` 이 가능하다. Pooling 과 달리 **학습 가능한 파라미터 (learnable parameters)**가 있어서 보다 최적화된 `Downsampling` 이 가능하다.
- (단점) `receptive field` 를 크게 만들기 어렵다. `receptive field`를 키우기 위해서는 kernel size 를 키워야 하는데, 그렇게 되면 **learnable parameters** 가 늘어나기 때문에 overfitting 을 유발할 수 있다.

![[2D convolution using a kernel size of 3, stride of 1 and padding](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)](./imgs/1_1okwhewf5KCtIPaFib4XaA.gif)

[2D convolution using a kernel size of 3, stride of 1 and padding](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

- (해법) 이러한 단점을 보완하기 위해 고안된 방법이 Dilated (Atrous) convolution 이다.
- [atrous 는 프랑스어로 ‘A trous = 구멍이 있는’](https://better-tomorrow.tistory.com/entry/Atrous-Convolution) 이라는 뜻이다. 즉, Atrous Convolution 은 kernel 사이가 구멍이 난 듯 띄워져 있는 상태로 수행되는 합성곱 연산이다.
- 이러한 kernel 사이의 간격을 **dilation rate**라고 정의하며, 일반적인 convolution의 경우 **dilation rate=1** 이다. **ex)** 즉 위 그림의 일반적인 convolution의 경우 dilation rate=1 이고, 아래 그림의 경우에는 dilation rate=2 이다.
- 이렇게 되면 **동일한 파라미터 개수**를 가짐에도 불구하고 receptive field 를 더 키울 수 있다.

![[2D convolution using a 3 kernel with a dilation rate of 2 and no padding](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)](./imgs/1_1okwhewf5KCtIPaFib4XaA%201.gif)

[2D convolution using a 3 kernel with a dilation rate of 2 and no padding](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

![Untitled](./imgs/Untitled%202.png)

- (활용) Semantic Segmentation 에서 높은 성능을 내기 위해서는 [**픽셀단위의 조밀한 예측(dense prediction)**](https://better-tomorrow.tistory.com/entry/Atrous-Convolution)이 필요하다. 이를 위해 입력 이미지에 대해서 CNN의 마지막 feature map의 한 픽셀(pixel)이 어느 정도의 영역을 커버할 수 있는지를 결정하는 **receptive field**의 크기가 중요하다 (= contextual info에 좋음).
- 이때 Atrous convolution을 사용하면 일반적인 convolution (standard convolution)과 동일한 크기의 파라미터를 가짐에도 receptive field 는 키울 수 있다.
- 아래 그림은 kernel=7 에서 diliation rate=1 일 때와 dilation rate=2 일 때의 feature map을 보여준다. 동일한 해상도(resolution) 에서 일반적인 convolution의 결과는 feature map이 sparse하게 추출된다. 반면, Atrous convolution 의 경우에는 feature map이 dense하게 추출된다. 이는 feature map 의 한 픽셀이 커버하는 receptive field 의 영향력 크기에서 차이가 나기 때문이다.

![DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](./imgs/Untitled%203.png)

DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

- `torch.nn.Conv2d( ..., dilation=2, ...)` - ([source](https://gaussian37.github.io/dl-pytorch-conv2d/))

### Depthwise convolution

- Standard convolution 의 연산 방법은 입력 영상의 채널(channel)에 각각 filter를 적용하고 그 출력물을 더하여 하나의 feature map을 출력한다.
- (단점) 이러한 방식은 **특정 채널의 Spatial Feature Map 을 추출하는 것이 불가능**하다는 단점이 있다. 또한 Convolution layer 가 깊어질수록 kernel 개수가 증가하면서 **연산량 또한 증폭**된다.
- 아래 그림은 RGB multi-channel 이미지에서 대한  standard convolution 연산의 예시다.
  
    $F = \mathbf{conv}(W, \mathbf{x})+b$
    

![RGB image ([source](https://unsplash.com/photos/_d3sppFprWI))](./imgs/Untitled%204.png)

RGB image ([source](https://unsplash.com/photos/_d3sppFprWI))

![[Each of the kernels of the filter “slides” over their respective input channels, producing a processed version of each. Some kernels may have stronger weights than others, to give more emphasis to certain input channels than others](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)](./imgs/1_8dx6nxpUh2JqvYWPadTwMQ.gif)

[Each of the kernels of the filter “slides” over their respective input channels, producing a processed version of each. Some kernels may have stronger weights than others, to give more emphasis to certain input channels than others](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

![[The kernels of a filter each produce one version of each channel, and the filter as a whole produces one overall output channel](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)](./imgs/1_8dx6nxpUh2JqvYWPadTwMQ%201.gif)

[The kernels of a filter each produce one version of each channel, and the filter as a whole produces one overall output channel](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

![[add the bias term](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)](./imgs/1_8dx6nxpUh2JqvYWPadTwMQ%202.gif)

[add the bias term](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

- (해법) 이러한 단점을 해결하기 위해 제시된 것이 `Depthwise convolution`이다. 특징은 각 channel 마다 spatial feature를 추출하기 위해 channel 별  filter가 존재하고, 때문에 **input channel 수 = ouput channel 수** 로 같게 된다.
- 아래 그림은 RGB 이미지에 `Depthwise convolution`을 적용한 예시다. 이처럼 Feature map 결과가 하나로 병합되지 않고 **(RGB) 채널 각각에 대한 Spatial Feature Map을 얻게 된다**.

![([source](https://coding-yoon.tistory.com/122))](./imgs/Untitled%205.png)

([source](https://coding-yoon.tistory.com/122))

- (장점) `Depthwise convolution` 을 사용하면 파라미터 (parameter) 개수를 줄일 수 있다. $C$ 채널을 가진 입력 이미지를 기준으로 `Standard convolution` 은 한 개의 feature map을 생성하기 위해 $C \times K\times K$  사이즈의 커널(kernel)이 필요하다. 이러한 feature map을 $M$ 개 생성한다면 $M\times C \times K\times K$   개의 파라미터가 필요하다.
- 반면 `Depthwise convolution` 을 사용하면 입력 채널과 출력 특징 맵의 개수 $M$이 서로 같아야 하므로 ($C=M$) 최종적으로 생성되는 feature map은 $C \times K \times K$ 개의 파라미터가 필요하다.
    - W : width
    - H : height
    - C : channel
    - K : kernel size
    - M : channel size

![Untitled](./imgs/Untitled%206.png)

- (장점) 계산 비용(computational cost) 측면에서도 이점이 있다. 계산 비용은 곱셈과 덧셈 연산의 횟수와 관련있다.

![([source](https://wikidocs.net/147019))](./imgs/Untitled%207.png)

([source](https://wikidocs.net/147019))

- (구현) 입력 채널 수 만큼 그룹을 나눈 [Grouped convolution](https://supermemi.tistory.com/116)으로 구현이 가능하다 ([ref1](https://supermemi.tistory.com/118))([ref2](https://down-develope.tistory.com/16))
    - (i.e) `nn.Conv2d(in_channel=n_dim, out_channel=n_dim, ..., groups=n_dim)`

### Depthwise-separable convolution

- `Depthwise convolution` 뒤에 $1 \times 1$ convolution을 연결한 구조다.
- `Standard convolution`은 spatial dimension과 channel dimension을 동시에 처리한다. 반면 `Depthwise-separable convolution`은 이를 분리시켜 따로 처리하는 방식이다. 두 축으로 따로 처리해도 최종 결과는 두 축 모두를 처리한 것이기 때문에 **Standard convolution의 역할을 대체**할 수 있다.

![([source](https://paperswithcode.com/method/depthwise-separable-convolution))](./imgs/Untitled%208.png)

([source](https://paperswithcode.com/method/depthwise-separable-convolution))

- (장점) `Standard convolution`에 비해 계산 비용(computational cost)과 파라미터(parameter) 개수 측면에서 훨씬 적다.
- (Num_parameters) 채널이 $C$ 인 입력에 대해 $K\times K$ 사이즈의 커널로 feature map을 $M$ 개 생성한다면,
    - `Standard convolution`은 $M \times C \times K \times K$ 개의 파라미터가 필요하다.
    - 반면, `Depthwise-seperable convolution`은 Depthwise convolution의 결과에 $C\times1\times1$ 크기의 convolution을 $M$ 개 연결한 것이므로 총 $(C\times K \times K)+(M\times C \times 1 \times 1)$ 개의 파라미터가 필요하다.
- 이는 $C(K^2+M) : MCK^2$ 비율로 `Standard convolution`에 비해 $\frac{K^2+M}{MK^2}$ 배 적다.

![([source](https://coding-yoon.tistory.com/122))](./imgs/Untitled%209.png)

([source](https://coding-yoon.tistory.com/122))

- 계산 비용(computational cost) 또한 output 사이즈가 $H\times W$ 일 때, $(num\_parameters \times H \times W)$ 번 연산해야 하므로 `Standard convolution` 보다 적다.

![([source](https://wikidocs.net/147019))](./imgs/Untitled%2010.png)

([source](https://wikidocs.net/147019))

- (구현) [depthwise convolution 후 pointwise convolution을 연쇄적으로 적용](https://coding-yoon.tistory.com/122)
  
    ```python
    nn.Sequential( nn.Conv2d(n_dim, n_dim, ..., groups=n_dim), # depthwise 
    							 nn.Conv2d(n_dim, output_dim, kernel_size=1), # pointwise
                  )
    ```
    

### 마치며

- Downsampling 에서 Pooling 방식와 Convolution  방식의 장단점 비교.
- 최적화된 Downsampling 을 위해서는 학습이 가능한 Convolution 방식이 필요하지만 receptive field를 키울 수록 계산 비용이 커지는 이슈가 있다. 이를 해결하는 방법들에 대해 알아보자.
- `Atrous convolution`과 `Depthwise-separable convolution`은 semantic segmentation 모델 중 상위 성능을 보이는 [Deep lab](https://www.google.com/search?q=deeplab+semantic+segmentation&bih=1511&biw=1422&hl=ko&sxsrf=APq-WBs64uN5iznRbVH53CZ3792V47TuzQ%3A1644750964652&ei=dOgIYt-XJ4eImAWnn4Bg&oq=deep+lab+seman&gs_lcp=Cgdnd3Mtd2l6EAMYADIGCAAQChAeOgcIABBHELADOgcIIxDqAhAnOgQIIxAnOg4ILhCABBCxAxDHARDRAzoICAAQgAQQsQM6CwgAEIAEELEDEIMBOg0ILhCABBDHARDRAxAKOhEILhCABBCxAxCDARDHARDRAzoFCAAQgAQ6CwguEIAEELEDEIMBOggILhCABBCxAzoFCC4QgAQ6CggAEIAEELEDEAo6BwgAEIAEEAo6BQguEMsBOgUIABDLAToHCAAQChDLAToECAAQCjoECAAQHjoGCAAQCBAeOgQIABANOgYIABANEB46CAgAEAgQDRAeSgQIQRgASgQIRhgAUJUJWMArYLg0aAVwAXgAgAGEAYgB_Q2SAQQwLjE1mAEAoAEBsAEKyAEKwAEB&sclient=gws-wiz)에 공통적으로 사용되므로 숙지해두자.

---

**Reference** 

1. [https://wikidocs.net/147019](https://wikidocs.net/147019)
2. [https://gaussian37.github.io/dl-concept-global_average_pooling/](https://gaussian37.github.io/dl-concept-global_average_pooling/)
3. [https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
4. [https://better-tomorrow.tistory.com/entry/Atrous-Convolution](https://better-tomorrow.tistory.com/entry/Atrous-Convolution)
5. [https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
6. [https://youtu.be/vfCvmenkbZA](https://youtu.be/vfCvmenkbZA)
7. [https://hichoe95.tistory.com/48](https://hichoe95.tistory.com/48)
8. [https://eehoeskrap.tistory.com/431](https://eehoeskrap.tistory.com/431)
9. [https://youtu.be/T7o3xvJLuHk](https://youtu.be/T7o3xvJLuHk)
10. [https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder#kl-divergence-term](https://velog.io/@tobigs16gm/VAEVariational-Auto-Encoder#kl-divergence-term)