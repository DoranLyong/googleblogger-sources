# 블로그 글쓰기(Moments in Time Dataset)

## Table of Contents

---

## Abstract

- one million labeled videos for 339 classes corresponding to dynamic events unfolding within 3 seconds.
- the average number of labeled videos per class is 1,757 with a median of 2,775.
- meaningful events include not only **people**, but also **objects**, **animals**, and **natural phenomena** (human & non-human).

## Dataset Overview

![sample Videos.png](./page_img/(TPAMI2019)MiT_sample_1.png)



## Notable Experimental Setup

**Data**. ****They generate a training set of 802,264 videos with between 500 and 5,000 videos per class for 339 different classes. Then, for evaluation of performance a validation set of 33,900 videos with 100 videos for each class is prepared. Finally, they withhold a test set of 67,800 videos consisting of 200 videos per class.

- $train:val:test = 802,264 : 33,900 : 67,800\approx 8.8:0.37:0.75$

**Preprocessing**. In this section we can summarize how to prepare inputs which are applied to both ***Two-Stream*** and ***I3D***. Firstly, they extract RGB frames from the videos at 25 fps and resize the RGB frames to a standard $340\times256$ pixels. In the interest of performance, *optical flow*(**OF**) on consecutive frames is pre-computed using an off-the-shelf implementation of TVL1 OF algorithm from the OpenCV. Especially, **for fast computation** the values of optical flow fields are discretized into integers, the displacement is clipped with a maximum absolute value of 15, and the value range is scaled to 0-255. The $x$ and $y$ displacement fields of every optical flow frame are then stored as two grayscale images to reduce storage. Additionally, to correct for camera motion, they subtract the mean vector from each displacement field in the stack. **On the augmentation steps**, they use random cropping and subtract the ImageNet mean/std from images (이미지 정규화 값을 ImageNet 셋팅으로 함).

**Evaluation metric**. They introduce **top-1** and **top-5** classification accuracy as the scoring metrics. Top-1 accuracy indicates the percentage of testing videos for which the top confident predicted label is correct. Top-5 accuracy indicates the percentage of the testing videos for which the ground-truth label is among the top 5 ranked predicted labels. Especially, top-5 accuracy is appropriate for video classification as videos may contain multiple actions. **For evaluation** they randomly select 10 crops per frame and average the results. 

**Baseline models for Video Classification**. They split the results into three modalities (**spatial**, **temporal**, and **auditory**), as well as **spatiotemporal** such as (*[TSN](https://www.google.com/search?q=temporal+segment+networks%3A+towards+good+practices+for+deep+action+recognition.&sxsrf=AOaemvLRkSmN86My8nE95sUhx0glMwtSBA%3A1642044868850&ei=xJ3fYZ6kM5fahwOLiIjwDw&ved=0ahUKEwielvPo5a31AhUX7WEKHQsEAv4Q4dUDCA4&uact=5&oq=temporal+segment+networks%3A+towards+good+practices+for+deep+action+recognition.&gs_lcp=Cgdnd3Mtd2l6EAMyBAgAEB4yBAgAEB46BwgjELADECdKBAhBGAFKBAhGGABQxydY1kFg8ERoAXAAeACAAXyIAdwSkgEEMS4yMZgBAKABAcgBAcABAQ&sclient=gws-wiz)* and *[TRN](https://www.google.com/search?q=Temporal+relational+reasoning+in+videos&sxsrf=AOaemvKUOiLFIF3R0nl-Qa55jESYbXd1BA%3A1642044878964&ei=zp3fYaOnOpn6-QaVwaSYCw&ved=0ahUKEwijxtzt5a31AhUZfd4KHZUgCbMQ4dUDCA4&uact=5&oq=Temporal+relational+reasoning+in+videos&gs_lcp=Cgdnd3Mtd2l6EAMyBQgAEIAEMgUIABCABDoHCAAQRxCwA0oECEEYAEoECEYYAFDDA1jDA2CDCGgDcAJ4AIABbogBbpIBAzAuMZgBAKABAqABAcgBCsABAQ&sclient=gws-wiz)*). 

**Spatial modality**. They introduce a 50 layer resnet ($Resnet50$) trained on randomly selected RGB frames from each video. The training setups for this are divided into three versions; training from scratch ($ResNet50-scratch)$, initialized on [Places](https://www.google.com/search?q=+Learning+deep+features+for+scene+recognition+using+places+database&sxsrf=AOaemvL7oLq28B4JyTqkQ7_A48RyUm-w0w%3A1642044911646&ei=753fYfDyJoH8wQOXpqD4CA&ved=0ahUKEwiwpqf95a31AhUBfnAKHRcTCI8Q4dUDCA4&uact=5&oq=+Learning+deep+features+for+scene+recognition+using+places+database&gs_lcp=Cgdnd3Mtd2l6EAMyBQgAEIAEMgUIABDLAToHCAAQRxCwA0oECEEYAEoECEYYAFCnA1inA2CCCGgDcAJ4AIABZYgBZZIBAzAuMZgBAKABAqABAcgBBsABAQ&sclient=gws-wiz) ($ResNet50-Places)$, and initialized on [ImageNet](https://www.google.com/search?q=Imagenet+classification+with+deep+convolutional+neural+networks&sxsrf=AOaemvKe-IstQuIRJ9OSIFryWgACNuQY4Q%3A1642045290334&ei=ap_fYY_uE8bywQOCvp7gDw&ved=0ahUKEwiPy_Cx5631AhVGeXAKHQKfB_wQ4dUDCA4&uact=5&oq=Imagenet+classification+with+deep+convolutional+neural+networks&gs_lcp=Cgdnd3Mtd2l6EAMyCggAEIAEEIcCEBQyCggAEIAEEIcCEBQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABDLATIFCAAQywEyBQgAEIAEMgUIABDLAToHCAAQRxCwA0oECEEYAEoECEYYAFCdA1idA2CcBmgDcAJ4AIABY4gBY5IBATGYAQCgAQKgAQHIAQrAAQE&sclient=gws-wiz) ($ResNet50-ImageNet)$. In testing, they average the prediction from 6 equi-distance frames. 

**Auditory modality**. Sound signals contain complementary or even mandatory information for recognition of particular classes, such as cheering or talking. So, they use raw waveforms as the input modality and finetune a SoundNet network which was pretrained on 2 million unlabeled videos from Flickr with the output layer changed to predict moment classes (*SoundNet*). 

**Temporal modality**. Following the [Two-Stream paradigm](https://www.google.com/search?q=Two-stream+convolutional+networks+for+action+recognition+in+videos.+&sxsrf=AOaemvJ9gXdxJMRixkvH3osh8fN80Hpj-Q%3A1642045311604&ei=f5_fYfOwJJaM-AbV9KW4Dg&ved=0ahUKEwiz7IK85631AhUWBt4KHVV6CecQ4dUDCA4&uact=5&oq=Two-stream+convolutional+networks+for+action+recognition+in+videos.+&gs_lcp=Cgdnd3Mtd2l6EAMyBAgAEB4yBAgAEB4yBAgAEB46BwgAEEcQsANKBAhBGABKBAhGGABQwgNYwgNg_wVoA3ACeACAAXmIAXmSAQMwLjGYAQCgAQKgAQHIAQrAAQE&sclient=gws-wiz), the optical flow between adjacent frames, which encoded in Cartesian coordinates as displacements, are computed by stacking together 5 consecutive frames to form a 10 channel image (the $x$ and $y$ displacement channels). Then, the first convolutional layer of [BNInception](https://www.google.com/search?q=.+Batch+normalization%3A+Accelerating+deep+network+training+by+reducing+internal+covariate+shift&sxsrf=AOaemvKUaC8hW3Gsml1_5QM6KdIbARwVGQ%3A1642045621085&ei=taDfYf7IBOSn2roP0Yyr4As&ved=0ahUKEwi-98vP6K31AhXkk1YBHVHGCrwQ4dUDCA4&uact=5&oq=.+Batch+normalization%3A+Accelerating+deep+network+training+by+reducing+internal+covariate+shift&gs_lcp=Cgdnd3Mtd2l6EAMyBQgAEIAEOgcIABBHELADSgQIQRgASgQIRhgAUK8EWK8EYPkIaANwAngAgAFjiAFjkgEBMZgBAKABAqABAcgBCsABAQ&sclient=gws-wiz) model is modified in order to accept 10 input channels (Batch Norm-Inception).

**Spatial-Temporal modality**. Three representative action recognition models (at that moment) are introduced: Temporal Segment Networks(TSN), Temporal Relation Networks(TRN), and Inflated 3D Convolutional Networks (I3D). 

- TSN aims to efficiently capture the long-range temporal structure of videos using a sparse frame-sampling strategy. The TSN’s spatial stream(*TSN-spatial*) is fused with an optical flow stream (*TSN-Flow*) via average consensus to form the two stream TSN. The base model for each stream is a BNInception model with three time segments.
- TRN learns temporal dependencies between video segments that best characterize a particular action. This “plug-and-play” module can simultaneously model several short and long range temporal dependencies to classify actions that unfold at multiple time scales.
- I3D inflates the convolutional and pooling kernels of a pretrained 2D network to a third dimension. The inflated 3D kernel is initialized from the 2D model by repeating the weights from the 2D kernel over the temporal dimension. This improves learning efficiency and performance as 3D models contain far more parameters than their 2D counterpart and a strong initialization greatly improves training.

**Ensemble**. Combination of the top performing model of each modality (spatial + spatiotemporal + auditory). (skip this part) 

## Results

- **TABLE 2** shows that using the model trained on MiT dataset performs more generalized results so that it is applicable for transfer learning (ImageNet 처럼 video understanding에서 사전 학습 모델로 사용하기 적합하다는 주장).
- This dataset presents a difficult task for the field of computer vision as the labels correspond to different levels of abstraction (a verb like "falling" can apply to many different agents and scenarios and involve objects and scenes of different categories)

---

## Reference

[1] [Moments in Time Dataset: one million videos for event understanding, TPAMI2019](http://moments.csail.mit.edu/TPAMI.2019.2901464.pdf)

[2] [Moments in Time Dataset Homepage](http://moments.csail.mit.edu/#)