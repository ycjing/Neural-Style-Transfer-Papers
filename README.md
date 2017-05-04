# Neural-Style-Transfer-Papers <img class="emoji" alt=":art:" height="30" width="30" src="https://assets-cdn.github.com/images/icons/emoji/unicode/1f3a8.png">
Selected papers and corresponding collected codes in our review paper "**Neural Style Transfer: A Review**"

## Citation (to be updated)
If you find this repository useful for your research, please cite

```
@misc{XXX,
  title = {XXX},
  author = {XXX},
  booktitle = {XXX},
  year = {XXX}
}
```

## A Taxonomy of Current Methods

### 1. Descriptive Neural Methods Based On Image Iteration

####  1.1. MMD-based Descriptive Neural Methods

:white_check_mark: [**A Neural Algorithm of Artistic Style**] [[Paper]](https://arxiv.org/pdf/1508.06576.pdf) *(First Neural Style Transfer Paper)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/jcjohnson/neural-style)
*   [TensorFlow-based](https://github.com/anishathalye/neural-style)
*   [Caffe-based](https://github.com/fzliu/style-transfer) 
*   [Keras-based](https://github.com/titu1994/Neural-Style-Transfer)
*   [MXNet-based](https://github.com/pavelgonchar/neural-art-mini)
*   [MatConvNet-based](https://github.com/aravindhm/neural-style-matconvnet)

:white_check_mark: [**Image Style Transfer Using Convolutional Neural Networks**] [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) *(CVPR 2016)*

:white_check_mark: [**Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses**] [[Paper]](https://arxiv.org/pdf/1701.08893.pdf) *(CVPR 2017)*

:white_check_mark: [**Demystifying Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1701.01036.pdf)  *(Theoretical Explanation)*

:white_check_mark: [**Content-Aware Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1601.04568.pdf) 

:white_check_mark: [**Towards Deep Style Transfer: A Content-Aware Perspective**] [[Paper]](http://www.bmva.org/bmvc/2016/papers/paper008/paper008.pdf)  *(BMVC 2016)*

####  1.2. MRF-based Descriptive Neural Methods

:white_check_mark: [**Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis**] [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Li_Combining_Markov_Random_CVPR_2016_paper.pdf)  *(CVPR 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/chuanli11/CNNMRF)

:white_check_mark: [**Neural Doodle_Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork**] [[Paper]](https://arxiv.org/pdf/1603.01768.pdf) 

###  2. Generative Neural Methods Based On Model Iteration

:white_check_mark: [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**] [[Paper]](https://arxiv.org/pdf/1603.08155.pdf)  *(ECCV 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/jcjohnson/fast-neural-style)
*   [TensorFlow-based](https://github.com/hzy46/fast-neural-style-tensorflow)
*   [Chainer-based](https://github.com/yusuketomoto/chainer-fast-neuralstyle)

:sparkle: **Pre-trained Models:**

*   [Torch-models](https://github.com/ProGamerGov/Torch-Models)
*   [Chainer-models](https://github.com/gafr/chainer-fast-neuralstyle-models)


:white_check_mark: [**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**] [[Paper]](http://www.jmlr.org/proceedings/papers/v48/ulyanov16.pdf)  *(ICML 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/DmitryUlyanov/texture_nets)
*   [TensorFlow-based](https://github.com/tgyg-jegli/tf_texture_net)

:white_check_mark: [**Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis**] [[Paper]](https://arxiv.org/pdf/1701.02096.pdf)  *(CVPR 2017)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/DmitryUlyanov/texture_nets)

:white_check_mark: [**Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks**] [[Paper]](https://arxiv.org/pdf/1604.04382.pdf)  *(ECCV 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/chuanli11/MGANs)

:white_check_mark: [**A Learned Representation for Artistic Style**] [[Paper]](https://arxiv.org/pdf/1610.07629.pdf)  *(ICLR 2017)*

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization)

:white_check_mark: [**Fast Patch-based Style Transfer of Arbitrary Style**] [[Paper]](https://arxiv.org/pdf/1612.04337.pdf) 

:sparkle: **Code:**

*   [Torch-based](https://github.com/rtqichen/style-swap)

## Slight Modifications of Current Methods

###  1. Modifications of Descriptive Neural Methods

:white_check_mark: [**Exploring the Neural Algorithm of Artistic Style**] [[Paper]](https://arxiv.org/pdf/1602.07188.pdf) 

:white_check_mark: [**Improving the Neural Algorithm of Artistic Style**] [[Paper]](https://arxiv.org/pdf/1605.04603.pdf) 

:white_check_mark: [**Preserving Color in Neural Artistic Style Transfer**] [[Paper]](https://arxiv.org/pdf/1606.05897.pdf) 

:white_check_mark: [**Controlling Perceptual Factors in Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1611.07865.pdf) 

:sparkle: **Code:**

*   [Torch-based](https://github.com/leongatys/NeuralImageSynthesis)

###  2. Modifications of Generative Neural Methods

:white_check_mark: [**Instance Normalization：The Missing Ingredient for Fast Stylization**] [[Paper]](https://arxiv.org/pdf/1607.08022.pdf) 

:sparkle: **Code:**

*   [Torch-based](https://github.com/DmitryUlyanov/texture_nets)

:white_check_mark: [**Depth-Preserving Style Transfer**] [[Paper]](https://github.com/xiumingzhang/depth-preserving-neural-style-transfer/blob/master/report/egpaper_final.pdf) 

:sparkle: **Code:**

*   [Torch-based](https://github.com/xiumingzhang/depth-preserving-neural-style-transfer)

## Extensions to Specific Types of Images

:white_check_mark: [**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork**] [[Paper]](https://arxiv.org/pdf/1603.01768.pdf) 

:sparkle: **Code:**

*   [Torch-based](https://github.com/alexjc/neural-doodle）

:white_check_mark: [**Painting Style Transfer for Head Portraits Using Convolutional Neural Networks**] [[Paper]](http://dl.acm.org/citation.cfm?id=2925968)  *(SIGGRAPH 2016)*

:white_check_mark: [**Son of Zorn's Lemma Targeted Style Transfer Using Instance-aware Semantic Segmentation**] [[Paper]](https://arxiv.org/pdf/1701.02357.pdf) 

:white_check_mark: [**Artistic Style Transfer for Videos**] [[Paper]](https://arxiv.org/pdf/1604.08610.pdf)  *(GCPR 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/manuelruder/artistic-videos)

:white_check_mark: [**DeepMovie: Using Optical Flow and Deep Neural Networks to Stylize Movies**] [[Paper]](https://arxiv.org/pdf/1605.08153.pdf) 

## Application

:white_check_mark: [**Prisma**](https://prisma-ai.com/) 

:white_check_mark: [**Ostagram**](https://ostagram.ru/) 

:sparkle: **Code:**

*   [Website code](https://github.com/SergeyMorugin/ostagram)

:white_check_mark: [**Deep Forger**](https://deepforger.com/) 




## Application Papers

:white_check_mark: [**Bringing Impressionism to Life with Neural Style Transfer in Come Swim**] [[Paper]](https://arxiv.org/pdf/1701.04928.pdf) 

:white_check_mark: [**Imaging Novecento. A Mobile App for Automatic Recognition of Artworks and Transfer of Artistic Styles**] [[Paper]](https://www.micc.unifi.it/wp-content/uploads/2017/01/imaging900.pdf) 

## Blogs 

:white_check_mark: https://code.facebook.com/posts/196146247499076/delivering-real-time-ai-in-the-palm-of-your-hand/

:white_check_mark: https://research.googleblog.com/2016/10/supercharging-style-transfer.html

