# PaddleHub 迁移学习与ULMFiT微调策略

## 一、简介

迁移学习（Transfer Learning）顾名思义就是将模型在其它领域学到的知识迁移至目标领域的学习过程中，帮助模型取得更好的学习效果。通过迁移学习，模型能够在短时间内取得更好的学习效果。

迁移学习通常由预训练阶段、微调阶段两部分组成。预训练阶段通常在大规模数据集中进行，例如CV任务中的ImageNet包含千万张标注图片，NLP任务中的English Wikipedia包含25亿个单词，这样训练得到的预训练模型能够很好地学习到不同领域中的通用知识。但预训练阶段使用的数据集往往与我们想要完成的任务的数据集存在差异，例如如果你只是想简单地判断一副图像是否是玫瑰花，ImageNet就没有提供相关的标注，因此为了更好的学习目标领域的知识，通常还需要对预训练模型参数进行微调。

PaddleHub中集成了ERNIE、BERT、LAC、ELMo等[NLP预训练模型](https://www.paddlepaddle.org.cn/hub)，ResNet、GoogLeNet、MobileNet等[CV预训练模型](https://www.paddlepaddle.org.cn/hub)；以及Adam + Weight Decay、L2SP、ULMFiT等微调策略。本文主要介绍ULMFiT微调策略在PaddleHub中的使用。

## 二、 ULMFiT

[ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)提出了三种微调策略：slanted triangular learning rates、discriminative fine-tuning以及gradual unfreezing。

1. slanted triangular learning rates

   slanted triangular learning rates（STLR）是一种学习率先上升再下降的微调策略，如下图所示：

   ![image-20190917170542047](https://user-images.githubusercontent.com/11913168/65138331-61316c80-da3d-11e9-9acb-c29385785e24.png)

   其计算公式如下：

   ![image-20190917170707549](https://user-images.githubusercontent.com/11913168/65138349-6bec0180-da3d-11e9-821d-98bc7f2d6f1e.png)

   其中T表示训练的迭代次数，PaddleHub会自动计算训练总step数；cut_frac是学习率上升在整个训练过程占用的比例；cut表示学习率转折处的step；t表示当前的step；p表示当前step学习率的缩放比例；ratio表示LR最低下降至最大学习率η<sub>max</sub>的几分之一；η<sub>t</sub>表示当前step的学习率。论文中，作者采用的超参设置为：cut_frac=0.1, ratio=32, η<sub>max</sub>=0.01。本次策略实验过程中，保持ratio=32, η<sub>max</sub>=0.01不变，仅调整cut_frac。

2. Discriminative fine-tuning

   Discriminative fine-tuning 是一种学习率逐层递减的策略，通过该策略可以减缓底层的更新速度。其计算公式为：

   <div align=center>η<sup>l-1</sup>=η<sup>l</sup>/factor</div>
其中η<sup>l</sup>表示第l层的学习率；η<sup>l-1</sup>表示第l-1层的学习率；factor表示逐层衰减率，论文中作者根据经验设置为2.6。这个策略能够让模型微调过程中不断减缓底层的更新速度，尽可能的保留预训练模型中习得的底层通用知识。PaddleHub通过op的拓扑关系自动计算模型的层次，因此针对这一策略，PaddleHub提供了一个额外的超参：dis_blocks，用于设置划分的层数，默认为3，如果设置为0，则不采用Discriminative fine-tuning。

3. Gradual unfreezing

   Gradual unfreezing是一种逐层解冻的策略，通过该策略可以优先更新上层，再慢慢解冻下层参与更新。PaddleHub在Gradual unfreezing策略中引入了一个额外的超参：frz_blocks，其作用与默认值与第2点提到的dis_blocks一致。在微调过程中，每经过一个epoch，模型解冻一个block，所有未被冻结的block都会参与到模型的参数更新中。

本文接下来将对ULMFiT策略在NLP以及CV任务中的使用进行实验说明，由于slanted triangular learning rates与warmup + linear decay在原理上相似，本文也将对比slanted triangular learning rates与warmup + linear decay的实验效果。

## 三、 在NLP迁移学习中使用ULMFiT策略

1. 数据集与预训练模型的选择

   本次实验选取两个数据集，一个为中文数据集Chnsenticorp，另一个为英文数据集CoLA，两个数据集的训练集数据规模相似，前者包含9601个句子，后者包含8551个句子。针对中文数据集Chnsenticorp与英文数据集CoLA，本次实验分别使用ELMo与“bert_uncased_L-12_H-768_A-12”作为其预训练模型。

2. Baseline与实验设置

   Baseline不采用任何策略，学习率在微调过程中保持恒定。Chnsenticorp与CoLA任务均只使用1张显卡，Batch size均设置为32，总共迭代3个epoch。Chnsenticorp设置学习率为1e-4，由于采用ELMo预训练模型，无需设置句子最大长度；CoLA设置学习率为5e-5，句子最大长度设置为128。实验效果如下表所示：

   | -             | Chnsenticorp | CoLA              |
   | :------------ | :----------- | :---------------- |
   | Module        | ELMo         | Bert              |
   | Batch size    | 32           | 32                |
   | Num epoch     | 3            | 3                 |
   | Learning rate | 1e-4         | 5e-5              |
   | Max length    | -            | 128               |
   | Dev           | acc = 0.8766 | matthews = 0.5680 |

   其中Chnsenticorp采用准确率（accuracy）得分，CoLA采用马修斯相关系数（matthews correlation coefficient）得分。

   在后续调优过程中，如无特别说明，实验设置（例如Batch size、Num epoch等）均与Baseline一致。

3. slanted triangular learning rates（STLR）策略实验与分析

   理论上，STLR与warmup + linear decay相似，因此本次实验同时对warmup + linear decay策略进行了实验。实验中STLR的超参仅调整cut_fraction，其它超参与论文设置一致；warm up proportion为学习率上升在总步数中的比重，linear decay则在warmup结束的时刻开始，linear decay的终止学习率设置了2组，1组为0，另一组与STLR的终止学习率一致，为learning rate/32。实验结果如下表所示：

   | slanted triangular cut_fraction     | 0（Baseline）  | 0      | 0      | 0          | 0      | 0.1     | 0.2    |
   | :---------------------------------- | :------------ | :----- | :----- | :--------- | :----- | :------ | :------ |
   | warm up proportion                  | 0             | 0.1    | 0.2    | 0.1        | 0.2    | 0       | 0       |
   | linear decay end                    | - Unused      | 0      | 0      | lr/32      | lr/32  | -       | -       |
   | Chnsenticorp                        | 0.8766        | 0.8725 | 0.8758 | **0.8825** | 0.8733 | 0.8791  | 0.8791  |
   | CoLA                                | 0.5680        | 0.5780 | 0.5786 | **0.5887** | 0.5826 | 0.5880  | 0.5827  |

   从实验结果可以看到，STLR实验效果上与warmup + linear decay (end learning rate = lr/32)接近，并且在两个任务中模型的性能都得到了提升。建议用户在尝试slanted triangular或warmup+linear decay策略时尝试更多的超参设置，在使用warmup+linear decay策略时考虑设置linear decay end。

4. Discriminative fine-tuning策略实验与分析

   本小节对Discriminative fine-tuning策略进行实验分析。固定训练总epoch=3，实验结果如下表所示：

   | dis_blocks        | -<br />（Baseline） | 3          | 5      |
   | ----------------- | ------------------- | ---------- | ------ |
   | epoch             | 3                   | 3          | 3      |
   | Chnsenticorp      | **0.8766**          | 0.8641     | 0.6766 |
   | CoLA           | 0.5680              | **0.5996** | 0.5749 |

   由于Discriminative fine-tuning策略会降低模型底层的更新速度，影响模型的拟合能力。实验结果表明，dis_blocks设置过大会导致模型性能明显下降。为了提升模型拟合能力，本小节继续增大epoch大小至5、8。

   对于Chnsenticorp，实验结果如下表所示：

   | dis_blocks        | -<br />（Baseline） | -          | 5      | -          | 5          |
   | ----------------- | ------------------- | ---------- | ------ | ---------- | ---------- |
   | epoch             | 3                   | 5          | 5      | 8          | 8          |
   | Chnsenticorp      | 0.8766              | 0.8775     | 0.8566 | 0.8775     | **0.8792** |

   可以看到当dis_blocks=5时，epoch=8时，模型性能超越Baseline。

   在CoLA任务中，dis_block=3，epoch=3时的模型得分已经超越了Baseline，因为可以进一步增大dis_blocks，观察其实验效果，结果如下表所示：

   | dis_blocks | -<br />（Baseline） | 3          | -      | 7      | -      | 7      |
   | ---------- | ------------------- | ---------- | ------ | ------ | ------ | ------ |
   | epoch      | 3                   | 3          | 5      | 5      | 8      | 8      |
   | CoLA    | 0.5680              | **0.5996** | 0.5680 | 0.5605 | 0.5720 | 0.5788 |

   实验结果表明，dis_blocks过大同样会导致性能下降的问题，当dis_blocks=7时，模型在epoch=5性能低于Baseline (epoch=3)，直至epoch=8才略微超过Baseline (epoch=8)，但仍显著低于dis_blocks=3，epoch=3的模型表现。建议用户采用discriminative fine-tuning时，应当设置较小的dis_blocks，如果设置过大的dis_blocks，则需提升训练的epoch。

5. Gradual unfreezing策略实验与分析

   本小节对Gradual unfreezing策略进行实验分析，frz_blocks设置为3，第一个epoch只更新最顶层的block，此后每一个epoch解冻一个block参与更新，实验结果如下表所示：

   | gradual unfreezing | -（baseline） | 3      |
   | :----------------- | :------------ | :----- |
   | Chnsenticorp    | 0.8766        | **0.8850** |
   | CoLA            | 0.5680        | **0.5704** |

   实验结果表明通过延后更新预训练模型中的底层参数，该策略不论是对Chnsenticorp数据集还是对CoLA数据集均有效。

## 四、在CV迁移学习中使用ULMFiT策略

1. 数据集与预训练模型的选择

   本小节采用resnet50作为预训练模型。基于数据规模、任务数据集与预训练数据集的相似度，本次实验选择了以下四个数据集：

   - **indoor67**（相似度小，规模小）：该数据集包含1.5万个样例，标签包含concert_hall, locker_room等67个室内物体，这些标签未出现在预训练数据集ImageNet中。理论上，相似度小规模小的数据集在微调时既不能更新得太快导致过拟合，也不能太小导致欠拟合，是最难调整的一类数据集。
   - **food101**（相似度小，规模大）：该数据集包含10万个样例，标签包含Apple pie, Baby back ribs等101个食品，这些标签同样几乎没有出现在ImageNet中。理论上，相似度小规模大的数据集可以较快更新，是模型充分拟合。
   - **dogcat**（相似度大，规模大）：该数据集包含2.2万个样例，标签只有dog和cat两类，单类数据规模较大，且均出现在ImageNet中。理论上，相似度大规模大的数据集不论采用怎样的策略均能获得非常良好的模型性能，通过细致地调整策略可以略微提升模型性能。
   - **dogcat 1/10**（相似度大，规模小）：该数据集由dogcat数据集中随机抽取1/10个样例组成，包含0.22万个样例，”dog”/”cat”两类标签。理论上，相似度大规模小的数据集由于与预训练数据集相似度大，模型在微调过程中可以保留更多的预训练参数。

2. Baseline与实验设置

   Baseline未采用任何策略，学习率在微调过程中保持恒定。所有任务均使用两张显卡，Batch size设置为40，训练迭代一个epoch，学习率均设置为1e-4，评估指标为准确率（accuracy）。实验效果如下表所示：

   | -                | **indoor67**   | **food101**    | **dogcat**     | **dogcat 1/10** |
   | :--------------- | :------------- | :------------- | -------------- | --------------- |
   |  Batch size      | 40             | 40             | 40             | 40              |
   | Num epoch        | 1              | 1              | 1              | 1               |
   | Learning rate    | 1e-4           | 1e-4           | 1e-4           | 1e-4            |
   | Dev              | 0.6907         | 0.7272         | 0.9893         | 1.0             |
   | Test             | 0.6741         | 0.7338         | 0.9830         | 0.9719          |

   在后续调优过程中，如无特别说明，实验设置（例如Batch size、Num epoch等）均与Baseline一致。

3. slanted triangular learning rates（STLR）策略实验与分析

   本小节采用STLR微调策略进行实验，实验结果如下表所示，其中cut_fraction=0为Baseline，不采用任何策略：

   | cut_fraction | indoor67<br />相似度小规模小      | food101<br />相似度小规模大          | dogcat <br />相似度大规模大      | dogcat 1/10<br />相似度大规模小   |
   | :----------- | :-------------------------------- | :----------------------------------- | :------------------------------- | :-------------------------------- |
   | 0 (baseline)            | dev:  0.6907<br />test:0.6741     | dev: 0.7272<br />test:0.7338         | dev:0.9893<br />test:0.9830      | dev:**1.0**<br />test:0.9719      |
   | 0.01         | dev: 0.7148<br /> test:0.7053     | dev:**0.7637**<br /> test:**0.7656** | dev: **0.9946**<br />test:0.9924 | dev: *0.4481* <br />test:*0.5346* |
   | 0.05         | dev: **0.7226**<br /> test:0.7130 | dev:0.7605<br /> test:0.7612         | dev: 0.9901<br />test:0.9919     | dev: **1.0**<br />test:0.9844     |
   | 0.1          | dev: 0.7128<br /> test:**0.7155** | dev: 0.7606<br /> test:0.7582        | dev: 0.9924<br />test:**0.9928** | dev: 0.9958<br />test:0.9688      |
   | 0.2          | dev: 0.6361<br /> test:0.6151     | dev: 0.7581<br /> test:0.7575        | dev: 0.9941<br />test:0.9897     | dev: 0.9916<br /> test:**0.9916** |

   从实验结果可以看到，采用该策略后，模型在各个数据集中的性能都获得了提升，尤其在相似度小规模大的数据集中，模型在验证集上的结果从0.7272提升至0.7637。

   值得注意的是dogcat 1/10在cut_fraction=0.01时会出现异常结果，这是由于dogcat 1/10的训练集大小为1803，在总batch size=80的时候，总迭代步数为22，当设置cut_fraction=0.01时，由第二章STLR计算公式可得，cut=0，这会导致后续计算p时会出现除数为0的问题，导致实验结果异常。因此对于小规模数据集，建议设置较小的batch size。

4. Discriminative fine-tuning策略实验与分析

   本小节对Discriminative fine-tuning策略进行实验分析。理论上，预训练得到的模型的底层学到的是图像当中的通用特征，那么对于相似度大的数据集，可以保留更多的预训练参数；而对于相似度小的数据集，则应保留更少的预训练参数，让模型在任务数据集中得到更多训练。本实验通过设置不同的dis_blocks来控制底层的学习率衰减次数，dis_blocks越大则底层的学习率衰减越多，预训练参数保留得越多。实验结果如下表所示：

   | dis_blocks | indoor67<br />相似度小规模小        | food101<br />相似度小规模大         | dogcat <br />相似度大规模大       | dogcat 1/10<br />相似度大规模小   |
   | :----------- | :---------------------------------- | :---------------------------------- | :-------------------------------- | :-------------------------------- |
   | 0 (baseline)           | dev:  0.6907<br />test:0.6741       | dev: 0.7272<br />test:0.7338        | dev:0.9893<br />test:0.9830       | dev:**1.0**<br />test:0.9719      |
   | 3            | dev:**0.7842**<br />test:**0.7575** | dev:**0.7581**<br />test:**0.7527** | dev:**0.9933**<br /> test:0.9897  | dev:0.9958<br /> test:**0.9802**  |
   | 5            | dev: 0.7092<br />test:0.6961        | dev:0.7336<br /> test:0.7390        | dev: 0.9928<br /> test:**0.9910** | dev: 0.9958<br /> test:**0.9802** |

   观察实验结果，可以发现该策略在四类数据集中均有良好的效果。在相似度小规模小的数据集中，当dis_blocks设置为3时，实验效果提升明显，但设置为5的时候，test成绩反而不如Baseline，对于相似度小规模小的数据集，调优过程应当注意寻找合适的超参数设置。对于相似度大规模大的数据集无论是否采用该策略均有优良的表现，采用该策略可以略微提升模型性能。

5. Gradual unfreezing策略实验与分析

   本小节验证Gradual unfreezing策略的实验效果。理论上，该策略与Discriminative fine-tuning策略相似，对于相似度大的数据集可以更慢的解冻底层，而相似度小的数据集可以早点解冻底层使它们拟合任务数据集。本次实验设置了不同的frz_blocks控制底层的解冻速度，每个epoch模型解冻一个block。实验结果如下表所示：

   | frz_blocks | epoch | indoor67<br />相似度小规模小        | food101<br />相似度小规模大      | dogcat <br />相似度大规模大      | dogcat 1/10<br />相似度大规模小   |
   | :--------- | ----- | :---------------------------------- | :------------------------------- | :------------------------------- | :-------------------------------- |
   | 0          | 1     | dev:  0.6907<br />test:0.6741       | dev: **0.7272**<br />test:0.7338 | dev:0.9893<br />test:0.9830      | dev:**1.0**<br />test:0.9719      |
   | 0          | 3     | dev:0.7697<br /> test:0.7574        | dev: 0.7427<br /> test:0.7407    | dev:0.9919<br /> test:**0.9910** | dev:**1.0**<br />test:0.9719      |
   | 3          | 1     | dev:0.7210<br /> test:0.7018        | dev: 0.7251<br />test:0.7270     | dev:**0.9924**<br /> test:0.9857 | dev:**1.0**<br />test:0.9719      |
   | 3          | 3     | dev: 0.7763<br /> test:0.7627       | dev:0.7346<br /> test:0.7356     | dev: 0.9849<br /> test:0.9887    | dev: **1.0**<br /> test: 0.9719   |
   | 5          | 1     | dev: 0.7236<br /> test:0.6961       | dev: 0.7168<br /> test:0.7204    | dev: 0.9892<br /> test:0.9861    | dev: **1.0**<br />test:**0.9802** |
   | 5          | 3     | dev:**0.7802**<br />test:**0.7689** | dev:**0.7461<br />** test:0.7389 | dev:**0.9924**<br /> test:0.9892 | dev:**1.0**<br /> test:**0.9802** |

   从表中结果可得，frz_blocks=5的实验组在epoch=3时超越了frz_blocks=3的实验结果，设置较大的frz_blocks会降低模型的拟合能力，但提升epoch，使模型得到充分的训练后，模型能够取得更好的效果。建议用户采用Gradual unfreezing策略时设置足够大的epoch。

## 五、总结

本文描述了在NLP、CV任务中使用ULMFiT策略微调PaddleHub预训练模型的过程，尝试了warm up + linear decay, slanted triangular learning rate, Discriminative fine-tuning, Gradual unfreezing四种微调策略。

slanted triangular learning rate和warm up + linear decay在原理上和实验结果上都是相似的，Discriminative fine-tuning和Gradual unfreezing微调策略在使用中，应当注意它们会降低模型的拟合能力，可以适当提高训练的轮数。

PaddleHub 1.2已发布AutoDL Finetuner，可以自动搜索超参设置，详情请参考[PaddleHub AutoDL Finetuner](./autofinetune.md)。如有任何疑问欢迎您在issues中向我们提出！

## 六、参考文献

1. Howard J, Ruder S. Universal Language Model Fine-tuning for Text Classification[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 328-339.
2. website: [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
