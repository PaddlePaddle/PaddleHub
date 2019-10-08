# PaddleHub 迁移学习与ULMFiT微调策略

## 一、简介

迁移学习（Transfer Learning）顾名思义就是将模型在其它领域学到的知识迁移至目标领域的学习过程中，帮助模型取得更好的学习效果，讲究的是“他山之石，可以攻玉”。通过迁移学习，模型能够在短时间内取得更好的学习效果。

迁移学习通常由预训练阶段、微调阶段两部分组成。预训练阶段通常在超大规模数据集中进行，例如CV任务中的ImageNet包含千万张标注图片，NLP任务中的English Wikipedia包含25亿个单词，这样训练得到的预训练模型能够很好地学习到不同领域中的通用知识，具有很好的泛化能力。但预训练阶段使用的数据集往往与我们想要完成的任务的数据集存在差异，例如如果你只是想简单地判断一副图像是否是玫瑰花，ImageNet就没有提供相关的标注，因此为了更好的学习目标领域的知识，通常还需要对预训练模型参数进行微调。微调过程中，我们应该同时兼顾模型的拟合能力与泛化能力，控制好学习率，既不能太小使模型学习不充分，也不能太大丢失了过多的预训练通用知识。

PaddleHub中集成了ERNIE、BERT、LAC、ELMo等[NLP预训练模型](https://www.paddlepaddle.org.cn/hub)，ResNet、GoogLeNet、MobileNet等[CV预训练模型](https://www.paddlepaddle.org.cn/hub)；以及Adam + Weight Decay、L2SP、ULMFiT等微调策略。本文主要介绍ULMFiT微调策略在PaddleHub中的实验结果及其思考分析。

## 二、 ULMFiT

[ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)提出了三种微调策略：slanted triangular learning rates、discriminative fine-tuning以及gradual unfreezing。

1. slanted triangular learning rates

   slanted triangular learning rates（STLR）是一种学习率先上升再下降的微调策略，如下图所示：

   ![image-20190917170542047](https://user-images.githubusercontent.com/11913168/65138331-61316c80-da3d-11e9-9acb-c29385785e24.png)

   其计算公式如下：

   ![image-20190917170707549](https://user-images.githubusercontent.com/11913168/65138349-6bec0180-da3d-11e9-821d-98bc7f2d6f1e.png)

   其中T表示训练的迭代次数，PaddleHub会自动计算训练总step数；cut_frac是学习率上升在整个训练过程占用的比例；cut表示学习率转折处的step；t表示当前的step；p表示当前step学习率的缩放比例；ratio表示LR最低下降至最大学习率η<sub>max</sub>的几分之一；η<sub>t</sub>表示当前step的学习率。论文中，作者采用的超参设置为：cut_frac=0.1, ratio=32, η<sub>max</sub>=0.01。本文实验部分保持ratio=32, η<sub>max</sub>=0.01不变，仅调整cut_frac。

2. Discriminative fine-tuning

   Discriminative fine-tuning 是一种学习率逐层递减的策略，通过该策略可以减缓底层的更新速度。其计算公式为：

   <div align=center>η<sup>l-1</sup>=η<sup>l</sup>/factor</div>
其中η<sup>l</sup>表示第l层的学习率；η<sup>l-1</sup>表示第l-1层的学习率；factor表示逐层衰减率，论文中作者根据经验设置为2.6。这个策略能够让模型微调过程中不断减缓底层的更新速度，尽可能的保留预训练模型中习得的底层通用知识。在PaddleHub中，我们针对这个策略还提供了另外一个超参：dis_blocks。由于预训练模型中没有记录op的层数，Paddlehub通过op的前后关系推测op所在的层次，这会导致诸如LSTM这类计算单元的op会被当作是不同层的op。为了不使层次划分太细，我们将层次进行了分块，用块的概念代替原论文中层的概念，通过设置dis_blocks即可设置块的个数。默认为3，如果设置为0，则不采用Discriminative fine-tuning。

3. Gradual unfreezing

   Gradual unfreezing是一种逐层解冻的策略，通过该策略可以优先更新上层，再慢慢解冻下层参与更新。在PaddleHub中，我们仍然引入了一个额外的超参：frz_blocks，其概念与上一小节提到的dis_blocks一致，在微调过程中，每经过一个epoch，模型解冻一个block，所有未被冻结的block都会参与到模型的参数更新中。

本文接下来将对ULMFiT策略在NLP以及CV任务中的使用进行实验说明，由于slanted triangular learning rates与warmup + linear decay在原理上高度相似，本文也将对比slanted triangular learning rates与warmup + linear decay的实验效果。

## 三、 NLP实验与分析

  本章将介绍ULMFiT策略在NLP任务中的使用。

1. 数据集与预训练模型的选择

   为探寻ULMFiT策略的一般性规律，本章选取两个数据集，一个为中文数据集Chnsenticorp，另一个为英文数据集CoLA，两个数据集的训练集数据规模相似，前者包含9601个句子，后者包含8551个句子。针对中文数据集Chnsenticorp，为了凸现实验效果对比，本章未选择ERNIE，而是选择Elmo作为其预训练模型。针对英文数据集CoLA，本章选择“bert_uncased_L-12_H-768_A-12”作为其预训练模型。

2. Baseline与实验设置

   Baseline不采用任何策略，学习率在微调过程中保持恒定。Chnsenticorp与CoLA任务均在单卡运行（只使用1张显卡），Batch size均设置为32，总共迭代3个epoch。Chnsenticorp设置学习率为1e-4，由于采用ELMO预训练模型，无需设置句子最大长度；CoLA设置学习率为5e-5，句子最大长度设置为128。实验效果如下表所示：

   | -             | Chnsenticorp | CoLA              |
   | :------------ | :----------- | :---------------- |
   | Module        | Elmo         | Bert              |
   | Batch size    | 32           | 32                |
   | Num epoch     | 3            | 3                 |
   | Learning rate | 1e-4         | 5e-5              |
   | Max length    | -            | 128               |
   | Dev           | acc = 0.8766 | matthews = 0.5680 |
   | Test          | acc = 0.8733 | -                 |

   其中Chnsenticorp汇报准确率（accuracy）得分，CoLA汇报马修斯相关系数（matthews correlation coefficient）得分。由于CoLA未公开测试集，且其在线测评每日仅限提交3次，本章只汇报其验证集上的得分。从实验结果中，可以看到模型在Chnsenticorp数据集的泛化能力较好，在CoLA数据集的泛化能力较弱。

   在下文中，如果没有特别说明，实验设置（例如Batch size、Num epoch等）均与Baseline一致。

3. warmup + linear decay策略实验与分析

   为了与slanted triangular learning rates进行结果对比，本小节对warmup + linear decay策略进行了实验。在本实验中，warm up proportion为学习率上升在总步数中的比重，linear decay则在warmup结束的时刻开始，学习率在训练结束时下降至0。实验结果如下表所示，其中warm up proportion=0为baseline：

   | warm up proportion | 0（baseline） | 0.1    | 0.2        |
   | :----------------- | :------------ | :----- | :--------- |
   | Chnsenticorp dev   | **0.8766**    | 0.8725 | 0.8758     |
   | Chnsenticorp test  | **0.8733**    | 0.8700 | 0.8691     |
   | CoLA dev           | 0.5680        | 0.5780 | **0.5786** |

   理论上该策略影响模型的拟合能力，从实验结果可以看到该策略在Chnsenticorp数据集中产生很小的副作用，而在CoLA数据集中该策略能够减缓预训练模型参数的更新速度，提升模型泛化能力。

4. slanted triangular learning rates（STLR）策略实验与分析

   在本小节中，我们将进一步设置warmup + linear decay的超参数以更好地与STLR策略比较。从第二章的理论介绍中，我们可以看到STLR在原理上与warmup + linear decay十分相似，较为明显的区别在于STLR的最终速度为1/ratio，在本文中ratio一概为32。因此本小节进一步设置了linear decay最终速度为learning rate/32的实验组。STLR的超参数仅调整cut_fraction为0.1、0.2，其它超参与论文设置一致。实验结果如下表所示：

   | warm up proportion                  | 0（Baseline） | 0.1    | 0.2    | 0.1        | 0.2    | 0       | 0       |
   | :---------------------------------- | :------------ | :----- | :----- | :--------- | :----- | :------ | :------ |
   | linear decay end                    | - Unused      | 0      | 0      | lr/32      | lr/32  | -       | -       |
   | **slanted triangular cut_fraction** | **0**         | **0**  | **0**  | **0**      | **0**  | **0.1** | **0.2** |
   | Chnsenticorp dev                    | 0.8766        | 0.8725 | 0.8758 | **0.8825** | 0.8733 | 0.8791  | 0.8791  |
   | Chnsenticorp test                   | **0.8733**    | 0.8700 | 0.8691 | 0.8666     | 0.8616 | 0.8691  | 0.8716  |
   | CoLA dev                            | 0.5680        | 0.5780 | 0.5786 | **0.5887** | 0.5826 | 0.5880  | 0.5827  |

   由实验结果可以看到STLR无论是原理上还是实验效果上都与warmup + linear decay (end learning rate = lr/32) 相差无几，在CoLA数据集中均有增益效果，而在Chnsenticorp数据集中依然产生了副作用。而对比warmup + linear decay (end learning rate = lr/32) 与warmup + linear decay (end learning rate = 0) 可以看到在模型微调接近结束时，保持较小的学习率对模型性能的提升可能仍有帮助，但这会引入新的超参数end_learning_rate。

   基于上述实验结果，我们建议用户尝试slanted triangular或者warmup+linear decay的策略时尝试更多的超参设置寻找该策略的Bias-Variance Tradeoff平衡点，合理的超参设置能够提升模型性能。

5. Discriminative fine-tuning策略实验与分析

   本小节对Discriminative fine-tuning策略进行实验分析。固定训练总epoch=3，实验结果如下表所示：

   | dis_blocks        | -<br />（Baseline） | 3          | 5      |
   | ----------------- | ------------------- | ---------- | ------ |
   | epoch             | 3                   | 3          | 3      |
   | Chnsenticorp dev  | **0.8766**          | 0.8641     | 0.6766 |
   | Chnsenticorp test | **0.8733**          | 0.8683     | 0.7175 |
   | CoLA dev          | 0.5680              | **0.5996** | 0.5749 |

   从实验结果中，可以看到由于Discriminative fine-tuning策略降低了模型底层的更新速度，该策略会抑制拟合能力，dis_blocks设置越大，模型的拟合能力越弱。为了提升模型的拟合能力，本小节继续增大epoch大小至5、8。
   
   对于Chnsenticorp，实验结果如下表所示：

   | dis_blocks        | -<br />（Baseline） | -          | 5      | -          | 5          |
   | ----------------- | ------------------- | ---------- | ------ | ---------- | ---------- |
   | epoch             | 3                   | 5          | 5      | 8          | 8          |
   | Chnsenticorp dev  | 0.8766              | 0.8775     | 0.8566 | 0.8775     | **0.8792** |
   | Chnsenticorp test | 0.8733              | **0.8841** | 0.8400 | **0.8841** | 0.8625     |

   可以看到当dis_blocks=5时，epoch=8时，模型在dev上的得分超越Baseline（epoch=3），但test中的得分则始终低于相同epoch、不采用Discriminative fine-tuning策略的实验结果。

   由于CoLA在dis_block=3，epoch=3时的成绩已经超越了Baseline，因为我们可以进一步增大dis_blocks，观察其实验效果，其结果如下表所示：

   | dis_blocks | -<br />（Baseline） | 3          | -      | 7      | -      | 7      |
   | ---------- | ------------------- | ---------- | ------ | ------ | ------ | ------ |
   | epoch      | 3                   | 3          | 5      | 5      | 8      | 8      |
   | CoLA dev   | 0.5680              | **0.5996** | 0.5680 | 0.5605 | 0.5720 | 0.5788 |

   实验结果表明，dis_blocks过大同样会导致模型在CoLA数据集欠拟合的问题，当dis_blocks=7时，模型在epoch=5性能低于Baseline (epoch=3)，直至epoch=8才略微超过Baseline (epoch=8)，但仍显著低于dis_blocks=3，epoch=3的模型表现。

   因此我们建议用户采用discriminative fine-tuning时，应当设置较小的dis_blocks，如果设置过大的dis_blocks，则需提升训练的epoch。

6. Gradual unfreezing策略实验与分析

   本小节对Gradual unfreezing策略进行实验分析，frz_blocks设置为3，第一个epoch只更新最顶层的block，此后每一个epoch解冻一个block参与更新，实验结果如下表所示：

   | gradual unfreezing | -（baseline） | 3      |
   | :----------------- | :------------ | :----- |
   | Chnsenticorp dev   | 0.8766        | 0.8850 |
   | Chnsenticorp test  | 0.8733        | 0.8816 |
   | CoLA dev           | 0.5680        | 0.5704 |

   实验结果表明通过延后更新预训练模型中的底层参数，该策略不论是对Chnsenticorp数据集还是对CoLA数据集均有效，我们建议用户采用这种策略。

## 四、CV实验与分析

1. 数据集与预训练模型的选择

   本小节采用resnet50作为预训练模型。在NLP任务中，我们仅对数据集进行了数据规模划分，这是因为NLP任务的预训练数据集通常是无标签数据集，它的训练过程是无监督学习，与微调过程的有监督学习差异较大，不好衡量微调数据集与预训练数据集的相似性。而resnet50预训练数据集ImageNet是带标签数据集，它的预训练过程是有监督学习，因此我们除了可以划分数据规模，还可以划分数据集与预训练数据集的相似度。基于此，我们选择了四个数据集用于实验：

   - **indoor67**（相似度小，规模小）：该数据集包含1.5万个样例，标签包含concert_hall, locker_room等67个室内物体，由于ImageNet中没有这些室内物品标签，我们认为它与预训练数据集相似度较小。理论上，相似度小规模小的数据集在微调时既不能更新得太快导致过拟合，也不能太小导致欠拟合，是最难调整的一类数据集。
   - **food101**（相似度小，规模大）：该数据集标签包含Apple pie, Baby back ribs等101个食品，这些标签同样几乎没有出现在ImageNet中，因此它与预训练数据集的相似度也较小。同时该数据集训练集包含10万个样例，远远大于indoor67的1.5万个样例，因此我们将indoor67归类为规模小，food101归类为规模大。理论上，相似度小规模大的数据集可以较快更新，是模型充分拟合。
   - **dogcat**（相似度大，规模大）：该数据集包含2.2万个样例，数据量没有比indoor67大很多，但标签只有dog和cat两类，数据规模会比indoor67充裕很多。猫狗标签在ImageNet中频繁出现且在ImageNet中划分的品种更加细致，因此它可以被认为是与预训练数据集相似度大的数据集。理论上，相似度大规模大的数据集不论采用怎样的策略均能获得非常良好的模型性能，通过细致地调整策略可以略微提升模型性能。
   - **dogcat 1/10**（相似度大，规模小）：为了构建相似度大，规模小的数据集，我们在dogcat数据集的基础上随机抽取了1/10个样例组成新的数据集，该数据集包含0.22万个样例，”dog”/”cat”两类标签。理论上，相似度大规模小的数据集由于与预训练数据集相似度大，模型在微调过程中可以保留更多的预训练参数。

2. Baseline与实验设置

   Baseline不采用任何策略，学习率在微调过程中保持恒定。所有任务均采用双卡运行（使用两张显卡），Batch size设置为40，总Batch size即为80，训练迭代一个epoch，学习率均设置为1e-4，评估指标为准确率（accuracy）。实验效果如下表所示：

   | -                | **indoor67**   | **food101**    | **dogcat**     | **dogcat 1/10** |
   | :--------------- | :------------- | :------------- | -------------- | --------------- |
   | Type             | 相似度小规模小 | 相似度小规模大 | 相似度大规模大 | 相似度大规模小  |
   | Module           | resnet50       | resnet50       | resnet50       | resnet50        |
   | Total Batch size | 80             | 80             | 80             | 80              |
   | Num epoch        | 1              | 1              | 1              | 1               |
   | Learning rate    | 1e-4           | 1e-4           | 1e-4           | 1e-4            |
   | Dev              | 0.6907         | 0.7272         | 0.9893         | 1.0             |
   | Test             | 0.6741         | 0.7338         | 0.9830         | 0.9719          |

   在下文中，如果没有特别说明，实验设置（例如Batch size、Num epoch等）均与Baseline一致。

3. warmup + linear decay策略实验与分析

   在本实验中，wup(warm up proportion)为学习率上升在总步数中的比重，linear decay则在warmup结束的时刻开始，学习率在训练结束时下降至0。实验结果如下表所示，其中warm up proportion=0为Baseline，它不采用linear decay：

   | wup  | indoor67<br />相似度小规模小          | food101<br />相似度小规模大          | dogcat<br />相似度大规模大        | dogcat 1/10<br />相似度大规模小    |
   | ---- | ------------------------------------- | ------------------------------------ | --------------------------------- | ---------------------------------- |
   | 0    | dev:  **0.6907**<br />test:**0.6741** | dev: 0.7272<br />test:0.7338         | dev:0.9893<br />test:0.9830       | <br />dev:**1.0**<br />test:0.9719 |
   | 0.1  | dev: 0.6506 <br />test:0.6324         | dev:**0.7573** <br />test:**0.7497** | dev:  **0.9964**<br />test:0.9924 | dev:0.9958 <br />test:0.9802       |
   | 0.2  | dev:0.6282 <br />test:0.6372          | dev: 0.7486 <br />test:0.7446        | dev: 0.9937 <br />test:**0.9937** | dev:0.9916 <br />test:**0.9813**   |

   从实验结果可以看到该策略在相似度小规模小的数据集不适合采用warm up + linear decay策略，由于数据集相似度小，设置warm up + linear decay策略会导致模型严重欠拟合。而在相似度大规模小的数据集中抑制拟合能力能够提升模型的泛化能力，提升test得分。在规模大的数据集中，相似度小的数据集提升尤其明显；而相似度大的数据集也有略微提升。

   上面的实验表明，设置合理的warm up proportion在Bias-Variance Tradeoff中找到平衡是有利于各类数据集中模型性能提升的。为此，我们进一步探索了更细致的warm up proportion设置。实验结果如下表所示：

   | wup  | indoor67<br />相似度小规模小          | food101<br />相似度小规模大          | dogcat<br />相似度大规模大        | dogcat 1/10<br />相似度大规模小    |
   | ---- | ------------------------------------- | ------------------------------------ | --------------------------------- | ---------------------------------- |
   | 0    | dev:  **0.6907**<br />test:**0.6741** | dev: 0.7272<br />test:0.7338         | dev:0.9893<br />test:0.9830       | <br />dev:**1.0**<br />test:0.9719 |
   | 0.01 | dev:0.6526<br />test:0.6611           | dev: **0.7564**<br />test:**0.7584** | dev:0.9955<br />test:0.9915       | <br />dev:0.9958<br />test:0.9730  |
   | 0.05 | dev:0.6493  <br />test:0.6397         | dev: 0.7544 <br />test:0.7573        | dev: 0.9950 <br />test:**0.9946** | dev:0.9958  <br />test:**0.9844**  |
   | 0.1  | dev: 0.6506 <br />test:0.6324         | dev:0.7573 <br />test:0.7497         | dev:  **0.9964**<br />test:0.9924 | dev:0.9958 <br />test:0.9802       |
   | 0.2  | dev:0.6282 <br />test:0.6372          | dev: 0.7486 <br />test:0.7446        | dev: 0.9937 <br />test:0.9937     | dev:0.9916 <br />test:0.9813       |

   从以上实验分析，food101、dogcat、dogcat 1/10设置更小的warm up proportion (<0.1)均在测试集上取得了更好的成绩，对于dogcat 1/10，它的训练集大小为1803，在总batch size=80的时候，总迭代步数为22，当设置wup=0.01时，实际上并没有warm up阶段，只有linear decay阶段。我们建议用户尝试设置较小的warm up proportion或者仅采用linear decay，观察模型的性能变化。

   对于相似度小规模小的数据集，我们进一步探究它的warm up proportion设置，由于indoor67训练集大小为12502，在总batch size=80时，总的迭代步数为156，在warm up proportion=0.01时，warmup阶段实际只有1步，我们无法再缩小warm up proportion了，因此我们尝试设置更小的batch size。在此次实验中，我们设置batch size为16，并只使用单张卡，实验结果如下表所示：

   | wup  | 0      | 0.01       | 0.05       |
   | ---- | ------ | ---------- | ---------- |
   | dev  | 0.6868 | **0.7448** | 0.7371     |
   | test | 0.6751 | 0.7237     | **0.7316** |

   在这里，wup=0的实验组为baseline，未采用linear decay，其结果与总batch size=80的结果相差不大。从这组实验我们发现在设置batch size=16，wup=0.05时，在相似度小规模小数据集中采用warm up + linear decay策略的效果终于超越了Baseline。对于小规模数据集，我们建议设置较小的batch size，再通过尝试设置较小的warm up proportion寻找Bias-Variance Tradeoff中的平衡点。

4. slanted triangular learning rates（STLR）策略实验与分析

   在本小节中，我们对STLR的效果进行实验，实验结果如下表所示，cut_fraction=0为Baseline，不采用任何策略：

   | cut_fraction | indoor67<br />相似度小规模小      | food101<br />相似度小规模大          | dogcat <br />相似度大规模大      | dogcat 1/10<br />相似度大规模小   |
   | :----------- | :-------------------------------- | :----------------------------------- | :------------------------------- | :-------------------------------- |
   | 0            | dev:  0.6907<br />test:0.6741     | dev: 0.7272<br />test:0.7338         | dev:0.9893<br />test:0.9830      | dev:**1.0**<br />test:0.9719      |
   | 0.01         | dev: 0.7148<br /> test:0.7053     | dev:**0.7637**<br /> test:**0.7656** | dev: **0.9946**<br />test:0.9924 | dev: *0.4481* <br />test:*0.5346* |
   | 0.05         | dev: **0.7226**<br /> test:0.7130 | dev:0.7605<br /> test:0.7612         | dev: 0.9901<br />test:0.9919     | dev: **1.0**<br />test:0.9844     |
   | 0.1          | dev: 0.7128<br /> test:**0.7155** | dev: 0.7606<br /> test:0.7582        | dev: 0.9924<br />test:**0.9928** | dev: 0.9958<br />test:0.9688      |
   | 0.2          | dev: 0.6361<br /> test:0.6151     | dev: 0.7581<br /> test:0.7575        | dev: 0.9941<br />test:0.9897     | dev: 0.9916<br /> test:**0.9916** |

   除了indoor67，其余数据集中的实验结果与第3小节中的实验结果差异不大。在第二章NLP部分，我们已经讨论过STLR与第3小节的实验设置的明显差别仅在于最终速度为1/ratio，保留一定终速度有利于缓解相似度小规模小数据集的欠拟合问题。我们建议用户采用较小的cut_fraction，该策略在相似度小规模大的数据集终有较为显著的效果。

   值得注意的是dogcat 1/10在cut_fraction=0.01时会出现异常结果，这是由于dogcat 1/10的训练集大小为1803，在总batch size=80的时候，总迭代步数为22，当设置cut_fraction=0.01时，由第二章STLR计算公式可得，cut=0，这会导致后续计算p时会出现除数为0的问题，导致无法微调，最终汇报的成绩为未经微调的模型直接进行测试的结果。

5. Discriminative fine-tuning策略实验与分析

   本小节对Discriminative fine-tuning策略进行实验分析。理论上，预训练得到的模型的底层学到的是图像当中的通用特征，那么对于相似度大的数据集，我们可以保留更多的预训练参数；而对于相似度小的数据集，我们则应该保留更少的预训练参数，让模型在任务数据集中得到更多训练。我们通过设置不同的dis_blocks来控制底层的学习率衰减次数，dis_blocks越大则底层的学习率衰减越多，预训练参数保留得越多。实验结果如下表所示：

   | cut_fraction | indoor67<br />相似度小规模小        | food101<br />相似度小规模大         | dogcat <br />相似度大规模大       | dogcat 1/10<br />相似度大规模小   |
   | :----------- | :---------------------------------- | :---------------------------------- | :-------------------------------- | :-------------------------------- |
   | 0            | dev:  0.6907<br />test:0.6741       | dev: 0.7272<br />test:0.7338        | dev:0.9893<br />test:0.9830       | dev:**1.0**<br />test:0.9719      |
   | 3            | dev:**0.7842**<br />test:**0.7575** | dev:**0.7581**<br />test:**0.7527** | dev:**0.9933**<br /> test:0.9897  | dev:0.9958<br /> test:**0.9802**  |
   | 5            | dev: 0.7092<br />test:0.6961        | dev:0.7336<br /> test:0.7390        | dev: 0.9928<br /> test:**0.9910** | dev: 0.9958<br /> test:**0.9802** |

   观察实验结果，可以发现相似度小规模小的数据集当dis_blocks设置为3时，实验效果大幅度提升，但设置为5的时候，test成绩反而如Baseline，对于相似度小规模小的数据集我们仍应注意寻找合适的超参数。对于相似度小规模大的数据集，模型可以在任务训练集中得到充分的学习，设置较小的dis_blocks适当减缓底层参数的更新有助于模型性能提升。对于相似度大规模大的数据集无论是否采用该策略均有优良的表现，采用该策略可以小幅提升模型性能。而对于相似度大规模小的数据集可以设置更大的dis_blocks，保留更多的底层参数，提升它的泛化能力。

6. Gradual unfreezing策略实验与分析

   本小节验证Gradual unfreezing策略的实验效果。理论上，该策略与Discriminative fine-tuning策略相似，对于相似度大的数据集可以更慢的解冻底层，而相似度小的数据集可以早点解冻底层使它们拟合任务数据集。我们设置了不同的frz_blocks控制底层的解冻速度，每个epoch模型解冻一个block。实验结果如下表所示：

   | frz_blocks | indoor67<br />相似度小规模小      | food101<br />相似度小规模大       | dogcat <br />相似度大规模大       | dogcat 1/10<br />相似度大规模小   |
   | :--------- | :-------------------------------- | :-------------------------------- | :-------------------------------- | :-------------------------------- |
   | 0          | dev:  0.6907<br />test:0.6741     | dev: **0.7272**<br />test:0.7338  | dev:0.9893<br />test:0.9830       | dev:**1.0**<br />test:0.9719      |
   | 3          | dev:0.7210<br />test:**0.7018**   | dev: 0.7251<br /> test:**0.7270** | dev:**0.9924**<br /> test:0.9857  | dev:**1.0**<br />test:0.9719      |
   | 5          | dev: **0.7236**<br /> test:0.6961 | dev: 0.7168<br /> test:0.7204     | dev: 0.9892<br /> test:**0.9861** | dev: **1.0**<br />test:**0.9802** |

   实验结果可得，相似度小规模小依然要注意超参数的设置，frz_blocks=3时模型性能提升，设置为5时模型性能下降；与Discriminative fine-tuning结论一致，相似度小规模大的数据集可以设置较小的frz_blocks，相似度大的数据集可以设置较大的frz_blocks。
   由于上述实验num epoch=1，实际上模型只更新了最顶层的block，在这里我们提高num epoch，使逐层解冻策略真正运用起来，实验结果如下表所示：

   | frz_blocks | epoch | indoor67<br />相似度小规模小        | food101<br />相似度小规模大      | dogcat <br />相似度大规模大      | dogcat 1/10<br />相似度大规模小   |
   | :--------- | ----- | :---------------------------------- | :------------------------------- | :------------------------------- | :-------------------------------- |
   | 0          | 1     | dev:  0.6907<br />test:0.6741       | dev: **0.7272**<br />test:0.7338 | dev:0.9893<br />test:0.9830      | dev:**1.0**<br />test:0.9719      |
   | 0          | 3     | dev:0.7697<br /> test:0.7574        | dev: 0.7427<br /> test:0.7407    | dev:0.9919<br /> test:**0.9910** | dev:**1.0**<br />test:0.9719      |
   | 3          | 1     | dev:0.7210<br /> test:0.7018        | dev: 0.7251<br />test:0.7270     | dev:**0.9924**<br /> test:0.9857 | dev:**1.0**<br />test:0.9719      |
   | 3          | 3     | dev: 0.7763<br /> test:0.7627       | dev:0.7346<br /> test:0.7356     | dev: 0.9849<br /> test:0.9887    | dev: **1.0**<br /> test: 0.9719   |
   | 5          | 1     | dev: 0.7236<br /> test:0.6961       | dev: 0.7168<br /> test:0.7204    | dev: 0.9892<br /> test:0.9861    | dev: **1.0**<br />test:**0.9802** |
   | 5          | 3     | dev:**0.7802**<br />test:**0.7689** | dev:**0.7461<br />** test:0.7389 | dev:**0.9924**<br /> test:0.9892 | dev:**1.0**<br /> test:**0.9802** |

   从表中结果可得，提升epoch，进行逐层解冻后，frz_blocks=5的实验组在epoch=3时超越了frz_blocks=3的实验结果，设置较大的frz_blocks会降低模型的拟合能力，但提升epoch，使模型得到充分的训练后，模型能够取得比不采用该策略更好的效果。我们建议用户采用Gradual unfreezing策略并设置合适的frz_blocks和足够的epoch。

## 五、总结

本文详细描述了使用ULMFiT策略微调PaddleHub预训练模型的来龙去脉。对于NLP任务，我们选取了两个规模相当的数据集；对于CV任务，我们划分了相似度小规模小、相似度小规模大、相似度大规模大、相似度大规模小四类数据集。我们实验了warm up + linear decay, slanted triangular learning rate, Discriminative fine-tuning, Gradual unfreezing四种策略。

warm up + linear decay和slanted triangular learning rate在原理上和实验结果上都是相似的，用户可以任选其中一种策略并寻找它的Bias-Variance Tradeoff平衡点。在采用Discriminative fine-tuning和Gradual unfreezing策略时，应当注意它们会降低模型的拟合能力，如果任务数据集与预训练数据集的相似度较大可以设置较大的超参，否则应该设置较小的超参；如果模型的拟合能力下降严重，可以适当提高训练的轮数。欢迎您使用上述策略调试您的PaddleHub预训练模型，同时我们欢迎您使用[PaddleHub Auto Fine-tune](https://github.com/PaddlePaddle/PaddleHub/blob/develop/tutorial/autofinetune.md)自动搜索超参设置，如有任何疑问请在issues中向我们提出！

## 六、参考文献

1. Howard J, Ruder S. Universal language model fine-tuning for text classification[J]. arXiv preprint arXiv:1801.06146, 2018.MLA
2. website: [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
