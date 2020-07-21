**若您想在自定义数据集上完成Fine-tune，请查看[PaddleHub适配自定义数据完成Fine-tune](../tutorial/how_to_load_data.md)**

## hub.dataset

### Class `hub.dataset.ChnSentiCorp`

ChnSentiCorp 是中文情感分析数据集，其目标是判断一段文本的情感态度。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.ChnSentiCorp()
```

数据集样例
```text
label   text_a
1       选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，>不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般
1       15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错
0       房间太小。其他的都一般。。。。。。。。。
...
```
以上类别“0”表示反对态度，“1”表示支持态度。每个字段以tab键分隔。

### Class `hub.dataset.LCQMC`

LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问答匹配中文数据集，其目标是判断两个问题的语义是否相同。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.LCQMC()
```

数据集样例
```text
text_a    text_b    label
喜欢打篮球的男生喜欢什么样的女生        爱打篮球的男生喜欢什么样的女生  1
我手机丢了，我想换个手机        我想买个新手机，求推荐  1
大家觉得她好看吗        大家觉得跑男好看吗？    0
...
```

以上类别“0”表示语义相同，“1”表示语义相反。每个字段以tab键分隔。

### Class `hub.dataset.NLPCC_DPQA`

NLPCC_DPQA 是由国际自然语言处理和中文计算会议NLPCC于2016年举办的评测任务，其目标是选择能够回答问题的答案。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.NLPCC_DPQA()
```

数据集样例
```text
qid    text_a    text_b    label
0       黑缘粗角肖叶甲触角有多大？      触角近于体长之半，第1节粗大，棒状，第2节短，椭圆形，3、4两节细长，稍短于第5节，第5节基细端粗，末端6节明显粗大。 1
0       黑缘粗角肖叶甲触角有多大？      前胸前侧片前缘直；前胸后侧片具粗大刻点。        0
0       黑缘粗角肖叶甲触角有多大？      足粗壮；胫节具纵脊，外端角向外延伸，呈弯角状；爪具附齿。        0
1       暮光闪闪的姐姐是谁？    暮光闪闪是一匹雌性独角兽，后来在神秘魔法的影响下变成了空角兽（公主），她是《我的小马驹：友情是魔法》（英文名：My Little Pony：Friendship is Magic）中的主角之一。       0
1       暮光闪闪的姐姐是谁？     她是银甲闪闪（Shining Armor）的妹妹，同时也是韵律公主（Princess Cadance）的小姑子。    1
...
```

以上qid表示问题的序号，类别“0”表示相应问题的错误答案，类别“1”表示相应问题的正确答案。每个字段以tab键分隔。

### Class `hub.dataset.MSRA_NER`

MSRA-NER(SIGHAN 2006) 数据集由微软亚研院发布，其目标是命名实体识别，是指识别中文文本中具有特定意义的实体，主要包括人名、地名、机构名等。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.MSRA-NER()
```

数据集样例
```text
text_a    label
海^B钓^B比^B赛^B地^B点^B在^B厦^B门^B与^B金^B门^B之^B间^B的^B海^B域^B。  O^BO^BO^BO^BO^BO^BO^BB-LOC^BI-LOC^BO^BB-LOC^BI-LOC^BO^BO^BO^BO^BO^BO
这^B座^B依^B山^B傍^B水^B的^B博^B物^B馆^B由^B国^B内^B一^B流^B的^B设^B计^B师^B主^B持^B设^B计^B，^B整^B个^B建^B筑^B群^B精^B美^B而^B恢^B宏^B。      O^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO
但^B作^B为^B一^B个^B共^B产^B党^B员^B、^B人^B民^B公^B仆^B，^B应^B当^B胸^B怀^B宽^B阔^B，^B真^B正^B做^B到^B“^B先^B天^B下^B之^B忧^B而^B忧^B，^B后^B天^B下^B之^B乐^B而^B乐^B”^B，^B淡^B化^B个^B人^B的^B名^B利^B得^B失^B和^B宠^B辱^B悲^B喜^B，^B把^B改^B革^B大^B业^B摆^B在^B首^B位^B，^B这^B样^B才^B能^B超^B越^B自^B我^B，^B摆^B脱
^B世^B俗^B，^B有^B所^B作^B为^B。    O^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO
...
```

以上label是针对每一个字的标签，并且一句话中的每个字以不可见字符“\002”分隔（如上述“^B”）。每个字段以tab键分隔。标注规则如下表：

| 标签 |  定义 |
| ---- | ---- |
| B-LOC | 地点的起始位置 |
| I-LOC | 地点的中间或结束位置 |
| B-PER | 人名的起始位置 |
| I-PER | 人名的中间或结束位置 |
| B-ORG | 机构名的起始位置 |
| I-ORG | 机构名的中间或者结束位置 |
| O     | 不关注的字 |

### Class `hub.dataset.Toxic`

Toxic 是英文多标签分类数据集，其目标是将一段话打上6个标签，toxic(恶意),severetoxic(穷凶极恶),obscene(猥琐),threat(恐吓),insult(侮辱),identityhate(种族歧视)，这些标签并不是互斥的。即这段话可以打上多个标签。
如

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.Toxic()
```

数据集样例
```text
id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
0000997932d777bf,"Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27",0,0,0,0,0,0
000103f0d9cfb60f,"D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",0,0,0,0,0,0
0002bcb3da6cb337,COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK,1,1,1,0,1,0
...
```
每个字段以","分隔。第一列表示样本ID，第二列表示样本文本数据，第3-8列表示相应样本是否含有对应的标签（0表示没有对应列的标签，1表示有对应列的标签）。如示例数据中的第三条数据，表示文本"COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"有标签toxic、severe_toxic、obscene和insult。

### Class `hub.dataset.SQUAD`

SQuAD 是英文阅读理解数据集，给定一个段落文本以及一个问题，其目标是在该段落中找到问题的答案位置。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.SQUAD()
```

关于该数据集详细信息可以参见[SQuAD官网介绍](https://rajpurkar.github.io/SQuAD-explorer/)

### Class `hub.dataset.GLUE`

GLUE是一个英文数据集集合，包含9项自然语言理解任务数据集：

- 文本分类任务数据集CoLA，其目标是给定一个句子，判断其语法正确性。
- 情感分析任务数据集吧SST-2，其目标是给定一个句子，判断其情感极性。
- 句子对分类任务数据集MRPC，其目标是给定两个句子，判断它们是否具有相同的语义关系。
- 回归任务数据集STS-B，其目标是给定两个句子，计算它们的语义相似性。
- 句子对分类任务数据集QQP，其目标是给定两个句子，判断它们是否具有相同的语义关系。
- 文本推理任务数据集MNLI，其目标是给定前提与假设，判断它们的逻辑关系（“矛盾“ / “中立” / “蕴含”）。该数据集又分为“匹配”与“不匹配”两个版本，“匹配”与“不匹配”指的是训练集与测试集的数据来源是否一致，是否属于相同领域风格的文本。在PaddleHub中，您可以通过“MNLI_m”和"MNLI_mm"来指定不同的版本
- 问题推理任务QNLI，其目标是给定问题，判断它的回答是否正确。
- 文本蕴含任务RTE，其目标是给定两个句子，判断它们是否具有蕴含关系。
- 文本蕴含任务WNLI，其目标是给定两个句子，判断它们是否具有蕴含关系。由于该数据集存在一些问题，我们暂时没有实现该数据集。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.GLUE(sub_dataset='SST-2')
MNLI_Matched = hub.dataset.GLUE(sub_dataset='MNLI_m')
MNLI_MisMatched = hub.dataset.GLUE(sub_dataset='MNLI_mm')
```
关于该数据集详细信息可以参见[GLUE官网介绍](https://gluebenchmark.com/)


### Class `hub.dataset.XNLI`

XNLI是一个跨语言自然语言推理数据集，其目标是给定前提与假设，判断它们的逻辑关系（“矛盾“ / “中立” / “蕴含”）。XNLI的验证集与测试集包含15种语言版本，在BERT与ERNIE中，它的训练集来自英文数据集MNLI，将其翻译至对应的语言版本即可。我们采用了相同的数据集方案，并划分了15种语言的数据集：

<table>
<thead>
</thead>
<tbody><tr>
  <td align="center">ar - Arabic</td>
  <td align="center">bg - Bulgarian</td>
  <td align="center">de - German</td>
</tr>
<tr>
  <td align="center">el - Greek</td>
  <td align="center">en - English</td>
  <td align="center">es - Spanish </td>
</tr>
<tr>
  <td align="center">fr - French </td>
  <td align="center">hi - Hindi</td>
  <td align="center">ru - Russian</td>
</tr>
<tr>
  <td align="center">sw - Swahili </td>
  <td align="center">th - Thai</td>
  <td align="center">tr - Turkish</td>
</tr>
<tr>
  <td align="center">ur - Urdu </td>
  <td align="center">vi - Vietnamese</td>
  <td align="center">zh - Chinese</td>
</tr>
</tbody></table>

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.XNLI(language='zh')
```

以中文数据集为例：

```text
premise hypo    label
从 概念 上 看 , 奶油 收入 有 两 个 基本 方面 产品 和 地理 .     产品 和 地理 是 什么 使 奶油 抹 霜 工作 .       neutral
我们 的 一个 号码 会 非常 详细 地 执行 你 的 指示       我 团队 的 一个 成员 将 非常 精确 地 执行 你 的 命令    entailment
男 女 同性恋 .  异性恋者        contradiction
```
每个字段以tab键分隔。类别netral表示中立，类别entailment表示蕴含，类别contradiction表示矛盾。

### Class `hub.dataset.TNews`

TNews是今日头条中文新闻（短文本）分类数据集，其目标是为短新闻进行分类。该数据集总共15个类别，包括 "news_story", "news_culture", "news_entertainment", "news_sports", "104": "news_finance", "news_house", "news_car", "news_edu", "news_tech", "news_military", "news_travel", "news_world", "stock", "news_agriculture", "news_game"。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.TNews()
```
数据集样例：
```text
6552277613866385923_!_104_!_news_finance_!_股票中的突破形态_!_股票
6553229876915077646_!_102_!_news_entertainment_!_他是最帅的古装男神，10国语言六门武术，演技在线却常演配角！_!_三生三世十里桃花,张智尧,杨门女将之女儿当自强,陆
小凤,印象深刻,陆小凤传奇,杨宗保,花满楼,古剑奇谭
6553551207028228622_!_102_!_news_entertainment_!_陈伟霆和黄晓明真的有差别，难怪baby会选择黄晓明_!_陈伟霆,黄晓明,粉丝
```
每个字段以“\_!\_”进行分隔，第一列表示数据ID，第二列表示类别ID， 第三列表示类别，第四列表示短新闻文本，第五列表示文本关键词。

### Class `hub.dataset.INews`

INews是一个互联网情感分析任务，其目标是判断中文长文本的情感倾向，数据集总共分3类（0，1，2）。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.INews()
```

数据集样例：
```text
label_!_id_!_title_!_content
1_!_bbf2e8a4824149bea36e703bbe8b0795_!_问大家一下，g20峰会之后，杭州的外国人是不是一下子多起来了_!_问大家一下，g20峰会之后，杭州的外国人是不是一下子多起来了，尤其是在杭州定居的外国人？
```
每个字段以“\_!\_”进行分隔，第一列表示类别，第二列表示数据ID， 第三列表示新闻标题，第四列表示新闻文本。


### Class `hub.dataset.DRCD`

DRCD是台达阅读理解数据集，属于通用领域繁体中文机器阅读理解数据集。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.DRCD()
```

数据格式和SQuAD相同，关于该数据集详细信息参见[DRCD](https://github.com/DRCKnowledgeTeam/DRCD)。

### Class `hub.dataset.CMRC2018`

CMRC2018聚焦基于篇章片段抽取的中文阅读理解，给定篇章、问题，其目标是从篇章中抽取出连续片段作为答案。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.CMRC2018()
```
数据格式和SQuAD相同，关于该数据集详细信息参见[CMRC](https://hfl-rc.github.io/cmrc2018/)

### Class `hub.dataset.BQ`

BQ是一个智能客服中文问句匹配数据集，该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值。数据中存在错别字、语法不规范等问题，但更加贴近工业场景。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.BQ()
```

数据集样例：
```text
请问一天是否都是限定只能转入或转出都是五万。    微众多少可以赎回短期理财        0
微粒咨询电话号码多少    你们的人工客服电话是多少        1
已经在银行换了新预留号码。      我现在换了电话号码，这个需要更换吗      1
```
每个字段以tab键分隔，第1，2列表示两个文本。第3列表示类别（0或1，0表示两个文本不相似，1表示两个文本相似）。

### Class `hub.dataset.IFLYTEK`

IFLYTEK是一个中文长文本分类数据集，该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.IFLYTEK()
```

数据集样例：
```text
70_!_随意发SendAnywherePro是一款文件分享工具，你可以快速的将分享手机中的照片、视频、联系人、应用、文件、文件夹等任何文件分享给其他人，可以在手机之前发送或接收，也可以通过官网www.sendweb.com不许要注册账号，只需输入一次key即可接收。
```
每个字段以“\_!\_”键分隔，第1列表示类别ID。第2列表示文本数据。

### Class `hub.dataset.THUCNEWS`

THUCNEWS是一个中文长文本分类数据集，该数据集共有4万多条中文新闻长文本标注数据，共14个类别，包括"体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.THUCNEWS()
```
数据集样例：

```text
0_!_体育_!_97498.txt_!_林书豪拒绝中国篮协邀请 将随中华台北征战亚锦赛　　信息时报讯 (记者 冯爱军) 中国篮协比中华台北篮协抢先一步了。据台湾媒体报道，刚刚成功签...倘若顺利，最快明年东亚区资格赛与亚锦赛就有机会看到林书豪穿上中华台北队球衣。”
```

每个字段以“\_!\_”键分隔，第1列表示类别ID，第2列表示类别，第3列表示文本数据。

### Class `hub.dataset.Couplet`

Couplet是一个开源对联数据集，来源于https://github.com/v-zich/couplet-clean-dataset。该数据集包含74万条对联数据，已利用敏感词词库过滤、删除了低俗或敏感内容。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.Couplet()
```
数据集样例：

```text
亲情似日堪融雪	孝意如春可著花
```

上下联中每个字以不可见字符“\002”分隔，上下联之间以tab键分隔。

### Class `hub.dataset.DogCatDataset`

DOGCAT是由Kaggle提供的数据集，用于图像二分类，其目标是判断一张图片是猫或是狗。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.DogCatDataset()
```
数据集样例：
```text
dog/4122.jpg 1
dog/6337.jpg 1
cat/3853.jpg 0
cat/5831.jpg 0
```
每个字段以空格键分隔。第一列表示图片所在路径，第二列表示图片类别，1表示属于，0表示不属于。

### Class `hub.dataset.Food101`

FOOD101 是由Kaggle提供的数据集，含有101种类别，其目标是判断一张图片属于101种类别中的哪一种类别。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.Food101()
```

关于该数据集详细信息参见[Kaggle Food101](https://www.kaggle.com/dansbecker/food-101)

### Class `hub.dataset.Indoor67`

INDOOR数据集是由麻省理工学院发布，其包含67种室内场景，其目标是识别一张室内图片的场景类别。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.Indoor67()
```

关于该数据集详细信息参见[Indoor67](http://web.mit.edu/torralba/www/indoor.html)

### Class `hub.dataset.Flowers`

Flowers数据集是是公开花卉数据集，数据集有5种类型，包括"roses"，"tulips"，"daisy"，"sunflowers"，"dandelion"。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.Flowers()
```

数据集样例：
```text
dandelion/7148085703_b9e8bcd6ca_n.jpg 4
roses/5206847130_ee4bf0e4de_n.jpg 0
tulips/8454707381_453b4862eb_m.jpg 1
```
每个字段以空格键分隔。第一列表示图片所在路径，第二列表示图片类别ID。

### Class `hub.dataset.StanfordDogs`

StanfordDogS数据集是斯坦福大学发布，其包含120个种类的狗，用于做图像分类。

**示例**

```python
import paddlehub as hub

dataset = hub.dataset.StanfordDogs()
```

关于该数据集详细信息参考[StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
