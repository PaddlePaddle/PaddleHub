# 贡献代码

PaddleHub非常欢迎贡献者。

首先，如果有什么不确定的事情，可随时提交问题或拉取请求。 不会有人因此而抱怨。我们会感激任何形式的贡献，不想用一堆规则来阻止这些贡献。

本文档包括了所有在贡献中需要注意的要点，会加快合并代码、解决问题的速度。

查看[概览](../overview.md)来初步了解。

下面是一些简单的贡献指南。

## 提交问题

当你使用PaddleHub遇到问题时，可以通过提交[issue](https://github.com/PaddlePaddle/PaddleHub/issues)来反馈。

在提出问题时，请说明以下事项：

* 按照问题模板的内容来填写问题细节，以便评审者查找问题原因。
* 出现问题的场景 (尽量详细，以便重现问题)。
* 错误和日志消息。
* 其它可能有用的细节信息。

## 提交新功能建议/BUG修复

* 在适配使用场景时，总会需要一些新的功能。 可以加入新功能的讨论，也可以直接提交新功能的Pull-Request请求。

* 在自己的 github 账户下 fork PaddleHub(https://github.com/PaddlePaddle/PaddleHub)。 在 fork 后， 利用git工具（add, commit, pull, push）提交PR。 然后就可以提交拉取请求了。

如何提PR，参考下列步骤：

### 第一步：将自己目录下PaddleHub远程仓库clone到本地：

```
https://github.com/USERNAME/PaddleHub
```

### 第二步：切换到远程分支develop

```
git checkout develop
```

### 第三步：基于远程分支develop新建本地分支new-feature

```
git checkout -b new-feature
```

### 第四步：使用pre-commit钩子

PaddleHub开发人员使用pre-commit工具来管理Git预提交钩子。它可以帮助我们格式化源代码Python，在提交（commit）前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit测试是 Travis-CI 中单元测试的一部分，不满足钩子的PR不能被提交到Paddle，首先安装并在当前目录运行它：

```shell
➜  pip install pre-commit
➜  pre-commit install
```


### 第五步：在new-feature分支上开发你的需求，提交你的更改

```
git commit -m "add new feature"
```

### 第六步：在准备发起Pull Request之前，需要同步原仓库（https://github.com/PaddlePaddle/PaddleHub ）最新的代码。

通过 git remote 查看当前远程仓库的名字。

```shell
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/PaddleHub (fetch)
origin	https://github.com/USERNAME/PaddleHub (push)
```

这里 origin 是自己用户名下的PaddleHub，接下来创建一个原始PaddleHub仓库的远程主机，命名为 upstream。
```shell
➜  git remote add upstream https://github.com/PaddlePaddle/PaddleHub
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。
```shell
➜  git fetch upstream
➜  git pull upstream develop
```

### 第七步：推送本地分支new-feature到自己的PaddleHub库

```
➜  git push origin new-feature
```

这样你的PaddleHub库的new-feature分支包含了你的最新更改，点击上面的“pull request”就可以推送请求了。

如果评审人员给出了反馈需要继续修正代码，可以从第五步重新开始，这样所有的提交都会显示到同一个pull request中。

## 代码风格和命名约定

* PaddleHub 遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/) 的 Python 代码命名约定。在提交拉取请求时，请尽量遵循此规范。 可通过`flake8`或`pylint`的提示工具来帮助遵循规范。

## 文档

文档使用了 [sphinx](http://sphinx-doc.org/) 来生成，支持 [Markdown](https://guides.github.com/features/mastering-markdown/) 和 [reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) 格式。 所有文档都在 [docs/](../../) 目录下。

* 在提交文档改动前，请先**在本地生成文档**：`cd docs/ && make clean && make html`，然后，可以在 `docs/_build/html` 目录下找到所有生成的网页。 请认真分析生成日志中的**每个 WARNING**，这非常有可能是或**空连接**或其它问题。

* 需要链接时，尽量使用**相对路径**。
