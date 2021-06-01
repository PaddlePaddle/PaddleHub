# How to contribution code

PaddleHub welcomes contributors.

First of all, feel free to submit a question or pull request if there is something you are unsure about. No one will complain about it. We appreciate contributions of any kind, and don't want to block them with a bunch of rules.

This document includes all the key points to keep in mind when making contributions. This will speed up the process of merging code and solving problems.

Click Overview for an initial overview.

Here are some simple guidelines for making contributions.

## Submit a Question

When you encounter a problem with PaddleHub, you can provide feedback by submitting [issue](https://github.com/PaddlePaddle/PaddleHub/issues).

When asking your question, specify the following:

* Fill in the details of the problem according to the problem template so that the reviewer can find the cause of the problem.
* Problem scenarios (as detailed as possible to reproduce the problem):
* Error and log messages.
* Other details that may be useful.

## Submit new feature suggestions/bug fixing

* When adapting a usage scenario, there is always a need for new features. You can either join the discussion of new features, or submit a Pull-Request for new features directly.

* Fork PaddleHub (https://github.com/PaddlePaddle/PaddleHub) under your own github account. After fork, use the git tools (add, commit, pull, push) to submit the PR. Then you can submit the pull-request.

To do the PR, follow the following steps:

### Step 1: Clone the PaddleHub remote repository in your own directory to your local.

```
https://github.com/USERNAME/PaddleHub
```

### Step 2: Switch to the remote branch develop.

```
git checkout develop
```

### Step 3: Create a local branch new-feature based on remote branch develop.

```
git checkout -b new-feature
```

### Step 4: Use the pre-commit hook.

PaddleHub developers use the pre-commit tool to manage Git pre-commit hooks. It helps us to format the source Python and automatically check some basic things before committing (for example, only one EOL per file, and not adding large files in Git).

The pre-commit test is a part of the unit tests in Travis-CI. PRs that do not meet the hooks cannot be committed to Paddle. First install and run it in the current directory.

```shell
➜  pip install pre-commit
➜  pre-commit install
```

### Step 5: Develop your requirements on the new-feature branch and commit your changes.

```
git commit -m "add new feature"
```

### Step 6: Before you are ready to launch a Pull Request, you need to synchronize the latest codes from the original repository (https://github.com/PaddlePaddle/PaddleHub).

Check the name of the current remote repository via git remote.

```shell
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/PaddleHub (fetch)
origin	https://github.com/USERNAME/PaddleHub (push)
```

Here, origin is PaddleHub under your own username. Next, create a remote host of the original PaddleHub repository and name it upstream.

```shell
➜  git remote add upstream https://github.com/PaddlePaddle/PaddleHub
➜  git remote
origin
upstream
```

Get the latest code for upstream and update the current branch.

```shell
➜  git fetch upstream
➜  git pull upstream develop
```

### Step 7: Push a local branch new-feature to your own PaddleHub repository.

```
➜  git push origin new-feature
```

So, the new-feature branch of your PaddleHub repository contains your latest changes, click "pull request" above to push the request.

If the reviewer gives you feedback that you need to continue fixing the codes, you can start over from step 5 so that all commits are displayed in the same pull request.

## Code Style and Naming Conventions

* PaddleHub follows the Python code naming convention of  [PEP8](https://www.python.org/dev/peps/pep-0008/). Try to follow this specification when submitting pull requests. You can use the `flake8` or `pylint` hint tool to help you follow this specification.

## Document

Documents are generated using  [sphinx](http://sphinx-doc.org/). Files support the [Markdown](https://guides.github.com/features/mastering-markdown/) and [reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) formats. All documents are in the [docs/](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/docs) directory.

* Before submitting document changes, please Generate documents locally: `cd docs/ && make clean && make html` and then, all generated pages can be found in `docs/_build/html` directory. Please carefully analyze Each WARNING in generated logs, which is very likely or Empty connections or other problems.

* Try to use Relative Path when you need links.

## Thanks for contribution

<p align="center">
    <a href="https://github.com/nepeplwu"><img src="https://avatars.githubusercontent.com/u/45024560?v=4" width=75 height=75></a>
    <a href="https://github.com/Steffy-zxf"><img src="https://avatars.githubusercontent.com/u/48793257?v=4" width=75 height=75></a>
    <a href="https://github.com/ZeyuChen"><img src="https://avatars.githubusercontent.com/u/1371212?v=4" width=75 height=75></a>
    <a href="https://github.com/ShenYuhan"><img src="https://avatars.githubusercontent.com/u/28444161?v=4" width=75 height=75></a>
    <a href="https://github.com/kinghuin"><img src="https://avatars.githubusercontent.com/u/11913168?v=4" width=75 height=75></a>
    <a href="https://github.com/haoyuying"><img src="https://avatars.githubusercontent.com/u/35907364?v=4" width=75 height=75></a>
    <a href="https://github.com/grasswolfs"><img src="https://avatars.githubusercontent.com/u/23690325?v=4" width=75 height=75></a>
    <a href="https://github.com/sjtubinlong"><img src="https://avatars.githubusercontent.com/u/2063170?v=4" width=75 height=75></a>
    <a href="https://github.com/KPatr1ck"><img src="https://avatars.githubusercontent.com/u/22954146?v=4" width=75 height=75></a>
    <a href="https://github.com/jm12138"><img src="https://avatars.githubusercontent.com/u/15712990?v=4" width=75 height=75></a>
    <a href="https://github.com/DesmonDay"><img src="https://avatars.githubusercontent.com/u/20554008?v=4" width=75 height=75></a>
    <a href="https://github.com/adaxiadaxi"><img src="https://avatars.githubusercontent.com/u/58928121?v=4" width=75 height=75></a>
    <a href="https://github.com/chunzhang-hub"><img src="https://avatars.githubusercontent.com/u/63036966?v=4" width=75 height=75></a>
    <a href="https://github.com/linshuliang"><img src="https://avatars.githubusercontent.com/u/15993091?v=4" width=75 height=75></a>
    <a href="https://github.com/eepgxxy"><img src="https://avatars.githubusercontent.com/u/15946195?v=4" width=75 height=75></a>
    <a href="https://github.com/houj04"><img src="https://avatars.githubusercontent.com/u/35131887?v=4" width=75 height=75></a>
    <a href="https://github.com/paopjian"><img src="https://avatars.githubusercontent.com/u/20377352?v=4" width=75 height=75></a>
    <a href="https://github.com/zbp-xxxp"><img src="https://avatars.githubusercontent.com/u/58476312?v=4" width=75 height=75></a>
    <a href="https://github.com/dxxxp"><img src="https://avatars.githubusercontent.com/u/15886898?v=4" width=75 height=75></a>
    <a href="https://github.com/1084667371"><img src="https://avatars.githubusercontent.com/u/50902619?v=4" width=75 height=75></a>
    <a href="https://github.com/Channingss"><img src="https://avatars.githubusercontent.com/u/12471701?v=4" width=75 height=75></a>
    <a href="https://github.com/Austendeng"><img src="https://avatars.githubusercontent.com/u/16330293?v=4" width=75 height=75></a>
    <a href="https://github.com/BurrowsWang"><img src="https://avatars.githubusercontent.com/u/478717?v=4" width=75 height=75></a>
    <a href="https://github.com/cqvu"><img src="https://avatars.githubusercontent.com/u/37096589?v=4" width=75 height=75></a>
    <a href="https://github.com/Haijunlv"><img src="https://avatars.githubusercontent.com/u/28926237?v=4" width=75 height=75></a>
    <a href="https://github.com/holyseven"><img src="https://avatars.githubusercontent.com/u/13829174?v=4" width=75 height=75></a>
    <a href="https://github.com/MRXLT"><img src="https://avatars.githubusercontent.com/u/16594411?v=4" width=75 height=75></a>
    <a href="https://github.com/cclauss"><img src="https://avatars.githubusercontent.com/u/3709715?v=4" width=75 height=75></a>
    <a href="https://github.com/hu-qi"><img src="https://avatars.githubusercontent.com/u/17986122?v=4" width=75 height=75></a>
    <a href="https://github.com/jayhenry"><img src="https://avatars.githubusercontent.com/u/4285375?v=4" width=75 height=75></a>
</p>

* Many thanks to [肖培楷](https://github.com/jm12138), Contributed to street scene cartoonization, portrait cartoonization, gesture key point recognition, sky replacement, depth estimation, portrait segmentation and other modules
* Many thanks to [Austendeng](https://github.com/Austendeng) for fixing the SequenceLabelReader
* Many thanks to [cclauss](https://github.com/cclauss) optimizing travis-ci check
* Many thanks to [奇想天外](http://www.cheerthink.com/)，Contributed a demo of mask detection
* Many thanks to [mhlwsk](https://github.com/mhlwsk)，Contributed the repair sequence annotation prediction demo
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp)，Contributed modules for viewing pictures and writing poems
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp) and [七年期限](https://github.com/1084667371),Jointly contributed to the Mid-Autumn Festival Special Edition Module
* Many thanks to [livingbody](https://github.com/livingbody)，Contributed models for style transfer based on PaddleHub's capabilities and Mid-Autumn Festival WeChat Mini Program
* Many thanks to [BurrowsWang](https://github.com/BurrowsWang) for fixing Markdown table display problem
* Many thanks to [huqi](https://github.com/hu-qi) for fixing readme typo
