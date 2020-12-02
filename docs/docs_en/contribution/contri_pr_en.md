# Contribution Code

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

Documents are generated using  [sphinx](http://sphinx-doc.org/). Files support the [Markdown](https://guides.github.com/features/mastering-markdown/) and [reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)  formats. All documents are in the  [docs/](../../)  directory.

* Before submitting document changes, please Generate documents locally: `cd docs/ && make clean && make html` and then, all generated pages can be found in `docs/_build/html` directory. Please carefully analyze Each WARNING in generated logs, which is very likely or Empty connections or other problems.

* Try to use Relative Path when you need links.
