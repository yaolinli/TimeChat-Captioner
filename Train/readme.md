# Training Framework

For training (including **SFT** and **GRPO**), we adopt the **ms-swift v3.11** framework.

To improve video processing efficiency, we made several modifications to the original ms-swift codebase.
These changes can be found under:

```Train/modified/*```


We recommend configuring your training framework using one of the following two approaches:

## Option 1: Follow Our Setup
A modified version of ms-swift is already provided under:

```ThirdPartyLib/ms-swift```


This version includes all the changes required for our training pipeline.
You can set up the environment by installing it in editable mode:

``` bash
pip install -e ThirdPartyLib/ms-swift
```

## Option 2: Use your Training Framework

You may also choose to use another training framework that better suits your needs.

Since we do not introduce any special modifications to the training algorithms themselves, in principle, any framework that supports both SFT and GRPO can be used to train **TimeChat-Captioner**.

In this case, you can refer to the code under:

```Train/modified/*```


to adapt:

the **reward function**, and

the video processing logic

to your chosen framework.