# AI Use

The report's appendix describes how we used AI assistants and how we checked their work. This file quotes example prompts, lightly edited for spelling and clarity, each with minimal context.

## Example 1: Onboarding onto the project (Yash)

With our brainstorming document attached, Yash asked an assistant to explain the scaffolded repository and suggest where he could contribute:

```text
The goal is to use quantization (in certain ways) to try to improve inference speed on VLAs such as pi0.5. The attached document shares from brainstorming of how this could work.

This project folder is currently the scaffolding to start development on this idea. Have a look through all of the .md to understand our process of how we're planning to do it.

I need you to do the following things first:

Explain the reproducible commands and procedure to test and run experiments.
Review the open pull requests and say whether it looks good to merge. For the checkpoint report merge, refer to this webpage: https://dsc-capstone.org/2026-27/assignments/projects/q1/
Say what are the next steps that I can contribute to. I have a 3090Ti if I need to recreate results/run some things.
```

## Example 2: Questioning a design choice and the report's wording (Issac)

Reviewing the assistant's draft of the report, Issac quoted a sentence from it:

```text
> The action-expert floor also assumes BF16 weights, although the checkpoint stores its normalization weights in FP32, so the shipped model cannot reach it.

What does the last part mean? Why store these weights in FP32? That seems dumb here. 

Also, the report switches between floor, roofline, understates and similar terms, so it's not immediately clear which direction is better. Let's normalize this.
```

The assistant traced the FP32 weights to a name-matching rule in LeRobot, which became a reduction proposed in the Discussion, and the report now uses one term, floor, throughout.
