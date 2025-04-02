This describes my approach at fine-tuning `DeepSeek-R1-Distill-Qwen-1.5B` to classify code execution outputs by `DeepSeek-R1-Distill-Qwen-7B`.

However, I did not achieve improvements even in offline results. Nevertheless, I share my learnings here.

These are some hypotheses that motivate this

- I expect the `DeepSeek-R1-Distill-Qwen-7B` to be already well-tuned by DeepSeek to solve math problems from scratch. If I don't have lots of GPUs, I should not expect to outperform what DeepSeek has tuned.
- However, it seems that `DeepSeek-R1-Distill-Qwen-7B` is bad at recognizing whether the code execution output makes sense - maybe the LLM hasn't been trained to do this.
- Given the code execution output, if I am able to train the `DeepSeek-R1-Distill-Qwen-1.5B` to classify whether the code execution output can result in the answer, I can get a gold medal performance.

Therefore, I have reduced the task here from solving the full problem of solving math Olympiad problems to just classification of code execution output. The role of the `DeepSeek-R1-Distill-Qwen-1.5B` is now to decide between

- Submitting an answer by writing an answer in `\boxed{}` in which the answer might be wrong or correct
- Decline to submit and answer by pointing out some mistake and continuing with writing more Python code

There are some constraints on finetuning LLMs

- With some [free GPU credits from Modal labs](https://modal.com/pricing) from a [fine-tuning course I participated](https://maven.com/parlance-labs/fine-tuning), the biggest instance I have access to is a single 8 x H100 instance node.
- I found out that the 7B model tuned by the second prize winner [has only 1k context length](https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO/blob/main/finetune_code/scripts_aimo/finetune_policy.sh) and he used [8 X A6000 GPUs](https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO/blob/main/finetune_code/README.md) which has 48GB memory. (H100 has 80GB memory, not much more).
- However, reasoning models cannot do much with only 1k context length. Also for GRPO, the group computation requires all the entire group to be in [memory](https://github.com/huggingface/trl/issues/3061#issuecomment-2769820939). I should start with a 1.5B model instead with maybe 4k context length - and it seems that I need 2 x H100 at minimum, and scale the context length later if my experiments proves successful.

This is how I intend to approach tuning LLMs

- I first try to get the LLM to respond in the correct format. I have two reward functions regarding formatting - one for length and one for responding in sentences of similar length spaced by two newlines.
- After the LLM has plateaued in formatting rewards, then I introduce the correctness reward.
- The analogy here is that your table tennis coach will get you to learn the correct form first before getting you to return serves.

Dataset

- I marked LLM generation that are eligible for training. The completion string that is eligible for training is the string from the code execution output to end of text, or start of Python code delimiter "```python" Samples are available [here](https://www.kaggle.com/code/huikang/r1-distill-qwen-tir/output?select=generation_logs_training.csv).
- I filtered for generations that are under 3000 input tokens and under 1000 output tokens - because I wanted to start training on 4k context length. 3000 input tokens should allow for some multi-step interaction, which I want the LLM to do well in.
- For each question, I pair a generation that leads to a wrong answer and another generation that leads to the correct answer.
- The full dataset is [here](https://github.com/tonghuikang/grpo_aimo2/blob/master/training_dataset.csv).

Progress

- I could reproduce the [tutorial code](https://huggingface.co/docs/trl/main/en/grpo_trainer) provided by Huggingface's TRL library, on `DeepSeek-R1-Distill-Qwen-1.5B`
- I could get the LLM to improve in their formatting reward
- I could not get the LLM to improve the correctness reward

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1680925%2F3313cd47a44c3461447ede65fccd2fa7%2FScreenshot%202025-04-01%20at%2021.09.46.png?generation=1743567017204241&alt=media)

The reward function is
= `style_func` + `correctness_func`
= `f(k) * (length_ref + formatting_ref)` + `(1 - f(k)) * correctness_ref`

`f(k) = 1` until 25% of training and then linearly decreases to `0.25`
The maximum value of `length_ref`, `formatting_ref, and `correctness_ref` should be 1

Here are the plots explained
- `length_ref` and `formatting_ref` increases and plateaued at 25% of training.
- `correctness_ref` did not manage to improve. Yes it did improve initially, but that happened when no weight is given to `correctness_ref`. However it is no better than random or choosing to not submit answers every time - both of which will provide a `correctness_ref` value of zero. If the classification is perfect, the `correctness_ref` should be around 0.5, for reference.
- This is the report with more charts - https://api.wandb.ai/links/htong-quora-quora/w7hjqk8y which costs around $40 of Modal credits each run.

Learnings

- I should learn to code from Pytorch. This was what the second place winner from last year did. I still don't understand why a 1.5B model with group size of 4 and context length of 4000 is taking up one entire H100 memory. If I manage Pytorch code, I should have a better idea why. Working with Pytorch should be easier now that we can code with tools like Claude Code. 
- I should understand better exactly how logprobs are being updated.
- I should get the pipeline working on easier math problems like GSM8K first.
- Maybe I should have done SFT finetuning to imitate how the full R1 will respond to the code execution output.

The code is available here: https://github.com/tonghuikang/grpo_aimo2

Happy to take questions and elaborate more!