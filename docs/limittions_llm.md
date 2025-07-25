## Problems with LLMs
1. Hallucination - generate incorrect information confidently. 
    *  Extrinsic hallucination - Output is inconsistent with the general knowledge
    *  Intrinsic hallucination
2. Lack of true understanding - Operate on statostical patterns, not true comprehension.
3. Bias
    * Sample bias
    * Label bias
    * Reporting bias
    * Mode collapsing
4. Jailbreaking
5. Context windows - Limited context, though improving.
6. Computational resources - Require significant resources.

## Below are the few reasons why LLMs like GPT-4 may produce hallucinations


1. Probabilistic Nature of Language Generation

    LLMs are fundamentally probabilistic systems trained to predict the next word in a sequence based on patterns learned from vast datasets. This training objective does not inherently equip them with mechanisms to verify the factual accuracy of their outputs. Consequently, when faced with prompts that lack clear answers or when the model's training data is insufficient, the model may generate plausible-sounding but incorrect or fabricated information. This tendency is a byproduct of the model's design to maximize likelihood over truthfulness.

2. Limitations in Training Data

    The quality and scope of the data used to train LLMs significantly influence their outputs. If the training data contains inaccuracies, biases, or lacks comprehensive coverage of certain topics, the model may produce hallucinations when queried about those areas. Moreover, **overrepresentation** of specific patterns or information in the training data can cause the model to **overfit**, leading to the generation of incorrect but seemingly familiar responses. Overfitting in LLMs means the model memorizes patterns seen during training so strongly that it blindly reproduces them, even in unrelated or incorrect contexts, rather than reasoning about what fits best.
    
    For example, Let’s say a model is trained heavily on tech forum posts where the phrase: “Restarting the system resolves most issues.” is repeated thousands of times. Now, if a user asks: “How can I fix a corrupted database index?”, even though the correct answer might involve running DB repair tools, the model may hallucinate and respond with: “Restarting the system should resolve this.” This happens because it overfit to a pattern seen too frequently in training — not because it “knows” it's the right solution for that specific problem.

    Similarly, in machine learning model development, the process of splitting data into training and testing sets is crucial for ensuring the model generalizes well. When dealing with categorical target variables, especially in scenarios with imbalanced class distributions, a simple random split can result in training or testing sets that do not accurately reflect the true proportions of each class in the original dataset. This is where techniques like stratification come into play. Stratifying the split, for instance using stratify=y in functions like train_test_split, ensures that the relative frequency of the target variable's classes is preserved in both the training and testing subsets. Without stratification in such cases, a model might be trained on a disproportionate representation of classes, potentially leading to overfitting on the majority class or poor performance on the minority class during testing. This situation mirrors how an LLM might "overfit" to prevalent patterns or biases within its training data, leading to confident but inaccurate or inappropriate responses (hallucinations) when encountering less represented or slightly different inputs, much like a classification model failing on a test set that doesn't accurately represent the real-world class balance. Ensuring representative data splits through stratification is analogous to curating a diverse and balanced training dataset for an LLM to mitigate the risk of such data-induced performance issues and hallucinations.

3. Decoding Strategies and Generation Techniques

    The methods employed during the text generation phase, such as top-k sampling or nucleus sampling, aim to enhance the diversity and creativity of the model's outputs. However, these techniques can also increase the likelihood of producing hallucinations, especially when the model ventures beyond well-represented knowledge areas. The balance between generating novel content and maintaining factual accuracy is delicate, and certain decoding strategies may inadvertently favor the former at the expense of the latter.

    ## Jailbreaking
    Trying to manipulate a bot into making what it's not supposed to do is known as jailbreaking.

