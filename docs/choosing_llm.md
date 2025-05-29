## Some Guidelines to choose an LLM for your purpose
Depends on the task and resources available.

### 1. Self-served or API based

The choice between API-based and self-served LLMs revolves around where the LLM is hosted and how you interact with it. An API-based approach means you access the LLM remotely through an application programming interface provided by a third-party company. You send your requests to their servers and receive responses, without needing to manage the underlying infrastructure. This grants you access to powerful proprietary models like GPT or Claude and can be simpler for prototyping or lower workloads. In contrast, a self-served scenario involves hosting the LLM on your own servers or infrastructure. This typically utilizes open-source LLMs, and you are responsible for deployment, management, and optimization. While this requires more technical expertise in MLOps, it offers greater control over customization, data security, and can be more cost-effective for high-volume usage or specific fine-tuning needs.

| Feature          | API-Based                                   | Self-Served                                                                 |
|------------------|---------------------------------------------|-----------------------------------------------------------------------------|
| **Reliability**  | Subject to API lags and downtimes.          | Can be optimized for inference, potentially lower cost; requires LLMOps skills. |
| **Capability**   | Access to most capable proprietary LLMs.    | Open-source LLMs catching up; smaller models good for specific tasks and fine-tuning. |
| **Customization**| Some providers offer fine-tuning services.    | Full control over fine-tuning and experimentation.                           |
| **Data Security**| Data sent to third-party API providers.     | Data remains on your own servers.                                           |
| **Price Scaling**| Pay per token; prices potentially lowering due to competition. | Pay for compute plus deployment costs; potentially cheaper with high workload. |
| **Takeaways**    | Good for prototyping; easier without MLOps team. | Consider if cost and efforts justify the move from API; requires MLOps skills. |


### 2. Size and capability tiers
A parameter in the context of LLMs refers to the values within the matrices that make up the layers of the LLM's neural network. Think of these matrices as the model's internal memory or knowledge base.

Higher parameters mean that the matrices are either larger or there are more of them. This generally indicates a larger and potentially more capable model. A larger size, meaning more parameters, theoretically enhances an LLM's capabilities but also increases its demand for resources like storage and compute.

Model size also determines inference latency, that is the speed of answer generation.

Very small models (roughly 3B parameters or less)  
Small Models (roughtly under 15B)  
Larger Models  
Mixture-of-experts models  
Proprietary models  
&nbsp;&nbsp;&nbsp;&nbsp;Reasoning models => for => multi-step problems  
&nbsp;&nbsp;&nbsp;&nbsp;Non-reasoning models => for => everyday tasks  
Larger models perform better, and not surprisingly so. Indeed, larger size allows LLMs to "memorize" more facts.

To establish guardrails, we need to set a framework of LLM operations using a system prompt, but smaller the LLM is, the easier it gives in. 

### 3. Metrics and benchmarks
 
1. Chatbot Arena : An Open Platform for Evaluating LLMs by Human Preference (developed by researchers at UC Berkeley SkyLab and LMArena). LMArena is an open-source platform for crowdsourced AI benchmarking, created by researchers from UC Berkeley SkyLab.  
[Chatbot Arena LLM Leaderboard](https://lmarena.ai/?leaderboard)

Numerical comparision of LLMs. Refer to benchmarks. Few as examples below:

2. MT-Bench : A set of challenging multi-turn questions graded by GPT-4.  

3. MMLU : Massive Multitask Language Understanding : which accesses world knowledge and problem-solving ability across a range of topics and difficulty levels : a test to measure model's multitask accuracy on 57 tasks, from Abstract Algenra to Virology.  

 Specialized LLMs
 Reasoning
 Multilinguality
 Safety guardrails

 ### Benchmarks
 MTEB - Massive text embedding benchmark
 SWE-bench - SW Engineering benchmark - https://www.swebench.com

 https://huggingface.co/spaces/allenai/WildBench : Benchmarking LLMs with Challenging Tasks from Real Users in the Wild.  
 https://huggingface.co/blog/leaderboard-patronus: Evaluating the performance of language models on FinanceBench, Legal Confidentiality, Creative Writing, Customer Support Dialogue, Toxicity, and Enterprise PII.  
 https://huggingface.co/blog/leaderboard-decodingtrust: Evaluating LLMs from the point of view of toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, robustness on adversarial demonstrations, privacy, machine ethics, and fairness.  
 https://huggingface.co/blog/leaderboard-hallucinations    
 https://balrogai.com  



 