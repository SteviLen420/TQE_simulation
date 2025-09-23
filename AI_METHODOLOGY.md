## AI Methodology and Multi-Model Validation

Over the course of this research, multiple large language models (LLMs) were systematically employed to ensure analytical rigor, minimize hallucinations, and maintain scientific consistency. Each model was used according to its relative strengths:

•	**Gemini 2.5 ULTRA (Google)**: Used as the primary analysis engine. Each new dataset was examined in a fresh session, which provided a “clean slate” and stronger concentration. This approach minimized context-drift and improved consistency. However, when too much data was loaded into a single window, Gemini occasionally produced mistakes. To mitigate this, multiple independent sessions were opened, and results were repeatedly cross-checked to confirm convergence. Gemini’s main limitation is weaker coding ability, where it tended to introduce more errors compared to GPT-5.
	
•	**ChatGPT (GPT-5 family)**: Used mainly for theoretical synthesis, code generation, and as a secondary validation model. GPT-5 was particularly strong at building the broader theoretical framework and at producing reproducible code for the simulations. Its major drawback, however, was context persistence: since conversations are more interconnected across sessions, it did not always start from a perfectly “clean slate,” which increased the risk of inherited errors. Despite this, GPT-5 served as a valuable second layer of validation, complementing Gemini’s raw data analysis with strong coding and theoretical integration.
	
•	**DeepSeek R1 (14B, offline version**): Tested as an additional model for redundancy. Its analytical capability, given its 14B parameter scale, was inherently limited compared to larger models trained on broader datasets. While it can be suitable for simpler, well-defined tasks, it proved “too weak” for complex, multivariable scientific analyses. In practice, it exhibited a high rate of hallucinations and insufficient robustness. It was therefore not used for primary interpretation but highlighted the importance of multi-model control.
	
•	**GPT-5 and Code Generation**: Another important observation is that GPT-5 models are traditionally very strong at code generation and debugging. This makes them particularly valuable for simulation workflows where reproducible, executable code is critical.

### Workflow and Context Management

A key methodological insight is the importance of chat context management. When a single session grows too long and contains a mix of raw data, code, and analyses, even advanced models may start to lose precision due to context window limits. To mitigate this, each new, complex analysis was deliberately started in a separate, clean chat. This “best practice” ensured that the full reasoning capacity of the model was focused exclusively on the given problem, avoiding context fragmentation.

### Workflow and Control

To maximize reliability, each major analysis step followed this pipeline:
	1.	**Raw data** input was analyzed in a dedicated chat session, ensuring a “clean slate” for each interpretation.
	2.	**Results were cross-validated** in a new session with another LLM, checking whether conclusions were consistent.
	3.	**Final synthesis** was performed by GPT-4o/5, which integrated the validated outputs into a coherent manuscript section.

This workflow leveraged the complementary strengths of different AI systems. GPT provided the deepest analysis, Gemini ensured logical consistency, and the limitations of DeepSeek highlighted the necessity of multi-model control.

**Key Insight:**

The use of multiple LLMs was not redundant but rather a methodological safeguard. By repeatedly testing the same dataset across independent reasoning systems, the risk of hidden bias or hallucinated results was reduced. The convergence of independent AI outputs onto the same conclusions provides stronger confidence in the validity of the presented results.
