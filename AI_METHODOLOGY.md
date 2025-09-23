AI Methodology and Multi-Model Validation

Over the course of this research, multiple large language models (LLMs) were systematically employed to ensure analytical rigor, minimize hallucinations, and maintain scientific consistency. Each model was used according to its relative strengths:
	•	ChatGPT (GPT-4o / GPT-5 family): Used as the primary analysis engine. These models provided the most coherent and logically consistent interpretations of the raw simulation data. They excelled in producing structured, publication-ready narratives that connect results with theory. Their main limitation was occasional confusion when too much heterogeneous input was given in a single session, which was mitigated by isolating analyses into separate, clean chat sessions.
	•	Gemini (Google): Used as a secondary validation model. Its strength lies in re-checking analyses produced by ChatGPT, especially for consistency across different data representations. Gemini was also effective at spotting alternative perspectives on the same dataset. However, its narrative clarity and coherence were typically weaker than ChatGPT’s.
	•	DeepSeek (14B, offline version): Tested as an additional model for redundancy. In practice, it exhibited a high rate of hallucinations and insufficient robustness for scientific analysis. While unsuitable for primary interpretation, its role was limited to exploratory testing.

Workflow and Control

To maximize reliability, each major analysis step followed this pipeline:
	1.	Raw data input was analyzed in a dedicated chat session, ensuring a “clean slate” for each interpretation.
	2.	Results were cross-validated in a new session with another LLM, checking whether conclusions were consistent.
	3.	Final synthesis was performed by GPT-4o/5, which integrated the validated outputs into a coherent manuscript section.

This workflow leveraged the complementary strengths of different AI systems. GPT provided the deepest analysis, Gemini ensured logical consistency, and the limitations of DeepSeek highlighted the necessity of multi-model control.

Key Insight

The use of multiple LLMs was not redundant but rather a methodological safeguard. By repeatedly testing the same dataset across independent reasoning systems, the risk of hidden bias or hallucinated results was reduced. The convergence of independent AI outputs onto the same conclusions provides stronger confidence in the validity of the presented results.
