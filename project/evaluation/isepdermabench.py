"""
Internal Benchmark specific for dermatology accuracy, it's split into different tasks:

https://huggingface.co/datasets/danielfdias98/ISEPDermaBench

Tasks:

Grounded Diagnosis: Evaluate model based on top-k accuracy of diagonsis
Context Ablation: Evaluate the model before missing Context and After
Visual Confusion: Diagnosis on easy and hard pairs between similar diseases
General Hallucation: Evaluate Hallucation on the Model

Open Ended: Diagnosis of the model based on a free-text ( Requires LLM as a Judge )
Evidence Grouding: Grounding of the Model based on the evidence described ( Requires LLM as a Judge )

"""