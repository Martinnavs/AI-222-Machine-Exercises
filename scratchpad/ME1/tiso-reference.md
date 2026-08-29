Task:
Using an agent, build a 3-layer CNN model for MNIST classification by implementing all layers/operations using einops/einsum. Train the model for 5 epochs and report the test split accuracy. 
Sample 16 images (from MNIST) and put them in 4x4 grid together with gt and prediction. Display all in a jupyter notebook.

Submit using your github repo link - agent initialized and committed - no human. Note you must still inspect and understand what the model and agent are doing.


Note: Do not go ahead and generate everything, this should be a collaborative approach between you and me, with you as the code and teacher and me as the student.

Instructions:
1. First generate an AI-only markdown file for the plan if you were to implement it by yourself, putting yourself as an engineer where the order of implementation makes sense. Store it within AI-222-Machine-Exercises under the claude-independent-plan/ dir. Add this as part of .gitignore.
2. Using what you've generated, be a socratic teacher/mentor and lead me through designing the process. Can also use ~/superpowers in conjunction with this to facilitate the discussion. Write the implementation within scratchpad/ME1/, where scratchpad is also part of .gitignore.
3. After each full implementation of one section, quiz me on what we've implemented so far.
4. After everything, format it to the desired format above (Jupyter Notebook) and document the main learnings, implementation decisions (ideally optimal), and a recap of what I need to know based on what we've talked about (ideally the gaps in my implem plan) if I were to implement this myself. Write this in the ME's scratchpad.

Output: 
For instruction 1, a markdown AI-only-posing-as-an-engineer plan.
For instruction 2, the code (in python) within the instructed directory.
For instruction 4, the machine exercise formatted Jupyter notebook.

Guardrails:
- Teaching first, where I should be the one "designing" this (of course under your supervision via the plan), implementation second. But ensure that the final implementation is as good as a ML engineer implements it.
- Einops (and by extension Einsum) is too arcane for a generic dev. In implementing this, utilize functions that promote readability (important -- don't overengineer ie creating one function for every call, just those that are repetitive or kinda obscure). The goal is to fulfill the requirements in the Task description without sacrificing readability.

Another guardrails:
- Installation of packages should be done via conda, which already exists here. Create one conda config for each machine exercise (for example ME1).
- For the actual implementation, we are limited to using Torch tensors only, no using of CNN, MLP, and other pre-implemented packages. The goal is to use Torch objects to store and process the data but the logic should be implemented by us.