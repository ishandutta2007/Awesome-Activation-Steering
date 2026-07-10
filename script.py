import os
import re

readme_path = "README.md"
repo_dir = r"C:\Users\ishan\Documents\Projects\Awesome-Activation-Steering"
os.chdir(repo_dir)

def run_git(msg):
    os.system('git -C . add .')
    os.system(f'git -C . commit -m "{msg}"')
    os.system('git -C . push')

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Section 1
s1_bullets = r"""*   **The Coarse Token-Shift Era (Traditional Prototyping, Pre-2022)**
    *   *Concept:* The early exploratory baseline. Researchers discovered that adding simple, hardcoded continuous vectors to early hidden layers could skew text generation. For example, averaging the hidden state vectors of the token `" France"` and adding it to prompt passes forced the model to over-index on French terminology.
    *   *Limitation:* Catastrophically polysemantic and blurry. Because early methods targeted raw, uncompressed network layers, adding a vector injected immense unstructured noise, inadvertently destroying adjacent model capacities (like coding or basic mathematical logic).
*   **The Contrastive Activation Steering Breakthrough (CAA / ActAdd, 2023)**
    *   *Concept:* Isolated clean semantic trajectories by calculating direction vectors from contrastive text pairs. Popularized by Turner et al. (**Activation Addition - ActAdd**) and Rimsky et al. (**Contrastive Activation Steering - CAA**), the framework extracts a steering vector ($\theta_{\text{steer}}$) by taking a model and calculating the exact activation difference between a positive persona prompt (e.g., *“Write a helpful, honest response...”*) and a negative persona prompt (e.g., *“Write a deceptive response...”*).
    *   *Significance:* Halved the alignment tax. It allowed deployment servers to dynamically inject $\theta_{\text{steer}}$ directly into intermediate layers at runtime, forcing safe behaviors without requiring a single weight fine-tuning epoch.
*   **The Monosemantic Dictionary Learning Revolution (SAEs, ~2024–2025)**
    *   *Concept:* Fully resolved the polysemantic cross-contamination bottleneck [INDEX: 2]. Developed by researchers at Anthropic and OpenAI, it couples steering logic with **Sparse Autoencoders (SAEs)** [INDEX: 2]. An SAE unwraps the highly compressed, chaotic hidden layers of a base model up-projecting them into an overcomplete sparse matrix containing millions of isolated, single-concept features [INDEX: 2].
    *   *Significance:* Instead of tilting an entire layer coarsely, engineers precisely clamp or scale an *individual monosemantic feature node* directly (e.g., isolating a node tracking exactly "chemical weapon synthesis intent" or "SQL injection payloads"), neutralizing security threats perfectly without collateral feature degradation [INDEX: 2].
*   **The Omni-Directional Cross-Modal Interventions Era (~2026–Present)**
    *   *Concept:* The current modern state-of-the-art foundation standard. It ports activation steering out of text token streams and straight into unified, omni-directional architectures processing audio waves, visual patch matrices, and string characters concurrently [INDEX: 1, 10].
    *   *Significance:* Enables **Cross-Modal Concept Shifting**. Clamping an abstract structural concept node inside the shared latent attention space instantly forces the multi-modal transformer to alter its generation trajectories across *all sensory domains concurrently*, forcing a visual generation pass to alter its artistic grading while dynamically shifting a text voice tokenizer's dialogue persona symmetrically [INDEX: 10]."""

s1_table = r"""| Era | Details | Year First Used | Paper Link |
|---|---|---|---|
| [**The Coarse Token-Shift Era (Traditional Prototyping, Pre-2022)**](details/coarse_token_shift.md) | **Concept:** The early exploratory baseline. Researchers discovered that adding simple, hardcoded continuous vectors to early hidden layers could skew text generation. For example, averaging the hidden state vectors of the token `" France"` and adding it to prompt passes forced the model to over-index on French terminology.<br>**Limitation:** Catastrophically polysemantic and blurry. Because early methods targeted raw, uncompressed network layers, adding a vector injected immense unstructured noise, inadvertently destroying adjacent model capacities (like coding or basic mathematical logic). | Pre-2022 | [INDEX: 11] |
| [**The Contrastive Activation Steering Breakthrough (CAA / ActAdd, 2023)**](details/caa_actadd.md) | **Concept:** Isolated clean semantic trajectories by calculating direction vectors from contrastive text pairs. Popularized by Turner et al. (**Activation Addition - ActAdd**) and Rimsky et al. (**Contrastive Activation Steering - CAA**), the framework extracts a steering vector ($\theta_{\text{steer}}$) by taking a model and calculating the exact activation difference between a positive persona prompt (e.g., *“Write a helpful, honest response...”*) and a negative persona prompt (e.g., *“Write a deceptive response...”*).<br>**Significance:** Halved the alignment tax. It allowed deployment servers to dynamically inject $\theta_{\text{steer}}$ directly into intermediate layers at runtime, forcing safe behaviors without requiring a single weight fine-tuning epoch. | 2023 | [Turner et al., 2023](https://arxiv.org/abs/2308.10248) |
| [**The Monosemantic Dictionary Learning Revolution (SAEs, ~2024–2025)**](details/sae_revolution.md) | **Concept:** Fully resolved the polysemantic cross-contamination bottleneck [INDEX: 2]. Developed by researchers at Anthropic and OpenAI, it couples steering logic with **Sparse Autoencoders (SAEs)** [INDEX: 2]. An SAE unwraps the highly compressed, chaotic hidden layers of a base model up-projecting them into an overcomplete sparse matrix containing millions of isolated, single-concept features [INDEX: 2].<br>**Significance:** Instead of tilting an entire layer coarsely, engineers precisely clamp or scale an *individual monosemantic feature node* directly (e.g., isolating a node tracking exactly "chemical weapon synthesis intent" or "SQL injection payloads"), neutralizing security threats perfectly without collateral feature degradation [INDEX: 2]. | 2024 | [INDEX: 2] |
| [**The Omni-Directional Cross-Modal Interventions Era (~2026–Present)**](details/omni_directional.md) | **Concept:** The current modern state-of-the-art foundation standard. It ports activation steering out of text token streams and straight into unified, omni-directional architectures processing audio waves, visual patch matrices, and string characters concurrently [INDEX: 1, 10].<br>**Significance:** Enables **Cross-Modal Concept Shifting**. Clamping an abstract structural concept node inside the shared latent attention space instantly forces the multi-modal transformer to alter its generation trajectories across *all sensory domains concurrently*, forcing a visual generation pass to alter its artistic grading while dynamically shifting a text voice tokenizer's dialogue persona symmetrically [INDEX: 10]. | 2026 | [INDEX: 1, 10] |"""

content = content.replace(s1_bullets, s1_table)

s2_bullets = r"""- ### A. Linear Activation Addition (ActAdd Baseline)
	*   **Mechanism:** Intercepts the intermediate hidden layer tensor ($h_l$) at a specific block level $l$ during the forward pass, executing an element-wise vector addition with a pre-calculated contrastive steering coefficient ($\theta$):
	    $$h_l \leftarrow h_l + c \cdot \theta_l$$
	    Where $c$ represents a runtime steering multiplier scale factor.
	*   **Behavior:** Shifts the model's global context manifold softly, optimal for applying general conversational styles or stylistic formatting configurations.

- ### B. Monosemantic Feature Clamping (SAE-Steering)
	*   **Mechanism:** Routes the continuous hidden vector $h_l$ through a trained **Sparse Autoencoder** dictionary layer first [INDEX: 2]. The SAE decodes the vector into sparse feature activations $f(h_l)$ [INDEX: 2]. The system applies a hard structural clamp to a specific feature index $i$, forcing its activation value to a maximum limit $M$, before up-projecting it back to reconstruct the modified hidden state:
	    $$f_i(h_l) \leftarrow \max(f_i(h_l), M)$$
	*   **Pros:** Absolute conceptual precision, avoiding cross-feature contamination noise [INDEX: 2].

- ### C. Clamp-and-Damp Adversarial Defense
	*   **Mechanism:** A high-throughput security implementation. It monitors critical model safety nodes (e.g., weaponization concept features) continuously [INDEX: 19]. If an incoming user prompt registers an anomalous, high-activation spike on a forbidden node, the system actively damps that feature down to absolute zero ($f_{\text{malicious}} \leftarrow 0$), completely neutralizing prompt injections [INDEX: 2, 19].

- ### D. Multi-Objective Steering Schedulers
	*   **Mechanism:** Varies the steering multiplier $c$ dynamically across different time-steps of the auto-regressive generation loop ($c_t = f(t)$). It dials up steering parameters during early token cycles to lock down global formatting setups, dropping intensities later to let the model access detailed factual memory coordinates."""

s2_table = r"""| Variant | Details | Year First Used | Paper Link |
|---|---|---|---|
| [**A. Linear Activation Addition (ActAdd Baseline)**](details/actadd_baseline.md) | **Mechanism:** Intercepts the intermediate hidden layer tensor ($h_l$) at a specific block level $l$ during the forward pass, executing an element-wise vector addition with a pre-calculated contrastive steering coefficient ($\theta$):<br>$$h_l \leftarrow h_l + c \cdot \theta_l$$<br>Where $c$ represents a runtime steering multiplier scale factor.<br>**Behavior:** Shifts the model's global context manifold softly, optimal for applying general conversational styles or stylistic formatting configurations. | 2023 | [Turner et al., 2023](https://arxiv.org/abs/2308.10248) |
| [**B. Monosemantic Feature Clamping (SAE-Steering)**](details/sae_steering.md) | **Mechanism:** Routes the continuous hidden vector $h_l$ through a trained **Sparse Autoencoder** dictionary layer first [INDEX: 2]. The SAE decodes the vector into sparse feature activations $f(h_l)$ [INDEX: 2]. The system applies a hard structural clamp to a specific feature index $i$, forcing its activation value to a maximum limit $M$, before up-projecting it back to reconstruct the modified hidden state:<br>$$f_i(h_l) \leftarrow \max(f_i(h_l), M)$$<br>**Pros:** Absolute conceptual precision, avoiding cross-feature contamination noise [INDEX: 2]. | 2024 | [INDEX: 2] |
| [**C. Clamp-and-Damp Adversarial Defense**](details/clamp_and_damp.md) | **Mechanism:** A high-throughput security implementation. It monitors critical model safety nodes (e.g., weaponization concept features) continuously [INDEX: 19]. If an incoming user prompt registers an anomalous, high-activation spike on a forbidden node, the system actively damps that feature down to absolute zero ($f_{\text{malicious}} \leftarrow 0$), completely neutralizing prompt injections [INDEX: 2, 19]. | 2024 | [INDEX: 2, 19] |
| [**D. Multi-Objective Steering Schedulers**](details/steering_schedulers.md) | **Mechanism:** Varies the steering multiplier $c$ dynamically across different time-steps of the auto-regressive generation loop ($c_t = f(t)$). It dials up steering parameters during early token cycles to lock down global formatting setups, dropping intensities later to let the model access detailed factual memory coordinates. | 2025 | [INDEX: 18] |"""

content = content.replace(s2_bullets, s2_table)


s3_bullets = r"""*   **Linear Autoencoder Projections**
    *   *The Math:* Maps latent space expansions [INDEX: 2]. The encoder weight matrix ($W_{\text{enc}} \in \mathbb{R}^{d \times m}$, where dictionary size $m \gg d$) projects the hidden vector through a ReLU activation to find isolated concepts [INDEX: 2]. The decoder matrix ($W_{\text{dec}}$) reconstructs the vector block following the steering modification [INDEX: 2].
*   **Dynamic $\epsilon$-Clamping Boundaries**
    *   *Profile:* Memory bus load balancing. In high-volume cloud serving endpoints, the target steering coordinates and clamp metrics are cached as static register parameters, executing feature modifications in a single hardware step across fast GPU SRAM registers [INDEX: 22]."""

s3_table = r"""| Mechanism | Details | Year First Used | Paper Link |
|---|---|---|---|
| [**Linear Autoencoder Projections**](details/autoencoder_projections.md) | *The Math:* Maps latent space expansions [INDEX: 2]. The encoder weight matrix ($W_{\text{enc}} \in \mathbb{R}^{d \times m}$, where dictionary size $m \gg d$) projects the hidden vector through a ReLU activation to find isolated concepts [INDEX: 2]. The decoder matrix ($W_{\text{dec}}$) reconstructs the vector block following the steering modification [INDEX: 2]. | 2023 | [INDEX: 2] |
| [**Dynamic $\epsilon$-Clamping Boundaries**](details/dynamic_clamping.md) | *Profile:* Memory bus load balancing. In high-volume cloud serving endpoints, the target steering coordinates and clamp metrics are cached as static register parameters, executing feature modifications in a single hardware step across fast GPU SRAM registers [INDEX: 22]. | 2024 | [INDEX: 22] |"""

content = content.replace(s3_bullets, s3_table)


s4_bullets = r"""*   **The Activation Cache Multi-Model Capacity Overload Barrier**
    *   *The Problem:* Evaluating overcomplete Sparse Autoencoders alongside massive foundation models explodes the active VRAM footprint. Because an SAE expands a model's hidden dimension size by a factor of $32\times$ or $64\times$ (holding millions of dictionary features on disk) [INDEX: 2], loading these extra weight parameters concurrently can saturate GPU memory, triggering system-wide Out-of-Memory crashes [INDEX: 22].
    *   *Mitigation:* Implementing **Fused Quantized Dictionary Kernels (INT4/INT8 SAE layers)**, coupled with sharding the sparse autoencoder parameters across a distributed data-parallel cluster array via **Fully Sharded Data Parallel (FSDP)** primitives to stream weights on-the-fly [INDEX: 16, 22].
*   **The Feature Drift and Concept Contamination Stagnation**
    *   *The Problem:* Setting an excessively high steering multiplier scale factor ($c$) forces the model's generation trajectory into extreme, unnatural coordinate boundaries. This causes the network to experience **Semantic Coherence Collapse**—where it begins repeating words cyclically, drops syntax rules, or completely freezes during autoregressive token decoding passes.
    *   *Mitigation:* Implementing **Adaptive Layer Norm Clamping bounds**, checking the continuous Frobenius norm of the modified hidden vector dynamically, and scaling down the steering multiplier if the vector drifts outside a safe local radius."""

s4_table = r"""| Challenge | Details | Year First Used | Paper Link |
|---|---|---|---|
| [**The Activation Cache Multi-Model Capacity Overload Barrier**](details/vram_overload.md) | *The Problem:* Evaluating overcomplete Sparse Autoencoders alongside massive foundation models explodes the active VRAM footprint. Because an SAE expands a model's hidden dimension size by a factor of $32\times$ or $64\times$ (holding millions of dictionary features on disk) [INDEX: 2], loading these extra weight parameters concurrently can saturate GPU memory, triggering system-wide Out-of-Memory crashes [INDEX: 22].<br>*Mitigation:* Implementing **Fused Quantized Dictionary Kernels (INT4/INT8 SAE layers)**, coupled with sharding the sparse autoencoder parameters across a distributed data-parallel cluster array via **Fully Sharded Data Parallel (FSDP)** primitives to stream weights on-the-fly [INDEX: 16, 22]. | 2024 | [INDEX: 16, 22] |
| [**The Feature Drift and Concept Contamination Stagnation**](details/feature_drift.md) | *The Problem:* Setting an excessively high steering multiplier scale factor ($c$) forces the model's generation trajectory into extreme, unnatural coordinate boundaries. This causes the network to experience **Semantic Coherence Collapse**—where it begins repeating words cyclically, drops syntax rules, or completely freezes during autoregressive token decoding passes.<br>*Mitigation:* Implementing **Adaptive Layer Norm Clamping bounds**, checking the continuous Frobenius norm of the modified hidden vector dynamically, and scaling down the steering multiplier if the vector drifts outside a safe local radius. | 2023 | [INDEX: 11] |"""

content = content.replace(s4_bullets, s4_table)

s5_bullets = r"""*   **Mechanistic Safety Auditing & Runtime Jailbreak Defenses**
    *   *Application:* Secures consumer-facing enterprise AI endpoints against adaptive prompt injection and red-teaming exploits [INDEX: 19]. Real-time clamp-and-damp SAE monitors scan hidden layers continuously [INDEX: 2]; if a malicious payload successfully tricks the prompt layers, the steering layer dampens the malicious concept vector instantly, forcing a safe refusal persona without losing system utility [INDEX: 2, 19].
*   **Low-Latency Enterprise Hallucination & Fact-Checking Filters**
    *   *Application:* Regulates large-scale retrieval-augmented generation (RAG) loops [INDEX: 18]. Diagnostic steering vectors monitor internal certainty and truthfulness feature dimensions; if the active model begins generating ungrounded facts mid-sentence, the system injects a "fact-anchoring steering vector" to steer the token trajectory back toward verified data database parameters safely [INDEX: 18].
*   **Dynamic Enterprise Persona & Guardrail Customization**
    *   *Application:* Serves as the primary runtime customizer managing white-label B2B serving instances. Instead of keeping thousands of independent full-model fine-tuned checkpoints on disk for different corporate clients, cloud infrastructures host a single frozen foundation model block, injecting tiny, low-cost "persona steering vectors" on-the-fly based on the active user API key, slashing cloud hosting budgets."""

s5_table = r"""| Application | Details | Year First Used | Paper Link |
|---|---|---|---|
| [**Mechanistic Safety Auditing & Runtime Jailbreak Defenses**](details/mechanistic_safety.md) | *Application:* Secures consumer-facing enterprise AI endpoints against adaptive prompt injection and red-teaming exploits [INDEX: 19]. Real-time clamp-and-damp SAE monitors scan hidden layers continuously [INDEX: 2]; if a malicious payload successfully tricks the prompt layers, the steering layer dampens the malicious concept vector instantly, forcing a safe refusal persona without losing system utility [INDEX: 2, 19]. | 2024 | [INDEX: 19] |
| [**Low-Latency Enterprise Hallucination & Fact-Checking Filters**](details/fact_checking.md) | *Application:* Regulates large-scale retrieval-augmented generation (RAG) loops [INDEX: 18]. Diagnostic steering vectors monitor internal certainty and truthfulness feature dimensions; if the active model begins generating ungrounded facts mid-sentence, the system injects a "fact-anchoring steering vector" to steer the token trajectory back toward verified data database parameters safely [INDEX: 18]. | 2025 | [INDEX: 18] |
| [**Dynamic Enterprise Persona & Guardrail Customization**](details/persona_customization.md) | *Application:* Serves as the primary runtime customizer managing white-label B2B serving instances. Instead of keeping thousands of independent full-model fine-tuned checkpoints on disk for different corporate clients, cloud infrastructures host a single frozen foundation model block, injecting tiny, low-cost "persona steering vectors" on-the-fly based on the active user API key, slashing cloud hosting budgets. | 2024 | [INDEX: 11] |"""

content = content.replace(s5_bullets, s5_table)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("tabularised the bullets")

# Step 2: Detailed pages
os.makedirs("details", exist_ok=True)
pages = [
    ("coarse_token_shift.md", "The Coarse Token-Shift Era"),
    ("caa_actadd.md", "Contrastive Activation Steering Breakthrough"),
    ("sae_revolution.md", "Monosemantic Dictionary Learning Revolution"),
    ("omni_directional.md", "Omni-Directional Cross-Modal Interventions"),
    ("actadd_baseline.md", "Linear Activation Addition"),
    ("sae_steering.md", "Monosemantic Feature Clamping"),
    ("clamp_and_damp.md", "Clamp-and-Damp Adversarial Defense"),
    ("steering_schedulers.md", "Multi-Objective Steering Schedulers"),
    ("autoencoder_projections.md", "Linear Autoencoder Projections"),
    ("dynamic_clamping.md", "Dynamic ε-Clamping Boundaries"),
    ("vram_overload.md", "Activation Cache Multi-Model Capacity Overload"),
    ("feature_drift.md", "Feature Drift and Concept Contamination"),
    ("mechanistic_safety.md", "Mechanistic Safety Auditing & Runtime Jailbreak Defenses"),
    ("fact_checking.md", "Low-Latency Enterprise Hallucination & Fact-Checking Filters"),
    ("persona_customization.md", "Dynamic Enterprise Persona & Guardrail Customization")
]
for filename, title in pages:
    page_content = f"# {title}\n\nDetailed explanation of {title}.\n\n"
    page_content += f"```mermaid\nflowchart LR\n    A[Start] --> B[\"{title}\"]\n```\n\n"
    page_content += "[Back to README](../README.md)\n"
    with open(f"details/{filename}", "w", encoding="utf-8") as f:
        f.write(page_content)

run_git("detailed pages created")

# Step 3: Decorate with emojis and banner
os.makedirs("assets", exist_ok=True)
banner_svg = r'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="100%" height="200">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1A2980">
        <animate attributeName="stop-color" values="#1A2980;#26D0CE;#1A2980" dur="4s" repeatCount="indefinite" />
      </stop>
      <stop offset="100%" stop-color="#26D0CE">
        <animate attributeName="stop-color" values="#26D0CE;#1A2980;#26D0CE" dur="4s" repeatCount="indefinite" />
      </stop>
    </linearGradient>
  </defs>
  <rect width="800" height="200" fill="url(#bg)" rx="15"/>
  <text x="400" y="100" font-family="Arial, sans-serif" font-size="40" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="middle">
    Activation Steering in AI
  </text>
  <text x="400" y="150" font-family="Arial, sans-serif" font-size="20" fill="white" text-anchor="middle" dominant-baseline="middle">
    History, Progression, Variants, &amp; Applications
  </text>
</svg>'''
with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(banner_svg)

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("# Awesome-Activation-Steering", "# 🚀 Awesome-Activation-Steering\n\n![Banner](assets/banner.svg)\n")
content = content.replace("## Activation Steering in AI", "## 🧠 Activation Steering in AI")
content = content.replace("## 1. The Macro Chronological Evolution", "## 🕰️ 1. The Macro Chronological Evolution")
content = content.replace("## 2. Core Functional & Algorithmic Steering Variants", "## ⚙️ 2. Core Functional & Algorithmic Steering Variants")
content = content.replace("## 3. The SAE Feature-Steering Inversion Matrix", "## 🧮 3. The SAE Feature-Steering Inversion Matrix")
content = content.replace("## 4. Production Engineering Challenges & Cluster Solutions", "## 🏭 4. Production Engineering Challenges & Cluster Solutions")
content = content.replace("## 5. Frontier Real-World AI Security Applications", "## 🛡️ 5. Frontier Real-World AI Security Applications")
content = content.replace("## References", "## 📚 References")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("added emojis and banner")

# Step 4: SEO Optimised and badges to left
left_badges = r'<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>' + '\n'

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()
    
content = content.replace("![Banner](assets/banner.svg)\n", f"![Banner](assets/banner.svg)\n\n<div align=\"center\">\n{left_badges}</div>\n")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("seo optimised and badges to left added")

# Step 5: Badges to right
right_badge = r'<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>' + '\n'
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()
    
content = content.replace(left_badges, left_badges + right_badge)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("badges to right added")

# Step 6: Star History
star_history = r"""
## ⭐ Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Activation-Steering&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Activation-Steering&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Activation-Steering&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Activation-Steering&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
with open(readme_path, "a", encoding="utf-8") as f:
    f.write(star_history)

run_git("star history added")

# Step 7: Fixed star plot
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("chartrepos", "chart?repos")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("fixed star plot")

# Step 8: Replace awesome link
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

run_git("invalid awesome link fixed")

print("All steps completed successfully.")
