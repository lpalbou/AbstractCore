# **Comprehensive Guide to State-of-the-Art Graph and Nested Graph Compression Approaches (December 2025\)**

## **1\. Introduction: The Graph Data Crisis and the Bifurcation of Compression**

As of December 2025, the digital ecosystem is characterized by an overwhelming deluge of structured data. With Internet of Things (IoT) devices alone contributing to a global data sphere exceeding 180 zettabytes annually, the management of graph-structured data—representing everything from social networks and biological interactions to cyber-security logs and software dependencies—has become a critical bottleneck in computational infrastructure.1 The challenge is no longer merely one of storage capacity; it is one of computational tractability and efficient retrieval. The sheer magnitude of modern graphs, often containing billions of nodes and trillions of edges, renders traditional processing pipelines prohibitively expensive and slow. Consequently, the field of graph compression has evolved rapidly, moving away from a monolithic focus on minimizing bits-per-edge toward a diversified landscape of specialized paradigms.  
By late 2025, a clear bifurcation has emerged in graph compression methodologies, driven by the distinct needs of **storage efficiency** versus **learning efficiency**.  
On one hand, we witness the maturity of **Graph Condensation (GC)**. This paradigm, primarily serving the machine learning community, does not seek to compress a graph in the lossless information-theoretic sense. Instead, it aims to synthesize a "proxy" graph—often less than 1% of the original size—that retains the rich semantic information required to train Graph Neural Networks (GNNs) to full accuracy.2 This is a lossy compression of topology but a lossless compression of "trainability." The year 2025 has seen this field transition from experimental gradient matching to robust, theoretically grounded frameworks that address cross-architecture generalization and privacy.4  
On the other hand, **Structural and Storage Compression** remains vital for database systems and archival. Here, the goal is lossless or near-lossless reduction of the physical footprint while maintaining the ability to query the graph efficiently. Innovations in 2025, such as **Zuckerli** and **OpenZL**, have pushed the boundaries of encoding density and hardware acceleration, challenging long-standing standards like WebGraph.5 Furthermore, the handling of dynamic data streams has necessitated new hierarchical summarization techniques like **HIGGS**, which allow for real-time compression of rapidly evolving networks without the need for periodic reconstruction.7  
This report provides an exhaustive analysis of these state-of-the-art (SOTA) approaches. We explore the neural mechanisms behind spectral sparsification, the algorithmic innovations in handling nested and compound graphs, and the theoretical breakthroughs in succinct data structures that are redefining the limits of compact graph representation.

## ---

**2\. Graph Condensation: Synthesizing Proxies for Efficient Learning**

Graph Condensation (GC) represents a paradigm shift in how massive graph datasets are utilized for Deep Learning. Traditional methods like graph coarsening or sampling select a subset of existing nodes and edges. In contrast, GC synthesizes a completely new, smaller set of node features and structures that may not exist in the original graph but capture its "training physics." As of 2025, this field has matured into a cornerstone of scalable GNN training, with new methods addressing the limitations of early bi-level optimization.

### **2.1 The Evolution of Optimization Strategies**

The fundamental objective of GC is to generate a condensed graph $\\mathcal{S}$ that minimizes the gap between a model trained on $\\mathcal{S}$ and one trained on the full graph $\\mathcal{T}$. The evolution of this objective function has driven the major advancements in 2024 and 2025\.

#### **2.1.1 One-Step Gradient Matching: DosCond**

Early condensation methods were plagued by the high computational cost of matching training trajectories over hundreds of epochs. A significant optimization in the 2025 landscape is **DosCond (Discrete One-Step Condensation)**. DosCond circumvents the expensive nested optimization of its predecessors by performing gradient matching for only a single step at the initialization phase.8  
The innovation of DosCond lies in its probabilistic modeling of discrete structures. It treats the adjacency matrix of the condensed graph as a learnable probabilistic model, allowing gradients to flow through the discrete graph structure generation process. This "one-step" strategy drastically reduces the pre-computation time required to generate the condensed graph—achieving speeds up to 15 times faster than multi-step matching methods—while retaining high fidelity. Empirical benchmarks demonstrate that DosCond can reduce dataset size by 90% while retaining approximately 98% of the classification performance on standard benchmarks like Citeseer and Reddit.8 This efficiency makes it a standard baseline for rapid condensation tasks where training time is a primary constraint.

#### **2.1.2 Addressing Spectral Shift: SGDD**

While gradient matching captures the optimization path of a GNN, it often fails to preserve the global structural properties of the graph, leading to a phenomenon known as **Laplacian Energy Distribution (LED) shift**. This shift means that the condensed graph may train a GNN to high accuracy on the specific task it was condensed for (e.g., node classification) but fail catastrophically if the model architecture changes or if the task shifts to anomaly detection or link prediction.  
**SGDD (Structure-broadcasting Graph Dataset Distillation)**, a leading method in 2025, addresses this by explicitly "broadcasting" the structural information of the original graph into the synthesis process.10 SGDD operates by aligning the spectral properties (eigenvalues of the Laplacian) of the synthetic graph with the original. By minimizing the LED shift, SGDD ensures that the condensed graph retains the frequency components of the original signal—preserving both the low-frequency signals typical of community structures and the high-frequency signals indicative of anomalies or boundaries.

* **Performance Impact:** This spectral alignment results in state-of-the-art performance across diverse datasets. For instance, on the YelpChi dataset, SGDD maintains 98.6% of the original test accuracy despite a 1,000$\\times$ reduction in graph scale.10  
* **Generalization:** Crucially, the reduction in LED shift (observed to be between 17% and 31%) translates to superior cross-architecture generalization, allowing graphs condensed by SGDD to be used effectively to train GNNs different from the one used during the condensation process.10

### **2.2 Structure Learning and Topology Reconstruction**

A critical challenge in GC is defining the topology of the synthetic nodes. Early methods used simple MLPs to predict edges from node features, which often resulted in synthetic graphs that ignored complex inter-class correlations (heterophily).

#### **2.2.1 GCSR: Self-Expressive Reconstruction**

**GCSR (Graph Condensation via Self-expressive Graph Structure Reconstruction)** represents the 2025 SOTA for handling complex topologies. Unlike methods that treat edge generation as a black-box neural process, GCSR posits that the structure of the condensed graph should be **self-expressive**—meaning each node's feature representation can be reconstructed as a linear combination of its neighbors.3  
This dictionary-learning approach forces the synthesized graph to respect the underlying manifold of the data. Visualizations using t-SNE confirm that GCSR preserves distinct class boundaries while maintaining essential connections between different classes (heterophily), whereas baseline methods often produce "collapsed" clusters where class distinctions are blurred.11 This makes GCSR particularly effective for datasets where the graph structure carries significant information independent of node features.

#### **2.2.2 Structure-Free Approaches**

An alternative school of thought that has gained traction in 2025 is **Structure-Free Graph Condensation (SFGC)**. These methods argue that for certain high-homophily datasets, the explicit topology can be distilled entirely into the node features, or "graph-free" data, which is then fed into a GNN as if it were a graph (potentially using an identity matrix or a fully connected implicit structure).3

* **Mechanism:** SFGC matches the training trajectories of a GNN on the original graph to a model trained on a set of isolated synthetic nodes.  
* **Trade-off:** While extremely efficient (eliminating the $O(N^2)$ complexity of adjacency matrix handling), SFGC often struggles with tasks that are heavily dependent on long-range interactions or complex structural motifs, as highlighted by benchmarks in 2025\.14

### **2.3 Benchmarking and the "Denoiser" Effect**

The maturity of the field in 2025 is signaled by the release of comprehensive benchmarking frameworks, most notably **GC-Bench** and **GC4NC**. These frameworks have moved the evaluation of GC beyond simple accuracy metrics.  
**Table 1: Comparison of Major 2025 Graph Condensation Benchmarks**

| Feature | GC-Bench | GC4NC |
| :---- | :---- | :---- |
| **Focus** | Performance, Complexity, Transferability | Robustness, Privacy, NAS, Denoising |
| **Datasets** | 12 (Graph & Node Level) | Node Classification Focus |
| **Key Insight** | Structure-free methods (SFGC) are competitive in accuracy but lack topological interpretability. | Condensation acts as a **denoiser**; condensed graphs are more robust to adversarial attacks than full graphs. |
| **Metrics** | Reduction Rate vs. Accuracy | Membership Inference Attack (MIA) Success Rate; NAS Rank Correlation |

**The Denoiser Insight:** One of the most profound findings from the **GC4NC** benchmark is the "denoising" property of condensation. By forcing the graph into a highly constrained budget (e.g., 0.5% of the original size), the optimization process naturally discards noise, outliers, and redundant features, preserving only the most robust, causal patterns. Consequently, models trained on condensed graphs often exhibit higher robustness to adversarial attacks and better privacy preservation (lower susceptibility to membership inference) than models trained on the full, noisy datasets.15 This reframes GC not just as a compression tool, but as a data sanitization and privacy-enhancing technology.

## ---

**3\. Spectral Sparsification: From Randomized Sampling to Neural Evolution**

While Graph Condensation focuses on learning utility, **Spectral Sparsification** focuses on mathematical fidelity. The goal is to find a sparse subgraph that preserves the quadratic form of the Graph Laplacian $x^T L x$ for all vectors $x$, ensuring that global properties like cut sizes, eigenvalues, and electrical resistance are maintained.

### **3.1 SpecNet: The Neural Sparsification Paradigm**

As of late 2025, the most significant advancement in this domain is the **Spectral Preservation Network (SpecNet)**. Traditional sparsification relies on calculating "effective resistances" for every edge—a computationally expensive process involving linear system solvers ($O(m \\log n)$ or similar). SpecNet replaces this static calculation with a learnable neural architecture.17

#### **3.1.1 The Joint Graph Evolution (JGE) Layer**

The core innovation of SpecNet is the **Joint Graph Evolution (JGE)** layer. In standard GNNs, the graph structure is a fixed input. In SpecNet, the structure is a dynamic variable that evolves alongside the node features.

* **Mechanism:** The JGE layer uses bilinear transformations to reparameterize the graph Laplacian at each layer. It takes the current adjacency $Q\_t$ and features $H\_t$ and produces a new, optimized topology $Q\_{t+1}$.17  
* **Adaptive Rewiring:** This allows the network to "rewire" the graph dynamically, adding useful connections for information flow and pruning redundant ones. This dynamic evolution is crucial for mitigating **oversmoothing**, a common failure mode in deep GNNs where node features become indistinguishable.17

#### **3.1.2 Spectral Concordance Loss**

To ensure the learned graph is actually a spectral sparsifier (and not just an arbitrary sparse graph), SpecNet minimizes a **Spectral Concordance (SC)** loss. This composite loss function enforces:

1. **Laplacian Alignment:** The spectrum of the learned graph must match the original.  
2. **Feature Consistency:** Nodes that are close in the feature space should remain connected (geometry preservation).  
3. **Trace Penalty:** A regularization term that explicitly penalizes the number of edges (via the trace of the adjacency matrix), driving the network toward sparsity.17

### **3.2 Theoretical Breakthroughs: Uniform Sampling Validity**

Parallel to neural approaches, theoretical computer science witnessed a breakthrough at **NeurIPS 2025** regarding the complexity of sparsification. For over a decade, it was believed that **importance sampling** (sampling edges proportional to effective resistance) was strictly necessary for high-quality spectral sparsification.  
However, new proofs have demonstrated that for graphs with **strong clusterability**, simple **Uniform Edge Sampling** is sufficient.19

* **The Structure Ratio:** The validity of uniform sampling is governed by the "Structure Ratio" $\\Upsilon(k) \= \\lambda\_{k+1} / \\rho\_G(k)$, where $\\lambda\_{k+1}$ is the $(k+1)$-th eigenvalue and $\\rho\_G(k)$ is the $k$-way expansion constant.  
* **Implication:** If a graph has well-defined clusters (a large structure ratio), the structural information is distributed evenly enough that random deletion of edges does not destroy the spectral properties. This result is transformative for processing massive clustered networks (like social graphs), as it allows for $O(1)$ sampling per edge without the expensive pre-computation of resistance scores.19

## ---

**4\. Dynamic and Streaming Graph Summarization**

In many 2025 applications—such as real-time fraud detection or cyber-security monitoring—graphs are not static objects but continuous streams of edge insertions and deletions. Compressing these streams requires data structures that support constant-time updates and sub-linear space complexity.

### **4.1 HIGGS: Hierarchy-Guided Graph Stream Summarization**

The state-of-the-art system for this challenge, presented at **ICDE 2025**, is **HIGGS (Hierarchy-Guided Graph Stream Summarization)**.7 HIGGS outperforms previous stream summaries like TCM, GSS, and Horae by adopting a **bottom-up, item-based hierarchy**.

#### **4.1.1 Architecture and Mechanism**

HIGGS is structured similarly to an aggregated B-tree, designed to handle the high velocity of edge updates.

* **Leaf Nodes:** These store the actual edge information for the most recent time intervals. To achieve compression, HIGGS uses **fingerprinting** (hashing node IDs to compact bit signatures) and compressed adjacency matrices within the leaves.22  
* **Hierarchy:** Unlike top-down methods that require global knowledge to partition the graph (impossible in a stream), HIGGS builds the hierarchy upwards. As leaf nodes fill up, they are aggregated into parent nodes that summarize the temporal and structural intervals. This allows the structure to grow dynamically with the stream.22  
* **Performance:**  
  * **Space:** HIGGS maintains linear space complexity $O(|E|)$ relative to the active window.  
  * **Accuracy:** It improves query accuracy (e.g., for edge weight estimation or reachability) by over **3 orders of magnitude** compared to sketch-based baselines.7  
  * **Latency:** The bottom-up insertion logic allows for $O(1)$ update time and reduces query latency by nearly two orders of magnitude.7

### **4.2 Dynamic Sparse Certificates**

For applications requiring strict connectivity guarantees (e.g., maintaining network reliability), "sketches" are insufficient. Instead, systems maintain **Sparse Certificates**—subgraphs that guarantee $k$-connectivity (if the certificate is $k$-connected, the full graph is $k$-connected).

* **Fully Dynamic Maintenance:** Innovations in 2025 have yielded algorithms that maintain these certificates under both edge insertions *and* deletions. The key advancement is a **localized search strategy** for edge deletions. Instead of rescanning the entire graph to check if a deleted edge breaks connectivity, the algorithm maintains auxiliary structures that identify "replacement paths" in the local neighborhood, significantly improving theoretical and practical runtimes.23  
* **Transitive Reduction:** Similarly, for directed graphs, dynamic algorithms now maintain the **transitive reduction** (the minimal set of edges required to preserve reachability) without full recomputation. This is critical for visualizing dependency graphs in real-time, where redundant edges create visual clutter.24

## ---

**5\. Static Storage and Retrieval: The Engineering Frontier**

When the objective is purely to reduce the storage footprint of a static graph while enabling fast traversal (e.g., for web search indices), the focus shifts to efficient encoding of adjacency lists.

### **5.1 Zuckerli vs. WebGraph: The New Standard**

For nearly two decades, **WebGraph** (BVGraph) was the de facto standard for compressing web-scale graphs. However, 2025 has cemented **Zuckerli** as the superior alternative for static compression density.5

* **Algorithmic Differences:** While WebGraph relies on reference coding (representing a node's neighbors as differences from a prototype node) and Gamma/Zeta codes, **Zuckerli** leverages **Asymmetrical Numeral Systems (ANS)** and advanced context modeling.  
* **Context Modeling:** Zuckerli's context model predicts the likelihood of a link based on local patterns more aggressively than WebGraph. This allows the entropy coder (ANS) to pack bits more efficiently.  
* **Comparison:** Empirical evaluations on billion-edge graphs show that Zuckerli produces files **10% to 29% smaller** than WebGraph.5 Crucially, it matches WebGraph's decompression speed and supports random access to adjacency lists, making it a viable drop-in replacement for large-scale mining systems.

### **5.2 OpenZL: Format-Aware Tensor Compression**

In October 2025, Meta introduced **OpenZL**, a framework that addresses the compression of *structured* graph data (e.g., feature matrices, tensors) in production environments.6

* **Format Awareness:** Unlike generic compressors (Zstd, Gzip) that see data as a byte stream, OpenZL allows users to define a schema (a graph of transformation functions). It parses the input into semantic tokens (e.g., "Node ID", "Embedding Vector") and applies distinct compression strategies to each.6  
* **The "Trainer":** OpenZL includes an offline optimization component called the "Trainer." It clusters data chunks to identify statistical properties and searches for the optimal sequence of transformations. For floating-point tensors (common in GNNs), this results in **21% better compression** than standard baselines.27  
* **Universal Decoder:** Despite the bespoke compression configurations, OpenZL generates a single "universal decoder" binary. This eliminates the operational complexity of managing different decompressors for different graph schemas.6

### **5.3 Hardware-Aligned Reordering**

Compression is also being adapted for hardware acceleration. In 2025, novel **node reordering algorithms** have been designed specifically to induce **N:M sparsity** patterns (e.g., 2:4 sparsity) required by NVIDIA's Sparse Tensor Cores.

* **Impact:** By reordering the graph to minimize sparsity violations (losslessly), these algorithms enable GNNs to utilize tensor core acceleration that was previously limited to dense vision models. This yields up to **43$\\times$ speedups** on Sparse Matrix-Matrix Multiplication (SpMM) kernels.28

## ---

**6\. Nested and Compound Graph Compression**

Nested graphs (or compound graphs) are hierarchical structures where a node can contain another graph (e.g., a software module containing functions). Compressing these requires preserving the containment hierarchy.

### **6.1 Modular Decomposition for Reachability**

**Modular Decomposition** is the primary technique for compressing these structures. A "module" is a set of nodes that share the exact same neighborhood outside the set.

* **Compression Mechanism:** By collapsing a module into a single representative node, the graph size is drastically reduced. If a module of size $k$ connects to $p$ external neighbors, modular decomposition replaces $k \\times p$ edges with $p$ edges.29  
* **2025 Application:** Recent work utilizes the **Modular Decomposition Tree (MDT)** for efficient reachability queries. Instead of traversing the massive original graph, queries are routed through the much smaller MDT, enabling fast in-memory analysis of massive dependency networks.30

### **6.2 Grammar-Based Compression**

For graphs with repetitive substructures (like chemical compounds or biological networks), **Grammar-Based Compression** (specifically **Re-Pair** for graphs) is SOTA.

* **Mechanism:** The algorithm iteratively identifies the most frequent pair of adjacent symbols (nodes/edges) and replaces them with a new rule (non-terminal). This recursively builds a grammar that represents the graph.31  
* **Query Efficiency:** A unique advantage is that certain queries, such as path existence, can be evaluated directly on the compressed grammar in linear time relative to the grammar size (which is logarithmic relative to the graph size).32

### **6.3 Hierarchical Code Summarization**

In the domain of "Code Graphs" (which model software as graphs), **HCGS (Hierarchical Code Graph Summarization)** represents the 2025 state of the art.

* **Bottom-Up Traversal:** HCGS builds summaries from the function level up to the module level.  
* **Vector-Conditioning:** Unlike text summarization, HCGS conditions the high-level summaries on the vector representations of the lower-level graph components. This creates a rich, multi-layered index that allows developers to "zoom in" from a high-level module view to specific function dependencies without loading the full code graph.33

## ---

**7\. Succinct Data Structures: The Theoretical Limit**

Succinct data structures aim to store graphs using space close to the information-theoretic lower bound ($Z$) while supporting queries in optimal time.

### **7.1 Advances in Intersection Graphs**

Significant progress has been made in 2024-2025 regarding **Intersection Graphs** (Interval, Chordal, Path graphs), which are critical in bioinformatics.  
**Table 2: 2025 SOTA Succinct Bounds**

| Graph Class | Previous Bound | 2025 SOTA Bound | Supported Queries | Source |
| :---- | :---- | :---- | :---- | :---- |
| **Path Graphs** | $n \\log n \+ o(n \\log n)$ | $(2+\\epsilon)n \\log n$ | Distance, Shortest Path ($O(\\frac{\\log n}{\\log \\log n})$) | 34 |
| **Chordal Graphs** | Generic | $n \\log n \+ O(n)$ | Adjacency ($O(1)$) | 35 |

**Key Innovation:** The new data structure for **Path Graphs** is the first compact distance oracle for this class. It allows for shortest path queries to be answered almost instantly without storing the full $O(n^2)$ distance matrix, a critical enabling technology for genome assembly algorithms that operate on massive path graphs.34

## ---

**8\. Conclusion**

The landscape of graph compression in December 2025 is defined by specialization. The "one-size-fits-all" approach of generic compression has been replaced by distinct paradigms tailored to specific computational goals.

1. **For Machine Learning:** The priority is **Semantic Fidelity**. **DosCond** and **GCSR** provide the ability to distill massive datasets into tiny, trainable proxies, with **SGDD** ensuring these proxies generalize across architectures. The **GC4NC** benchmark has further revealed that this process acts as a privacy-preserving denoiser.  
2. **For Network Science:** The priority is **Spectral Fidelity**. **SpecNet** brings the power of deep learning to sparsification, learning optimal topologies dynamically, while **Uniform Sampling** has been theoretically vindicated for clustered real-world graphs.  
3. **For Real-Time Systems:** The priority is **Update Velocity**. **HIGGS** demonstrates that hierarchical summarization can achieve constant-time updates and linear space, solving the challenge of high-velocity graph streams.  
4. **For Infrastructure:** The priority is **Storage Density and Throughput**. **Zuckerli** and **OpenZL** leverage advanced entropy coding (ANS) and format-awareness to minimize footprint and maximize bandwidth, with optimizations like **PG-Fuse** removing filesystem bottlenecks.

In summary, the state of the art has moved beyond simply compressing data; it is now about **compressing for purpose**—whether that purpose is training a neural network, proving a connectivity theorem, or querying a live data stream.

#### **Works cited**

1. Lossless Compression of Time Series Data: A Comparative Study \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2510.07015v1](https://arxiv.org/html/2510.07015v1)  
2. \[2401.11720\] Graph Condensation: A Survey \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/2401.11720](https://arxiv.org/abs/2401.11720)  
3. Graph Condensation Approaches \- Emergent Mind, accessed December 18, 2025, [https://www.emergentmind.com/topics/graph-condensation-approaches](https://www.emergentmind.com/topics/graph-condensation-approaches)  
4. \[2406.16715\] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/2406.16715](https://arxiv.org/abs/2406.16715)  
5. (PDF) Zuckerli: A New Compressed Representation for Graphs \- ResearchGate, accessed December 18, 2025, [https://www.researchgate.net/publication/347190689\_Zuckerli\_A\_New\_Compressed\_Representation\_for\_Graphs](https://www.researchgate.net/publication/347190689_Zuckerli_A_New_Compressed_Representation_for_Graphs)  
6. Introducing OpenZL: An Open Source Format-Aware Compression Framework, accessed December 18, 2025, [https://engineering.fb.com/2025/10/06/developer-tools/openzl-open-source-format-aware-compression-framework/](https://engineering.fb.com/2025/10/06/developer-tools/openzl-open-source-format-aware-compression-framework/)  
7. HIGGS: HIerarchy-Guided Graph Stream Summarization \- IEEE Xplore, accessed December 18, 2025, [https://ieeexplore.ieee.org/document/11112885/](https://ieeexplore.ieee.org/document/11112885/)  
8. amazon-science/doscond \- GitHub, accessed December 18, 2025, [https://github.com/amazon-science/doscond](https://github.com/amazon-science/doscond)  
9. Condensing Graphs via One-Step Gradient Matching \- OpenReview, accessed December 18, 2025, [https://openreview.net/pdf?id=WVpAeZd6ooY](https://openreview.net/pdf?id=WVpAeZd6ooY)  
10. RingBDStack/SGDD: Code for SGDD \- GitHub, accessed December 18, 2025, [https://github.com/RingBDStack/SGDD](https://github.com/RingBDStack/SGDD)  
11. Graph Data Condensation via Self-expressive Graph Structure Reconstruction, accessed December 18, 2025, [https://jhc.sjtu.edu.cn/\~gjzheng/paper/kdd2024\_GCSR/kdd2024\_GCSR\_paper.pdf](https://jhc.sjtu.edu.cn/~gjzheng/paper/kdd2024_GCSR/kdd2024_GCSR_paper.pdf)  
12. Graph Data Condensation via Self-expressive Graph Structure Reconstruction \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2403.07294v2](https://arxiv.org/html/2403.07294v2)  
13. Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data, accessed December 18, 2025, [https://papers.neurips.cc/paper\_files/paper/2023/file/13183a224208671a6fc33ba1aa661ec4-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2023/file/13183a224208671a6fc33ba1aa661ec4-Paper-Conference.pdf)  
14. \[2407.00615\] GC-Bench: An Open and Unified Benchmark for Graph Condensation \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/2407.00615](https://arxiv.org/abs/2407.00615)  
15. GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights \- OpenReview, accessed December 18, 2025, [https://openreview.net/pdf?id=ZhxeUImT89](https://openreview.net/pdf?id=ZhxeUImT89)  
16. GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2406.16715v2](https://arxiv.org/html/2406.16715v2)  
17. Spectral Neural Graph Sparsification \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2510.27474v1](https://arxiv.org/html/2510.27474v1)  
18. \[2510.27474\] Spectral Neural Graph Sparsification \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/2510.27474](https://arxiv.org/abs/2510.27474)  
19. Structure-Aware Spectral Sparsification via Uniform Edge Sampling \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2510.12669v1](https://arxiv.org/html/2510.12669v1)  
20. \[2510.12669\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/2510.12669](https://arxiv.org/abs/2510.12669)  
21. Structure-Aware Spectral Sparsification via Uniform Edge Sampling | OpenReview, accessed December 18, 2025, [https://openreview.net/forum?id=Z4eFqgYbha](https://openreview.net/forum?id=Z4eFqgYbha)  
22. \[Literature Review\] HIGGS: HIerarchy-Guided Graph Stream Summarization \- Moonlight, accessed December 18, 2025, [https://www.themoonlight.io/en/review/higgs-hierarchy-guided-graph-stream-summarization](https://www.themoonlight.io/en/review/higgs-hierarchy-guided-graph-stream-summarization)  
23. Preserving K-Connectivity in Dynamic Graphs \- IEEE Xplore, accessed December 18, 2025, [https://ieeexplore.ieee.org/document/11113033/](https://ieeexplore.ieee.org/document/11113033/)  
24. Fully Dynamic Algorithms for Transitive Reduction \- EPrints, accessed December 18, 2025, [http://eprints.cs.univie.ac.at/8468/1/LIPIcs.ICALP.2025.92.pdf](http://eprints.cs.univie.ac.at/8468/1/LIPIcs.ICALP.2025.92.pdf)  
25. (PDF) Fully Dynamic Algorithms for Transitive Reduction \- ResearchGate, accessed December 18, 2025, [https://www.researchgate.net/publication/391219591\_Fully\_Dynamic\_Algorithms\_for\_Transitive\_Reduction](https://www.researchgate.net/publication/391219591_Fully_Dynamic_Algorithms_for_Transitive_Reduction)  
26. Zuckerli: A New Compressed Representation for Graphs \- arXiv, accessed December 18, 2025, [https://arxiv.org/pdf/2009.01353](https://arxiv.org/pdf/2009.01353)  
27. Lightning Talk: Model Checkpoint Compression With OpenZL \- Nick Terrell & Teja Rao, Meta, accessed December 18, 2025, [https://www.youtube.com/watch?v=Drnt6rkeA9c](https://www.youtube.com/watch?v=Drnt6rkeA9c)  
28. Accelerating GNNs on GPU Sparse Tensor Cores through N:M Sparsity-Oriented Graph Reordering \- Research, accessed December 18, 2025, [https://research.csc.ncsu.edu/picture/publications/papers/ppopp25.pdf](https://research.csc.ncsu.edu/picture/publications/papers/ppopp25.pdf)  
29. Edge Compression Techniques for Visualization of Dense Directed Graphs \- Microsoft, accessed December 18, 2025, [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/edgecompression\_infovis2013.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/edgecompression_infovis2013.pdf)  
30. (PDF) Modular Decomposition-Based Graph Compression for Fast Reachability Detection, accessed December 18, 2025, [https://www.researchgate.net/publication/335521870\_Modular\_Decomposition-Based\_Graph\_Compression\_for\_Fast\_Reachability\_Detection](https://www.researchgate.net/publication/335521870_Modular_Decomposition-Based_Graph_Compression_for_Fast_Reachability_Detection)  
31. Grammar-Based Graph Compression | Request PDF \- ResearchGate, accessed December 18, 2025, [https://www.researchgate.net/publication/316234938\_Grammar-Based\_Graph\_Compression](https://www.researchgate.net/publication/316234938_Grammar-Based_Graph_Compression)  
32. \[1704.05254\] Grammar-Based Graph Compression \- arXiv, accessed December 18, 2025, [https://arxiv.org/abs/1704.05254](https://arxiv.org/abs/1704.05254)  
33. Hierarchical Graph-Based Code Summarization for Enhanced Context Retrieval \- arXiv, accessed December 18, 2025, [https://arxiv.org/html/2504.08975v1](https://arxiv.org/html/2504.08975v1)  
34. Succinct Data Structures for Path Graphs and Chordal Graphs Revisited \- Dalhousie University, accessed December 18, 2025, [https://web.cs.dal.ca/\~mhe/publications/dcc24\_pathchordalgraphs.pdf](https://web.cs.dal.ca/~mhe/publications/dcc24_pathchordalgraphs.pdf)  
35. Succinct Data Structures for Chordal Graph with Bounded Leafage or Vertex Leafage, accessed December 18, 2025, [https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.WADS.2025.35](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.WADS.2025.35)