<h1>Conformalized Decision Risk Assessment</h1>


<h2>Abstract</h2>

High-stakes decisions in domains such as healthcare, energy, and public policy are often made by human experts using domain knowledge and heuristics, yet are increasingly supported by predictive and optimization-based tools.
A dominant approach in operations research is the predict-then-optimize paradigm, where a predictive model estimates uncertain inputs, and an optimization model recommends a decision.
However, this approach often lacks interpretability and can fail under distributional uncertainty -- particularly when the outcome distribution is multi-modal or complex -- leading to brittle or misleading decisions.
In this paper, we introduce CREDO, a novel framework that quantifies, for any candidate decision, a distribution-free upper bound on the probability that the decision is suboptimal.
By combining inverse optimization geometry with conformal prediction and generative modeling, CREDO produces risk certificates that are both statistically rigorous and practically interpretable.
This framework enables human decision-makers to audit and validate their own decisions under uncertainty, bridging the gap between algorithmic tools and real-world judgment.
