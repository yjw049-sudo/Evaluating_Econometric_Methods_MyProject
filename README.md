# Research Notes: LLM Topic Labels, K-Means Clusters, and Discourse Structure in Parliamentary Speeches

# Research Motivation

This project starts from a methodological question:

> If large language models can classify texts in a way that is close to human topic judgment, does a simpler method such as K-means still have value?

For historical parliamentary speeches, an LLM can often give a plausible topic label. It can classify a speech as agriculture, public finance, transportation, trade, banking, or parliamentary affairs. In this sense, the LLM works much like a human reader. It reads the text and asks:

> What is this speech about?

K-means works differently. It does not understand the text in the same way. It groups texts by word use and lexical similarity. It asks a different question:

> Which texts use similar words and similar language structures?

This difference is the starting point of the project.

The main argument is not that K-means is better than the LLM. The argument is that K-means provides a different kind of information. The LLM identifies the policy topic of a speech. K-means helps reveal how that topic is discussed.

# Initial Research Design

The initial plan was to compare several models.

The original design included:

1. using part of the data to train a K-means model;
2. using AI labels to train supervised models, such as logistic regression or random forest;
3. comparing in-sample and out-of-sample results with the AI labels;
4. identifying texts where the models and the AI labels disagree.

This design treated AI labels as a reference point. The goal was to see whether traditional models could reproduce AI labels, and whether there were systematic differences between model predictions and AI classifications.

However, this design raised an important problem.

K-means clusters are not semantic labels. They are not created to match human topic categories. They are created from patterns in word use. Therefore, it is not appropriate to treat K-means clusters as direct substitutes for AI topic labels.

This changed the direction of the project.

The main question became:

> When K-means and AI labels do not match, is this simply model error, or does the mismatch reveal another structure in the text?

# Main Research Question

The final research question is:

> When LLM-generated labels classify parliamentary speeches by policy content, what additional information can K-means clusters provide about the discourse form of those speeches?

The project studies the relationship between two dimensions of text:

| Dimension | Method | Main question |
|---|---|---|
| Policy topic | LLM / AI label | What is the speech about? |
| Discourse form | K-means cluster | How is the speech written or organized? |

This distinction is central to the project.

# How Much Can We Trust the LLM?

This project does not treat AI labels as ground truth.

A more careful term is:

> provisional semantic reference labels

The AI labels are useful because they give a clear and plausible classification of the broad policy topic. Manual checks suggest that many AI labels are reasonable at the topic level.

For example, if a speech discusses wheat, grain, farm prices, agricultural exports, or food production, the AI usually classifies it as `Agriculture and Food Policy`. If a speech discusses taxation, income tax, public revenue, or debt, the AI usually classifies it as `Public Finance and Taxation`.

However, topic labels do not capture all relevant information in the text. The LLM mainly classifies the speech by content. It may not clearly separate different language forms inside the same topic.

Therefore, the AI labels are useful, but they need to be audited. K-means provides one way to do this.

# Data and Variables

The data consist of Canadian parliamentary speeches from 1920 to 1949.

The main variables are:

| Variable | Meaning |
|---|---|
| `speechtext` | cleaned text used for modeling |
| `speechtext_original` | original speech text used for close reading |
| `year` | year of the speech |
| `topic` | original topic description |
| `category` | AI-generated policy topic label |
| `kmeans_cluster` | K-means cluster assignment |
| `cluster_type` | broader discourse type assigned to the cluster |
| `historical_period` | period group |

The historical periods are:

| Period | Years |
|---|---|
| `postwar_1920s` | 1920–1929 |
| `depression_1930s` | 1930–1938 |
| `wartime` | 1939–1945 |
| `postwar_reconstruction` | 1946–1949 |

# Method

The speeches are represented with TF-IDF features. A K-means model with 20 clusters is then applied to the corpus.

The main K-means setting is:

    KMeans(
        n_clusters=20,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )

Each cluster is interpreted using:

1. top TF-IDF terms;
2. the distribution of AI labels inside the cluster;
3. cross-tabulations between AI labels and K-means clusters;
4. representative original speeches;
5. historical interpretation.

The 20 clusters are not assumed to be true topic categories. They are interpreted as lexical and discourse structures.

# Comparing AI Labels and K-Means Clusters

The project uses two main comparisons.

# AI Category to K-Means Cluster

This comparison asks:

> For each AI category, how are its speeches distributed across K-means clusters?

This helps answer whether an AI topic label is internally homogeneous.

If one AI category is spread across many clusters, this suggests that the AI label contains several subtopics or discourse forms.

For example, `Agriculture and Food Policy` may include speeches about wheat, farm prices, export rules, legal regulation, relief policy, and parliamentary procedure.

# K-Means Cluster to AI Category

This comparison asks:

> For each K-means cluster, which AI categories appear inside it?

This helps answer whether a cluster corresponds closely to one AI topic.

Some clusters are highly concentrated in one AI category. Other clusters are mixed across many categories.

Both outcomes are important.

A highly concentrated cluster shows that K-means can identify a clear policy vocabulary. A mixed cluster may show a cross-topic discourse form, such as legal language, budget language, or general parliamentary debate.

# Three Types of Relationships

The relationship between AI labels and K-means clusters can be grouped into three types.

# Type 1: Clear Topic Alignment

Some clusters match AI labels very well.

| Cluster | Main vocabulary | Main AI label |
|---:|---|---|
| 19 | tariff, duty, import, protection | International Trade and Tariffs |
| 9 | railway, line, Canadian National | Transportation and Communications |
| 4 | tax, income, revenue, taxation | Public Finance and Taxation |
| 17 | wheat, grain, bushel, wheat board | Agriculture and Food Policy |
| 2 | bank, loan, credit, money | Banking and Monetary Policy |

These cases show that K-means and AI labels align when a policy field has clear and repeated vocabulary.

For example, speeches about tariffs often use words such as tariff, duty, import, and protection. Speeches about wheat often use words such as wheat, grain, bushel, and wheat board. In these cases, lexical similarity and semantic topic judgment point in the same direction.

# Type 2: Substructure Inside AI Labels

Some AI labels are split across several K-means clusters.

This means that the AI label is correct at the broad topic level, but it contains internal variation.

For example, `Agriculture and Food Policy` may include:

- wheat and grain policy;
- farm prices;
- agricultural markets;
- product inspection;
- export regulation;
- legal authority;
- federal-provincial administration;
- general parliamentary debate.

The AI label says that these speeches are about agriculture. K-means shows that they are not written in the same way.

# Type 3: Cross-Topic Discourse Clusters

Some clusters do not match a single AI label. Instead, they appear across many policy areas.

Examples include:

| Cluster | Interpretation |
|---:|---|
| 15 | general parliamentary discourse |
| 5 | legal and legislative language |
| 7 | budgetary and administrative expenditure language |
| 8 | committee and procedural language |
| 18 | federal-provincial or intergovernmental language |

These clusters have low match rates with AI labels because they do not mainly measure policy topic. They measure discourse form.

This is a key finding. A low match rate is not necessarily a failure. It may show that K-means captures a structure that the AI topic label does not capture.

# Key Example: Same Topic, Different Discourse Forms

Manual checks support this interpretation.

The following two speeches both belong to `Agriculture and Food Policy`. Both discuss agricultural product regulation. However, K-means places them in different clusters because their language structures differ.

# Example 1: Legal and Administrative Authority

ID: 570680

Topic: Inspection and Sale Act regulatory powers

K-means: Cluster 5 — legal / administrative regulatory language

Excerpt:

> To-day this power is being taken away from Parliament and it is proposed to give authority to certain officials of the department to control this matter by regulations that shall be adopted by the department.
> [...]
> As far as I can see it is now contemplated to take away jurisdiction from Parliament and vest it in the minister or the officials of his department.
> [...]
> If the provisions of the law as they stand are defective why not make the correction here, instead of requiring Parliament to abandon the right of making regulations for the packing of fruit and vesting the power in the officials of the department.

This speech discusses agricultural products such as apples and berries. However, its main language is not about agricultural production or markets. Its key words are:

- Parliament;
- authority;
- jurisdiction;
- minister;
- officials;
- regulations;
- provisions of the law.

The speech frames agricultural regulation as a question of parliamentary power and administrative authority. Therefore, it is reasonable that K-means places it in a legal and administrative language cluster.

# Example 2: Amendment and Clause-Level Procedure

ID: 749345

Topic: amendment of dairy produce export regulations

K-means: Cluster 15 — general parliamentary / amendment discussion language

Excerpt:

> I think the situation can be met by a simpler amendment than that suggested by the hon. member. If we strike out the first seven words in paragraph (g), that will meet the whole situation. The minister will then be given power to withhold export certificates because the certificates are for export.
> [...]
> Let me read this again in a different way:
> If such butter and cheese has no export certificate issued, it cannot be exported.
> Refusal to grade is not needed at all.

This speech also discusses agricultural product regulation. It refers to butter, cheese, export certificates, grading, and export rules.

However, its language is focused on amendment and clause-level discussion. Important phrases include:

- simpler amendment;
- strike out the first seven words;
- paragraph (g);
- Let me read this again;
- not needed at all.

The speech is less about the general issue of administrative authority and more about the technical language of parliamentary amendment. Therefore, K-means places it in a different discourse cluster.

# Interpretation of the Examples

These examples show that disagreement between AI labels and K-means clusters does not always mean error.

The AI label is reasonable because both speeches are about agricultural regulation.

The K-means classification is also reasonable because the two speeches use different language structures.

This supports the main argument:

> AI labels classify policy content. K-means clusters reveal discourse form.

# From Clusters to Discourse Types

To make the 20 clusters easier to interpret, they are grouped into three discourse types.

| Discourse type | Clusters | Meaning |
|---|---|---|
| `substantive_policy_discourse` | 0, 1, 2, 4, 6, 9, 10, 11, 13, 14, 16, 17, 19 | concrete policy content |
| `institutional_procedural_discourse` | 5, 7, 8, 18 | legal, budgetary, committee, administrative, or federal-provincial language |
| `general_parliamentary_discourse` | 3, 12, 15 | general debate, party politics, government-opposition language, and broad parliamentary rhetoric |

This grouping is interpretive. It is based on top terms, representative texts, and the relationship between clusters and AI labels.

It should not be described as an automatic discovery of true discourse types. A more careful statement is:

> The 20 K-means clusters are grouped into three analytically defined discourse types.

# Time Variation

After defining the three discourse types, the project examines whether their shares change over time.

This step asks:

> Do changes in discourse type suggest changes in how policy topics were discussed and institutionally processed in Parliament?

For example, if institutional or procedural language increases within agriculture, this may suggest that agricultural issues were more often discussed through bills, amendments, administrative authority, relief arrangements, or federal-provincial coordination.

If substantive policy discourse increases, this may suggest more direct discussion of concrete policy objects, such as wheat, grain, prices, production, and food supply.

This analysis is descriptive. It does not claim causal identification.

A careful interpretation is:

> Changes in discourse-type shares provide descriptive evidence of shifts in how a policy area was discussed in Parliament.

# Corpus-Level Pattern

At the full-corpus level, general parliamentary discourse remains large across the whole period.

The wartime period shows an increase in substantive policy discourse and a decline in institutional/procedural discourse.

This suggests that parliamentary debate during World War II became more concentrated around concrete policy content, such as production, finance, military affairs, supply, and state administration. However, general parliamentary discourse remained important.

This full-corpus pattern provides a baseline for the topic-specific analysis.

# Case Study 1: Agriculture and Food Policy

Agriculture is the clearest case.

| Period | Speech count | Substantive % | General % | Institutional % |
|---|---:|---:|---:|---:|
| postwar_1920s | 4,935 | 39.39 | 40.65 | 19.96 |
| depression_1930s | 5,546 | 41.80 | 31.25 | 26.96 |
| wartime | 5,575 | 60.41 | 25.17 | 14.42 |
| postwar_reconstruction | 3,211 | 59.58 | 25.47 | 14.95 |

The pattern is clear.

In the 1920s, agricultural speeches were divided between substantive policy discourse and general parliamentary discourse.

During the Depression period, institutional/procedural discourse increased. This may reflect the importance of relief, legal arrangements, administrative programs, and federal-provincial coordination.

During wartime and postwar reconstruction, substantive policy discourse rose to about 60 percent. This suggests that agricultural debate became more focused on concrete policy content, such as wheat, grain, prices, markets, production, and food supply.

# Case Study 2: Transportation and Communications

Transportation and communications follow a different pattern.

| Period | Speech count | Substantive % | General % | Institutional % |
|---|---:|---:|---:|---:|
| postwar_1920s | 9,700 | 51.16 | 28.15 | 20.68 |
| depression_1930s | 6,564 | 45.86 | 29.59 | 24.56 |
| wartime | 2,418 | 47.68 | 32.75 | 19.56 |
| postwar_reconstruction | 2,893 | 39.96 | 35.81 | 24.23 |

Transportation was already highly substantive in the 1920s. This likely reflects the importance of concrete infrastructure issues, such as railways, ports, shipping, freight rates, and postal services.

In the postwar reconstruction period, substantive discourse declined, while general and institutional discourse increased. This may reflect a shift toward broader questions of national infrastructure, regulation, aviation, broadcasting, and communications governance.

This case shows that not all policy areas follow the same wartime or postwar pattern.

# Case Study 3: Public Finance and Taxation

Public finance and taxation show another pattern.

| Period | Speech count | Substantive % | General % | Institutional % |
|---|---:|---:|---:|---:|
| postwar_1920s | 4,762 | 34.88 | 34.42 | 30.70 |
| depression_1930s | 3,413 | 30.06 | 35.83 | 34.10 |
| wartime | 5,158 | 44.71 | 37.20 | 18.09 |
| postwar_reconstruction | 3,751 | 45.72 | 36.04 | 18.24 |

The main change is an increase in substantive policy discourse during wartime and postwar reconstruction, together with a decline in institutional/procedural discourse.

However, general parliamentary discourse remains high throughout the period. This is important. Fiscal issues are inherently political. Taxation, debt, public spending, and fiscal burden are closely tied to government responsibility and party debate.

# Main Findings

The main findings are:

1. AI labels and K-means clusters are not substitutes.
2. AI labels are useful for broad policy topic classification.
3. K-means clusters are useful for detecting lexical and discourse structures.
4. When a policy topic has clear specialized vocabulary, AI labels and K-means clusters align well.
5. When a text uses legal, procedural, budgetary, administrative, or general parliamentary language, K-means often cuts across AI labels.
6. These mismatches can reveal meaningful discourse structures.
7. These discourse structures change over time.
8. Different policy areas show different patterns of change.

# Methodological Contribution

This project offers a way to audit AI-generated labels.

The workflow is:

1. Use an LLM to generate broad policy topic labels.
2. Use TF-IDF and K-means to identify unsupervised lexical clusters.
3. Compare AI labels and K-means clusters.
4. Separate clear topic clusters from mixed discourse clusters.
5. Use representative texts to validate the interpretation.
6. Group clusters into broader discourse types.
7. Study how these discourse types vary across topics and time.

The key methodological point is:

> LLM labels can be useful and plausible, but they should not be treated as complete descriptions of the text.

K-means remains useful because it can reveal structures that the LLM topic label does not explicitly encode.

# Historical Contribution

The project also has a historical interpretation.

It shows that Canadian parliamentary speeches from 1920 to 1949 were organized by both policy topic and discourse form.

The comparison across agriculture, transportation, and public finance shows that different policy fields were discussed in different ways over time.

- Agriculture became more substantively policy-focused during wartime and postwar reconstruction.
- Transportation and communications became more general and institutional in the postwar reconstruction period.
- Public finance became more substantive during wartime and postwar reconstruction, but remained strongly political.

These patterns suggest that discourse form can be used as a descriptive measure of how Parliament processed different policy areas.

# Limitations

Several limitations are important.

First, the AI labels are not human-validated ground truth. They are provisional semantic reference labels.

Second, K-means is based on TF-IDF. It captures lexical similarity, not deep semantic meaning.

Third, the three discourse types are researcher-defined categories. They are based on top terms, cross-tabulations, and representative texts.

Fourth, the time trends are descriptive. They should not be interpreted as causal estimates.

Fifth, figures and tables are not enough. Representative text reading is necessary to support the interpretation.

# Recommended Framing

This project should not be framed as:

> Which model is more accurate, the LLM or K-means?

A better framing is:

> What does K-means reveal that LLM topic labels do not show?

Or:

> How can unsupervised lexical clusters help audit and interpret AI-generated topic labels?

The final argument can be stated simply:

> AI labels identify what Parliament was talking about. K-means clusters help reveal how Parliament talked about it.

# Final Summary

This project shows that LLMs and K-means are complementary tools for historical text analysis.

The LLM provides a plausible semantic topic label. K-means provides a reproducible measure of lexical and discourse similarity.

When the two methods agree, the policy topic usually has clear specialized vocabulary. When they disagree, the mismatch may reveal a meaningful discourse structure, such as legal language, procedural language, budget language, administrative language, or general parliamentary debate.

The main contribution is therefore not to prove that one model is better than the other. The main contribution is to show that AI topic labels can be audited and enriched through unsupervised analysis.

In short:

> The LLM tells us the topic.
> K-means helps us see the form of discussion.