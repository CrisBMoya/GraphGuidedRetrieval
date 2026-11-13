# Beyond Naive Text-to-SQL: Building a Graph-Guided, Polyglot Data Retrieval Agent with LangGraph, Neo4j, and Polars

## 1. The Origin: Why "Context" Isn't Enough (and Why We Wanted to Automate Domain Expertise)

In the early stages of this project, leadership and I were exploring ambitious product directions—often generating ideas faster than existing tooling or planning bandwidth could immediately support. Yet from that rapid ideation, a genuinely compelling engineering vision emerged.

At its core, our goal was simple to state and deceptively hard to engineer: **we wanted to remove the human domain-expert bottleneck from routine data analysis.**

We all know the standard data lifecycle: you learn a business, gather telemetry, monitor metrics, detect anomalies, and predict outcomes. Organizations collect data because data drives decisions. **But raw data is useless without context.**

A `1` is always a `1`. If you add `"kg"` next to it, it starts to mean something. If you add time—`"1 kg at day 30"`—you now have information. Yet if I tell you *"1 kg at day 30"* and you aren't in the agricultural, livestock, or industrial production business, you still don't know whether to celebrate a breakthrough or panic over an anomaly.

To know *what information means*, you need more than units and timestamps—you need a **domain expert**. In large-scale biological and industrial monitoring systems, operational data isn't sitting neatly in a single, self-explanatory table. It is fragmented across:

* **Relational databases (`PostgreSQL` and `MySQL`)** storing production cycles, daily mortality/loss rates, growth weights, and conversion ratios.
* **Time-series clusters (`InfluxDB`)** streaming high-frequency IoT telemetry like storage silo weights every minute.
* **Reference benchmark tables** (such as standard growth curves per genetic line or cohort) that define whether `"1 kg at day 30"` is on-target or critical.

In this post, I’ll walk through the architecture, design constraints, and candid lessons learned building this **Polyglot Data Retrieval & Query Engine**—an agentic GraphRAG pipeline I built with **LangGraph**, **Neo4j Vector Indexes**, **OpenAI (`gpt-4o` & `text-embedding-3-large`)**, and **Polars** that translates messy, multilingual natural language into verified, cross-database queries and domain-aware insights.

---

## 2. The Rules of Engagement (Our Engineering Conditions)

Before I wrote a single line of orchestration code, we agreed on three strict conditions for the concept:

1. **An LLM-First Architecture:** As often happens in applied R&D, the mandate was to explore the ceiling of modern LLM capabilities rather than build another rigid rule-based parser. Our objective was to push agentic orchestration to its limits on real-world operational data.
2. **A Numeric Answer Is Only the Baseline:** Plenty of BI tools or Text-to-SQL scripts can spit out `"1"`, `"yesterday"`, or `"20 liters"` and be technically correct. We wanted a complete answer with contextual insights—the way an experienced operations specialist or analyst would answer. Being numerically accurate was just table stakes.
3. **Zero Hardcoded Domain Context:** On this point, we were adamant. Even the earliest prototype could **not** rely on:
   * Hardcoded domain text stuffed into system prompts.
   * Hardcoded business facts written around the code.
   * Hardcoded or pre-baked results.
   Every available KPI, entity relationship, query template, and domain benchmark had to be discovered dynamically from the knowledge base.

---

## 3. Why Naive Text-to-SQL Hits a Wall

Imagine an operator asking in Spanish—potentially with typos, shorthand, or colloquial facility names:
> *"¿Qué datos de peso de silo hay para Valle Norte, unidad 1 y lote 224?"*  
> *(What silo weight data is available for North Valley, unit 1, and batch cycle 224?)*

Or a site manager asking:
> *"What is the latest conversion value for Sector Alpha and Unit 3 with Batch 108?"*

Our core objective was to turn any natural language question—independent of language, nicknames, layman's terms, or orthography errors—into an **unambiguous set of database IDs** with zero possibility of misinterpretation.

Here is why standard Text-to-SQL or naive RAG falls apart on this problem:

1. **Hierarchical & Combinatorial Entity Ambiguity:** `"Unit 1"` (`Unidad 1`) and `"Batch 224"` (`Lote 224`) are not globally unique names. Every sector (`Valle Norte`, `Sector Alpha`, `Sierra Sur`) has a `"Unit 1"`. In the underlying SQL tables, `Unit 1` in `Valle Norte` maps to an internal composite key (`unit_internal_id`), which only becomes unambiguous when traversed through its parent `Sector` and linked `Cycle` batch.
2. **Polyglot Persistence (SQL + Flux):** Asking for *growth weight* or *feed conversion* requires a SQL query against MySQL (`growth_weight_logs`, `unit_conversion_metrics`) or PostgreSQL (`inventories`). Asking for *silo weight* or *feed stock* requires a **Flux** query against **InfluxDB** (`iot_silo_telemetry`), parameterized by a time window derived from when that specific production cycle started in MySQL!
3. **Implicit Temporal Context:** Users rarely specify exact ISO timestamps when asking about a production batch or *"the last 3 days."* The system must infer whether time is bounded by an operational cycle (`Batch` / `Cycle`) or a relative clock window.
4. **Domain Benchmarks & Context Enrichment:** Returning raw numbers isn't enough. To act as a domain expert without hardcoding facts in prompts, the system has to dynamically pull reference curves and initial batch parameters alongside the requested KPI.

---

## 4. Diagnosing the Engineering Bottlenecks

The high-level idea sounded simple in our heads: *process user input -> grab the data -> enrich the data -> return a domain-expert response*. Once I actually started building it, I ran straight into three major bottlenecks:

### Bottleneck 1: Lexical Extraction vs. Compound Entities

When extracting keywords from `"Valle Norte, unidad 1 y lote 224"`, standard tokenizers or naive keyword prompts split `"unidad"` and `"1"`. Searching a vector index for `"unidad"` returns every production unit in the organization with identical cosine similarity. I needed a linguistically strict keyword extraction step that binds nouns, adjectives, and proper nouns to immediately following identifiers (`"unidad 1"`, `"lote 224"`), while ignoring date markers.

### Bottleneck 2: Category Discovery vs. Specific Entity Reranking (The Cartesian Explosion)

Even when searching for `"Unit 1"` in a dedicated Neo4j vector index, the database legitimately returns dozens of physical entity nodes with a `1.0` cosine similarity score—because dozens of sectors have a Unit 1 (`external_id: 2101`, `3101`, etc.).
Initially, I generated the Cartesian product of all candidate entity IDs and formatted a massive Cypher `RETURN EXISTS {...} AS a1, EXISTS {...} AS b1` query. As the knowledge graph grew, I needed a two-step **category discovery + per-category reranking** strategy followed by batched graph path validation.

### Bottleneck 3: Two-Tier Graph Abstraction (Ontology vs. Physical Entities)

Asking an LLM to write a pathfinding Cypher query directly over tens of thousands of physical `:Entity` nodes is slow and error-prone—which was what native Neo4j LLM tools did at the time. I needed to enforce a strict set of rules in the node database: separating **conceptual nodes** (an abstraction of how categories relate) from **physical entity nodes** (real-world sites, units, and cycles with specific names and IDs).

---

## 5. The Solution: A Graph-Guided Agentic Workflow

To solve these challenges without hardcoding domain logic, I designed a **multi-node state machine in LangGraph** backed by a **Two-Tier Neo4j Knowledge Graph**.

### Step 1: Designing the Two-Tier Knowledge Graph & Vector Store

In Neo4j, I enforced a strict schema rule splitting nodes into two connected layers linked by `[:INSTANCIATED_FROM]` relationships:

* **The `:Concept` Layer (The Conceptual Schema):** A lightweight meta-graph (`(:Concept:Sector)-[:CONTAINS]->(:Concept:Unit)-[:REGISTERS]->(:Concept:Cycle)-[:HAS_MEASURE]->(:Concept:KPI)`). Because conceptual nodes mirror the exact labels and relationships that physical entities take, this tiny graph serves as a compressed map of the entire database for the LLM.
* **The `:Entity` Layer (Physical Entities & `:EndNode` Routers):** The real-world operational entities populated from PostgreSQL/MySQL, embedded with OpenAI’s `text-embedding-3-large` (3072 dimensions), and indexed both globally and per concept category (`Sector`, `Unit`, `Cycle`, `KPI`).

Crucially, measurable leaf nodes are tagged as `:EndNode:KPI`. Rather than duplicating massive time-series tables inside Neo4j, these `:EndNode`s store the target database name, engine type (`mysql` vs. `influx`), and parameterized query templates directly on the node:

```python
# Attaching Federated Query Templates to Graph EndNodes
driver.execute_query(
    """
    MATCH (n:KPI {name:'weight'})
    SET n.query = 'SELECT * FROM growth_weight_logs WHERE unit_internal_id = {unitID} AND cycle_number = {cycleID} AND active = 1'
    """, database_=NEO4J_DATABASE
)

driver.execute_query(
    """
    MATCH (s:Sector)-[:CONTAINS]-(u:Unit)-[:REGISTERS]-(c:Cycle)-[:HAS_MEASURE]->(n:KPI {name:'silo weight'})
    SET n.query = 'from(bucket: "telemetry_metrics") 
|> range(start: {start_datetime}, stop: {end_datetime}) 
|> filter(fn:(r) => r._measurement == "iot_silo_telemetry" and
 r.unit_internal_id == "{unitID}")'
    """, database_=NEO4J_DATABASE
)
```

*(Note: While I used fixed query templates on `:EndNode`s for deterministic reliability in production, I also experimented with dynamic query construction by hooking a **SQL MCP** directly to the database. Depending on schema complexity, dynamic MCP generation can work well, though templates attached to graph nodes guarantee exact joins across polyglot stores).*

When rebuilding the graph, I also implemented **local embedding caching** and batched `CALL (n) { ... } IN TRANSACTIONS OF 1000 ROWS` mutations so Neo4j wouldn't run out of heap memory when updating high-cardinality node categories (>10,000 nodes).

---

### Step 2: The LangGraph State Machine

When a request arrives from the REST API or the interactive UI, it initializes a strongly typed state object (tracking everything from Polars DataFrames to the step-by-step execution trace). Rather than relying on a single monolithic prompt, the state machine orchestrates specialized stages with conditional routing gates—directing unresolvable or out-of-scope queries to a fallback handler that gracefully explains why the request couldn't be fulfilled.

#### 1. Dynamic Intent Routing

First, the **Intent Router** queries Neo4j for the live list of `:KPI:Entity` names and determines whether the user is inquiring about a measurable KPI (returning a boolean) or asking a general structural question:

* **Exploratory Metadata Path:** If the user asks *"Which batches and units exist for Sector Ridge?"*, an **Exploratory Cypher Generator** extracts hints via vector search, injects the `:Concept` schema, generates a `MATCH ... RETURN DISTINCT` Cypher query, rewrites node patterns via regex to enforce `:Entity` label scoping, and executes it with automatic syntax-retry logic. If it isn't a valid structural or KPI question, it routes straight to the fallback handler.
* **Metric Data Path:** Triggers the two-stage kANN discovery and graph disambiguation pipeline.

#### 2. Two-Stage kANN Search: Category Discovery + Per-Category Reranking

In the **Entity & Metric Discovery** stages, a linguistic extraction prompt pulls compound units—nouns, adjectives, proper nouns, and attached numbers (`["Valle Norte", "unidad 1", "lote 224", "peso de silo"]`).

Why did I run two separate k-Approximate Nearest Neighbors (kANN) vector searches?

1. **Stage 1 — Category Discovery:** Each extracted keyword is embedded and searched against the global entity vector index (`LIMIT 1, score > 0.5`). Here, I wasn't trying to lock in the final ID yet; I was asking: ***"Which category (label) of physical entity does this keyword belong to?"***
   From the returned nodes, the system also checks if an `:EndNode` (an actual measurable KPI) is present. If an `:EndNode` exists, it knows for certain the user is asking for something we measure and can answer numerically.
2. **Stage 2 — Per-Category Reranking:** First, the system pulls all distinct labels from the `:Concept` layer. Because the Neo4j database maintains dedicated vector indexes per label category (`Sector`, `Unit`, `Cycle`, `KPI`), I then re-queried each keyword against **only the index matching its category**.
   Essentially, this acts as a targeted **reranker**:
   * *Step 1 asks:* "Which category does this keyword belong to?"
   * *Step 2 asks:* "Given I know this keyword belongs to category $X$, which specific physical entity node within $X$ does it refer to?"

   Wait—what if step 1 matched three keywords to the same category, or the vector index returns multiple top-scoring candidates? Since the system is designed to answer one cohesive question at a time, only one physical entity per category should ultimately win (or all tied top-scoring IDs must be tested against the graph):

   ```python
   in_df = in_df.filter(pl.col("score") == pl.col("score").max())
   ```

#### 3. Conceptual Pathfinding & Combinatorial Graph Pruning

At this point, the state holds candidate physical IDs for `Sector`, `Unit`, `Cycle`, and `KPI`. How do I prove which `Unit 1` actually belongs to `Valle Norte` and registered `Batch 224`?

1. **Conceptual Path Generation:** The **Pathfinding Step** receives the `:Concept` schema in its context and the list of identified node labels. Because conceptual nodes mirror the exact relationships of the entity layer, the LLM constructs Cypher path queries connecting all identified labels:

   ```cypher
   MATCH (sector:Sector)-[:CONTAINS]-(unit:Unit)-[:REGISTERS]-(cycle:Cycle)-[:HAS_MEASURE]-(kpi:KPI)
   RETURN sector, unit, cycle, kpi
   ```

   Left as a label-only query, this would return *every* connected entity in the database.
2. **Placeholder Injection:** Using regex, I injected recognizable ID placeholders into each node pattern in the generated Cypher query—transforming `(:A)-[:REL]-(:B)` into `(:A {id: $id1})-[:REL]-(:B {id: $id2})`.
3. **Combinatorial Disambiguation:** The **Graph Pruning Step** substitutes every candidate ID combination into the placeholders and tests which combinations actually form a connected path in Neo4j. Because IDs are unique, no more than one path can exist between a specific tuple of unique IDs:

```python
# Batch Graph Disambiguation with Score Tie-Breaking
check_existence = f"""
// Iterate over each candidate ID combination dictionary in $rows
UNWIND $rows AS row

// Test the existence of the full graph pattern for each candidate tuple
OPTIONAL MATCH path = {placeholder_query}

// Return the input IDs along with a boolean indicating if the path exists in the graph
RETURN
{new_return_part}
path IS NOT NULL AS pathExists
"""

res = execute_cypher_query(check_existence, db=db, rows=comb_dict)
possible_ids = list(set([record for record in res if record["pathExists"]]))

# If multiple valid graph paths exist (e.g., fuzzy keyword matches), break ties using cumulative vector score
if len(possible_ids) >= 2:
    key_list = list(comb_dict[0].keys())
    scores = []
    for key_i in possible_ids:
        ids_found = [str(key_i[f"row.{i}"]) for i in key_list]
        score_df = filtered_scores.filter(pl.col("id").is_in(ids_found))
        scores.append(score_df["score"].sum())
    possible_ids = possible_ids[scores.index(max(scores))]
```

In a single round-trip to Neo4j, invalid combinations are pruned by the graph topology itself. The system has now pinpointed, without ambiguity, the exact physical entities the user is asking about.

#### 4. Temporal Resolution

What if the user asks for a KPI over a relative timeframe (*"oxygen concentration over the last 3 days"*) instead of naming a specific `Batch` or `Cycle`?
A conditional routing gate detects when a target KPI node is present but an explicit temporal cycle node is missing. It routes the state to the **Temporal Resolution Step**, which queries the `:EndNode` in Neo4j for its minimum time unit (e.g., `'1 day'` or `'15 minutes'`) and invokes a tool-bound LLM equipped with a date-math utility. The resulting ISO interval (`{"from": ..., "to": ...}`) is injected into the Cypher query (`WHERE date({date}) >= cycle.start AND date({date}) <= cycle.end`) to automatically resolve which operational cycle was active during that window.

#### 5. Polyglot Query Execution

Once all unique entity IDs are locked in, the **Query Construction Step** pulls the parameterized query template from the resolved `:EndNode` and the **Data Retrieval Step** fetches the actual records:

* **In Federated Production Mode:** It inspects the `:EndNode`'s `type` property in Neo4j:
  * If `type == "mysql"`, the SQL retrieval handler populates the SQL template with the resolved IDs and loads the result directly into a **Polars DataFrame** via SQLAlchemy (`pl.read_database`).
  * If `type == "influx"`, the time-series retrieval handler first queries MySQL (`cycle_initial_parameters`) to find the `cycle_start_date` for that sector and batch, computes a `[-5 days, +50 days]` window around the cycle, populates the Flux template, queries InfluxDB, and converts the result into a clean Polars DataFrame.
* **In Enriched Benchmark Mode:** In addition to querying PostgreSQL for the primary KPI, the retriever traverses Neo4j from the KPI node to find neighboring `:EndNode`s (`(:KPI)-[:COMPARES_AGAINST]->(:Standard)` and `(:KPI)-[:STARTS_WITH]->(:InitialState)`). It automatically fetches the reference growth curves (e.g., `STD-LINE-A`, `STD-LINE-B`) and initial cohort weights.

#### 6. Turning Numbers into Domain-Expert Insights

Remember our second rule of engagement: **a numeric answer is only the baseline; the system must answer like a domain expert without hardcoded facts.**

1. **Context Window Guardrail (Initial Response Formatter):** Before handing data to `gpt-4o`, I added a guardrail against massive time-series payloads (e.g., 50,000 rows of 1-minute silo weights from InfluxDB):

   ```python
   def format_primary_results(state):
       result = state["results"]
       if result.shape[0] > 300:
           result = result.describe()
       else:
           result = result.to_dicts()
           
       res_format = response_formatter.invoke({
           "dataframe": result, 
           "user_input": state["user_input"]
       })
   ```

   Using `result.describe()` in Polars summarizes the statistical distribution when row counts exceed 300, preventing context-window blowouts while still allowing the API layer to return the full `results.to_dicts()` payload to programmatic callers.
2. **Graph-Enriched RAG (Benchmark & Knowledge Graph Synthesis):** When supplementary domain tables (reference standards and initial parameters) and graph context exist, a **Benchmark Evaluator** summarizes the reference comparisons and a **Knowledge Graph Context Retriever** pulls qualitative domain context from the graph. Finally, an **Enrichment Synthesizer** combines the raw measurement, the target reference curve, and the graph context into a cohesive domain-expert evaluation—turning *"1 kg at day 30"* into an actionable operational insight.

---

## 6. Pros, Cons & Candid Reflections

Looking back at the system I built and where the AI ecosystem has moved since, here is my honest assessment of the trade-offs:

### What Worked Well (Pros)

* **Uncompromising Precision on Ambiguous Input:** The pipeline achieved our primary goal—turning messy, multilingual, typo-prone user questions into an unambiguous set of database IDs. By letting the Neo4j graph topology prune vector search candidates, it eliminates hallucinated joins or mismatched unit/sector IDs.
* **Zero Hardcoded Context:** Every KPI, entity hierarchy, benchmark table, and query template lives in the graph or relational stores, satisfying our strict no-hardcoding mandate.
* **Polyglot Routing Without Data Duplication:** Using `:EndNode`s as federated routers over MySQL, PostgreSQL, and InfluxDB kept time-series data where it belongs while giving the agent a unified semantic map.

### Limitations & Trade-Offs (Cons)

* **Precision Costs Latency:** Searching global vector indexes, reranking per category, generating conceptual paths, and testing combinations in Neo4j is accurate, but it takes time.
* **Single-Question Scope:** In this architecture, the vector reranking and combinatorial path-pruning steps resolve a single coherent path per turn—meaning you cannot ask multiple unrelated questions in one prompt (though the state machine is easily modifiable to loop over sub-questions).
* **Heavy Dependence on Graph Modeling Rules:** Whether this architecture works or fails is dictated almost entirely by how disciplined your node database construction is. The graph has strict structural rules (`:Concept` vs. `:Entity`, `:EndNode` metadata, 1-to-1 label mirroring). It is also an open question how many businesses have physical entity hierarchies complex enough to require this exact graph-pruning pattern—or so densely interconnected that even backend path disambiguation struggles.

### Would I Build It the Same Way Today? (Neo4j Pipeline vs. Direct SQL MCP)

To some readers, it may look like I took the long path: extracting keywords, running two-stage kANN searches, reranking by category, and testing hundreds of candidate Cypher combinations just to run a SQL query.

With the stronger reasoning models available today, I genuinely believe that for many standard relational setups, **you could hook a SQL MCP directly to your database and bypass the Neo4j disambiguation layer entirely.** I even tested dynamic query generation via MCP during this project.

Of course, giving an LLM direct MCP access to raw production schemas brings its own set of problems—schema sprawl, ambiguous foreign keys across legacy tables, token-heavy table dumps, and cross-engine coordination (like needing a MySQL start date to parameterize an InfluxDB Flux query). Someone else is probably fighting those exact battles right now. When determinism across polyglot industrial data matters more than raw speed, a graph-guided semantic router remains a remarkably solid pattern to have in your toolkit.
