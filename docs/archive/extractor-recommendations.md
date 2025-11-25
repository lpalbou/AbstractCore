
## Additional recommendations

The recommendations are universal because they are based on fundamental principles of data modeling, not on any specific domain. A well-structured knowledge graph in biology will follow the same design principles as one for software or history.

Here is a domain-agnostic, actionable set of recommendations to create simpler and more optimized knowledge graphs.

---

## 1. Differentiate Core Entities from Attributes 🎯

Before creating the graph, separate your concepts into "core entities" and their "attributes."

* **Action**: For every piece of information, ask: **"Is this a primary subject, or is it a property that describes a subject?"**
* **How-to**: Core entities are the "nouns" of your domain (e.g., a `Person`, a `Company`, a `MedicalDiagnosis`, a `ChemicalCompound`). Attributes are the descriptive data *about* those nouns (e.g., `height`, `foundingDate`, `symptomSeverity`, `boilingPoint`). Move attributes out of the top-level entity list and make them properties of their corresponding core entity.
* **Analogy**: Think of it like a contact card. The **person is the core entity**. Their name, phone number, and address are **attributes**. You wouldn't create a separate, top-level "phone number" entity and link it to the person.

---

## 2. Build Hierarchy and Consolidate 📂

Avoid a flat structure where every node is a peer. Group related entities to create a logical hierarchy. This is the most effective way to reduce complexity.

* **Action**: Identify "one-to-many" relationships and use them to nest information.
* **How-to**: If a `Book` entity has many `Chapter` entities, it's often better to list the chapters inside the `Book` entity's data structure. Create separate `Chapter` entities only if they are extremely complex and need to be linked to many other things independently.
* **Analogy**: You organize files on a computer into folders.  You don't place every single document and photo onto the desktop; you group them. Your KG should do the same.

---

## 3. Use Precise and Descriptive Relationships (Predicates) 🔗

The meaning of your graph is carried in the relationships. Vague predicates make the graph ambiguous and less useful.

* **Action**: Replace generic relationship terms like `relatedTo`, `uses`, or `affects` with specific, directional verbs that describe *how* things are related.
* **How-to**: Instead of `(Marie Curie) -relatedTo-> (Radium)`, a more powerful relationship is `(Marie Curie) -discovered-> (Radium)`. Instead of `(Rain) -affects-> (Crops)`, use `(Rain) -promotesGrowthOf-> (Crops)`. Create a small, controlled vocabulary for your common relationships.
* **Analogy**: It's the difference between saying "The two people are connected" and "This person is the **parent of** that person." Specificity adds immense value.

---

## 4. Standardize Your Schema 📝

Consistency is crucial for a graph that is easy to query and maintain. Define a clear structure for each type of entity.

* **Action**: For each entity *type* (e.g., `Person`, `Location`), define a consistent set of properties.
* **How-to**: Decide that every `Location` must have `latitude` and `longitude` properties. Stick to consistent naming conventions (e.g., always use `creationDate`, not a mix of `createdOn`, `date`, and `creationDate`).
* **Analogy**: This is like creating a standardized form or template. Anyone filling it out provides the same categories of information, which makes the collected data clean, predictable, and easy to analyze.