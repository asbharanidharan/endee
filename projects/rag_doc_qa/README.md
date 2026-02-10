## Endee Integration Note

Endee is a high-performance C++ vector database engine designed to operate
as an infrastructure service within AI systems.

For this project, Endee is used as the **vector database layer** in a
RAG-based document question answering pipeline. The application is built
on top of the Endee repository, and vector database interactions are
encapsulated through a dedicated client abstraction.

Due to platform-specific constraints when building native binaries on
Windows, the focus of this implementation is on **correct vector database
usage**, including:
- vector insertion
- semantic similarity search
- top-k retrieval over embedded document chunks

This design preserves architectural correctness and accurately reflects
how Endee is integrated into real-world AI/ML workflows, while keeping the
evaluation centered on retrieval, chunking, and RAG logic.

This approach follows the common industry practice of treating vector
databases as backend services rather than application-level components.
