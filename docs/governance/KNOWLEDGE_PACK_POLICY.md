# Knowledge pack policy

HLSGraph knowledge packs provide versioned interpretation rules for tools,
standards, and vendor documentation. They are general guidance, not observations
about a particular design.

## Allowed material

A published pack may contain:

- a stable document identifier, version, title, publisher, and official HTTPS URL;
- a section or command name sufficient to locate the cited material;
- applicability selectors such as vendor, tool, version, dialect, and stage;
- a short, original paraphrase of one narrowly scoped rule; and
- machine-readable consequences for HLSGraph's truth taxonomy.

The rule must bind `document_id`, `document_version`, and `section`. A rule for one
tool release must not silently apply to another release. Applicability matching is
fail-closed: a tool- or stage-specific rule is not selected when that context is
unknown.

## Prohibited material

Do not commit or distribute vendor PDFs, screenshots, extracted pages, OCR output,
large quotations, document chunks, embeddings of document text, or reconstructed
manuals. Do not use a knowledge rule to assert a design-specific fact or QoR value.

Local indexing of a user-owned document records only its URI, SHA-256 digest, size,
media type, modification timestamp, document identity, and optional official URL.
The indexer neither parses nor copies the document.

## Review requirements

Every new or changed rule requires a human reviewer to check:

1. the official URL, document version, and section identity;
2. that the paraphrase is accurate, short, and independently worded;
3. that applicability is no broader than the cited guidance;
4. that the rule cannot be confused with a tool observation or measurement; and
5. that no copyrighted document body or confidential design data is present.

Breaking semantic changes require a new rule ID or document version. Corrections
that alter applicability or effect must be called out in the pull request.

AMD, Vitis, Vivado, LLVM, MLIR, CIRCT, and other names remain the property of their
respective owners. Referencing a document does not imply endorsement.
