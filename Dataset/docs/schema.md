# Dataset Schema

The formal release schema is documented in `formal_schema.md`.

The bundled file contains the exact fields required by the CompFaith runner:

- `id`: unique sample identifier;
- `dataset`: dataset name;
- `context`: source document;
- `candidate`: generated response;
- `faithfulness`: continuous faithfulness reference in `[0, 1]`;
- `faithful_label`: binary faithful label;
- `hallucination_label`: binary hallucination label;
- `metadata`: dataset subdomain metadata.
