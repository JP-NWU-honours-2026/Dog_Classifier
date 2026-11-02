# CanineVision Dog Breed Merger & Simplified Split Manifest Checklist

**Feature**: Class Simplification & Breed Merging Studio  
**Target Output**: `JP_Model/split_manifest_simplified.json` & `Dog_Classifier_App` Interactive UI  
**Status**: In Progress  

---

## 1. Objectives

1. Allow users to group morphologically and taxonomically similar dog breeds (e.g. *American Staffordshire Terrier* & *Staffordshire Bullterrier*, *English Foxhound* & *Walker Hound*, *Poodles*, *Schnauzers*) into unified parent classes.
2. Select source manifest dynamically (`JP_Model/split_manifest_clean.json` or `JP_Model/split_manifest.json`).
3. Offer pre-curated similarity clusters with one-click toggles and editable target class names.
4. Provide a custom multi-breed group builder for arbitrary manual merges.
5. Display a live preview of resulting classes ($0 \dots K-1$), count distributions, and class reduction percentages.
6. Generate and export a valid `split_manifest_simplified.json` with continuous integer labels $0 \dots K-1$, updated `breed_name`, and preserved image provenance (`path` and `breed_folder`).

---

## 2. Implementation Checklist

### Phase 1: Backend Architecture (`server.py`)
- [x] **1.1 Source Manifest Discovery Endpoint (`GET /api/merger/source_manifests`)**:
  - Detects `split_manifest_clean.json` and `split_manifest.json` under `JP_Model/`.
  - Returns metadata, class counts (120), and split distributions.
- [x] **1.2 Pre-Curated Similarity Clusters Endpoint (`GET /api/merger/suggested_clusters`)**:
  - Pre-encodes high-confidence taxonomic clusters (Staffordshire Terriers, Foxhounds, Poodles, Schnauzers, Corgis, Setters, Retrievers, Belgian Shepherds, Swiss Mountain Dogs, Coonhounds, Bulldogs, etc.).
  - Includes sample thumbnail photo URLs from `/archive_images/` for instant visual comparison.
- [x] **1.3 Dynamic Merge Simulation & Preview Endpoint (`POST /api/merger/preview`)**:
  - Computes mapping from original 120 breeds to $K$ simplified classes.
  - Generates new continuous class IDs ($0 \dots K-1$).
  - Calculates split counts (`train`, `val`, `test`) and reduction telemetry.
- [x] **1.4 Manifest Generator Endpoint (`POST /api/merger/generate`)**:
  - Reads selected source manifest.
  - Remaps every sample in `train`, `val`, `test` to new `label` and `breed_name`.
  - Writes formatted `JP_Model/split_manifest_simplified.json`.
  - Saves audit summary to `JP_Model/results/merger_summary.json`.
- [x] **1.5 Manifest Download Endpoint (`GET /api/merger/download`)**:
  - Serves `split_manifest_simplified.json` for direct browser download.

---

### Phase 2: Frontend User Interface (`index.html`, `style.css`, `app.js`)
- [x] **2.1 Navigation & Tab Setup**:
  - Add 4th navigation tab: **"Class Simplifier & Merger"** (`#tabMerger`).
  - Create view container (`#viewMerger`).
- [x] **2.2 Telemetry & Source Selector Bar**:
  - Source manifest dropdown (`split_manifest_clean.json` vs. `split_manifest.json`).
  - 4 telemetry KPI cards:
    - Original Classes (120)
    - Merged Groups Count
    - Resulting Classes ($K$)
    - Class Reduction %
- [x] **2.3 Suggested Similarity Clusters Deck**:
  - Interactive cards with toggle switches, editable group name inputs, breed pills, image count badges, and thumbnail photos.
  - "Select All Suggested" / "Deselect All" bulk control buttons.
- [x] **2.4 Custom Breed Group Builder**:
  - Multi-select search dropdown to combine any custom set of breeds.
  - Name input and "Add Custom Group" button.
- [x] **2.5 Live Resulting Classes Inspector**:
  - Searchable list/table showing every final class, new ID, constituent breeds, and split distributions.
- [x] **2.6 One-Click Manifest Generation & Download UI**:
  - "Generate `split_manifest_simplified.json`" primary button with loading states.
  - Success notification banner with summary metrics and instant "Download JSON" link.

---

### Phase 3: Validation & Quality Assurance
- [x] **3.1 Manifest Integrity Test**:
  - Verified labels are strictly contiguous integers from $0$ to $K-1$ (`0..K-1: True`).
  - Verified 100% data preservation across splits (20,410 total samples match clean manifest).
  - Verified provenance fields (`path`, `breed_folder`) remain intact.
- [x] **3.2 Live Web Application Smoke Test**:
  - Verified compilation of all Python backend endpoints.
  - Verified seamless tab integration and real-time live preview calculation.
