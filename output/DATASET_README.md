# Vision Agent UI Automation Dataset

## Overview

This dataset contains captured UI states from automated task execution across 3 different productivity applications: **Linear**, **Notion**, and **Asana**. Each task demonstrates the vision agent's ability to navigate complex web UIs, identify interactive elements, and complete multi-step workflows autonomously.

---

## Dataset Structure

```
output/
├── linear_task_create_Newproject/     # Task 1: Create project in Linear
├── linear_task_filterBypriority/      # Task 2: Filter issues in Linear
├── notion_task_create_newPage/        # Task 3: Create page in Notion
├── asana_task_create_NewProject/      # Task 4: Create project in Asana
└── DATASET_README.md                  # This file
```

Each task folder contains:
- `manifest.json` - Structured metadata with task description, step-by-step actions, agent reasoning
- `step_N_*.jpg` - Screenshots at each decision point with bounding box overlays
- `step_N_cursor_*.png` - Screenshots showing click location (red dot)
- `step_N_click_result_*.jpg` / `step_N_type_result_*.jpg` - Post-action screenshots

---

## Tasks Summary

### Task 1: Create Project in Linear
**App:** Linear (Project Management)
**Objective:** Create a new project named "Softlight Plan"
**Steps:** 5
**Workflow:**
1. Navigate to Projects section
2. Click "Add project" button
3. Type project name in auto-focused field
4. Click "Create project" button
5. Verify project appears in list

**Key Challenges:** Modal form handling, auto-focused input detection

---

### Task 2: Filter Issues by Priority in Linear
**App:** Linear (Project Management)
**Objective:** Filter issues to show only medium priority items
**Steps:** 4
**Workflow:**
1. Click Filter button to open filter panel
2. Select "Priority" from filter options
3. Select "Medium" priority value
4. Verify filter is applied (indicator visible)

**Key Challenges:** Hierarchical menu navigation, visual confirmation of applied filter

---

### Task 3: Create New Page in Notion
**App:** Notion (Note-taking/Wiki)
**Objective:** Create a new page titled "My Firstpage"
**Steps:** 3
**Workflow:**
1. Click "New page" button
2. Type page title in auto-focused field
3. Verify page appears in sidebar

**Key Challenges:** Contenteditable fields, SPA URL updates, sidebar navigation

---

### Task 4: Create Blank Project in Asana
**App:** Asana (Project Management)
**Objective:** Create a new blank project named "Softlight"
**Steps:** 7
**Workflow:**
1. Click "New project or portfolio" button
2. Select "New project" from dropdown
3. Select "Blank project" template
4. Type project name
5. Click "Continue"
6. Click "Create project"
7. Verify project in sidebar

**Key Challenges:** Multi-step wizard, template selection, button state transitions

---

## Data Format

### manifest.json Schema

```json
{
  "task": "Human-readable task description",
  "states": [
    {
      "step": 1,
      "timestamp": 1763950895524,
      "screenshot": "step_1_1763950895524.jpg",
      "url": "https://...",
      "agent_thought": "Agent's reasoning about current state",
      "action_taken": "click|type|scroll|finish",
      "action_params": {
        "text": "Element/text to interact with",
        "element_index": 85,
        "selector": "CSS selector"
      }
    }
  ]
}
```

### Screenshot Types

| File Pattern | Description |
|-------------|-------------|
| `step_N_*.jpg` | State capture with red bounding boxes and element indices |
| `step_N_cursor_*.png` | Click location visualization (red dot) |
| `step_N_click_result_*.jpg` | Screenshot immediately after click action |
| `step_N_type_result_*.jpg` | Screenshot immediately after type action |

---

## Applications Covered

| App | Type | URL | Tasks |
|-----|------|-----|-------|
| Linear | Project Management | linear.app | 2 |
| Notion | Notes/Wiki | notion.so | 1 |
| Asana | Project Management | asana.com | 1 |

---

## Usage

This dataset can be used for:
- Training/evaluating UI grounding models
- Benchmarking vision-language agents on web automation
- Studying human-like UI navigation patterns
- Testing element detection and bounding box algorithms

---

## Technical Details

- **Element Detection:** DOM-based with CSS selectors for buttons, links, inputs, ARIA roles
- **Bounding Box Filtering:** IoU-based deduplication (90% threshold) inspired by OmniParser
- **Vision Model:** GPT-4o with high-detail image processing
- **Browser:** Playwright with persistent Chrome context
- **Resolution:** 1280x720 viewport

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Tasks | 4 |
| Total Steps | 19 |
| Applications | 3 |
| Avg Steps/Task | 4.75 |
| Success Rate | 100% |
