# WAM

This repository collects Chinese study notes and reference papers for World Models and World Action Models.

## Notes

The organized PDFs are in `note/`:

- `01_wam_foundations.pdf`: WAM mathematical foundations and World Model vs WAM.
- `02_model_planning_and_synthetic_rollout.pdf`: model-based planning and synthetic rollout.
- `03_wam_three_paradigms.pdf`: three WAM paradigms and their training/inference differences.

Original individual note PDFs are kept in `note/source/`.

## Web Report

Open the static report page at `web/index.html`, or run a local server from the repository root:

```bash
python -m http.server 8091 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8091/web/index.html`.

## Presentation

- `presentation/World_Action_Models_Academic_Report.pptx`: academic presentation deck for the WAM learning report.

## Papers

Reference papers are in `paper/` with readable filenames:

- `Cosmos_Policy_Fine_Tuning_Video_Models_for_Visuomotor_Control_and_Planning.pdf`
- `Fast_WAM_Do_World_Action_Models_Need_Test_Time_Future_Imagination.pdf`
