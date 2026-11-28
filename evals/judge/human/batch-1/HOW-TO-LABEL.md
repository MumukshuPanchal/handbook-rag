# How to label batch 1

24 judgements: 16 statements to hold against the passages they cite (part A), and 8 answers to hold against a reference answer (part B). You label from what the page shows and the rubric at its top, as the model readers did. Nobody expects you to know GitLab.

**The one thing not to do: do not look for how this batch was made before your labels are in place.** Read nothing about this sample before your labels are in place: not its card, its decision record, its sections in the README, the guide or the changelog, nor the commit or pull request that added it. The labels the sets give these statements and answers are in the project's files, so do not search them for an item's words either.

1. **Open the page.** Open `label.html` in this folder with your browser: double-click it, or drag it onto a browser window. It needs no internet and no server, and it sends nothing anywhere.
2. **Read the rubric at the top, then label one item at a time.** Press 1, 2, 3 or 4, or click a choice. N or → goes to the next item, P or ← back. Add a short note where you hesitate. "Cannot tell" is a fair answer; it is counted apart. Decide each item on its own: do not count how many of each label you have given.
3. **Stop and come back when you like.** The page keeps your labels in this browser as you go, and says so; if it cannot, it says that too. Come back in the same browser. If it finds labels from before, it says when they were made and asks whether to keep them.
4. **Save the file.** When every item has a label, press "Save labels file". Each save makes a new file, `labels-batch-1-<date>-<time>.json`, usually in your Downloads folder, and the page shows its name and time. To change a label, change it on the page and save again: the newest file holds your labels. The page keeps its copy until you press "Clear this browser's copy".
5. **Put the newest file in place.** In a terminal, in the project folder, run, with the name the page showed at your last save:

   ```bash
   uv run handbook-rag human-labels ~/Downloads/labels-batch-1-<date>-<time>.json
   ```

   It checks the file, says when it was saved and how many labels and notes it holds, and puts it here as `labels.json`. It refuses an older file when a newer one lies beside it. If it refuses, its message says what to do. Then `uv run handbook-rag eval-human --variant batch-1` gives the report.
