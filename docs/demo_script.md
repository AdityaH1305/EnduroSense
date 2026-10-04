# Demo script (about 6 minutes)

**Before the session**

```bash
streamlit run app/streamlit_app.py
```

- Open http://localhost:8501 and visit each page once, so everything is loaded (the first load takes about 10 seconds).
- Leave the threshold at 0.95 and *Show what was measured* switched **off**.
- Have `docs/EnduroSense_Slides.pptx` open at slide 8 as a fallback if the dashboard cannot be shown.

## 1. Three missions, one battery (2 minutes)

1. **Say:** "This is a battery from the test set, which no model ever saw, just before take-off. Here are three missions that were really flown."
2. **Point at** the line under each mission: *Minutes-left rule: GO*. "The rule in the brief approves all three."
3. **Point at** the middle card: P(success) 47%, NO-GO. "EnduroSense refuses this one. The two ranges overlap: the battery may not have the energy."
4. **Switch on** *Show what was measured* in the sidebar.
5. **Say:** "The two it approved were fine. The one it refused would have gone 5 Wh into the reserve."
6. **If asked how the case was chosen:** by a fixed rule from the test data (stated at the bottom of the page), not by hand.

## 2. Mission check (2 minutes)

1. Switch *Show what was measured* **off**. Open **Mission check**.
2. Keep *My own battery and mission*. Choose any battery and *Before a take-off*.
3. **Drag** *Distance per leg* from 300 m to 600 m. "The mission's range moves right and P(success) falls. Near 600 m the decision flips to NO-GO, while minutes-left still says GO." (With battery chain 3 before flight 5: 99.6% at 300 m, 90.9% at 600 m.)
4. **Point at the warning** that appears above about 330 m per leg: "The mission model says when it is asked about something it was never tested on." Setting the speed to 15 m/s shows the same kind of warning.
5. **Raise the payload** to show P(success) falling further.
6. Choose *Start from* → **A case EnduroSense gets wrong**, and switch *Show what was measured* on. "It is not perfect. Here it approved a mission that cut into the reserve. All such cases come from two batteries whose capacity the model over-estimated."

## 3. Fleet (1 minute)

1. Open **Fleet**. "Four drones, 40 tasks a day, real unseen batteries and real recorded flights."
2. **Point at** the left chart: 12 unsafe missions per 100 with the brief's rule, under 1 with EnduroSense.
3. **Point at** the right chart: the cost is about 10% more battery swaps.

## 4. Results (1 minute)

1. Open **Results**. "We wrote down four success criteria before opening the test set. Two were met, two were not."
2. **Read out** the two that were not met, and scroll to *What did not hold up*.

## Questions that are likely, and short answers

| Question | Answer |
|---|---|
| Why not just add a safety margin to minutes-left? | To be as safe, it has to refuse 9 in 10 feasible missions. The Results page shows it. |
| Is a probability better than point estimates plus a fixed margin? | Over all cases, no measurable difference. At take-off it refused fewer feasible missions (25% against 32%), and it needs no tuning. |
| How do you know the test set was not used earlier? | It is gated in code, every opening is logged, and a verification pass checked that no development file contains a test reading. |
| How big is the test set? | 16 battery chains and 40 flights; 11 chains have an energy label. That is why the ranges on our results are wide, and we say so. |
| Would it work on another drone? | Untested. The method carries over; the models would need that drone's data. |
| How fast is it? | About 5 ms per prediction on a laptop CPU. |
