# L1-C5 ER Unchanged-Eb Replay

- Verdict: **PASS_ACTION_SEPARATION**
- Episodes: 50
- Safe task-success rate: 0.040
- Exact matched-control rate: 0.280
- Attribution-eligible paired rate: 0.960

| Episode | Eligible | EB safe | Replay safe | Matched | Violated | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task9_ep000.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant tilt change=10.2deg>10.0deg |
| task9_ep001.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep002.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep003.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0333m<0.0500m |
| task9_ep004.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep005.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep006.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0386m<0.0500m |
| task9_ep007.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0207m>0.0200m |
| task9_ep008.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep009.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0212m>0.0200m |
| task9_ep010.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0441m<0.0500m |
| task9_ep011.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0246m<0.0500m |
| task9_ep012.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0201m>0.0200m |
| task9_ep013.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep014.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0124m>0.0100m |
| task9_ep015.npz | 0 | 0 | 1 | 0 | 0 | -- |
| task9_ep016.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0232m>0.0200m |
| task9_ep017.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0319m<0.0500m |
| task9_ep018.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0114m>0.0100m |
| task9_ep019.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target not finally stable for 15 consecutive steps (final_streak=0; tilt=15.2deg>10.0deg) |
| task9_ep020.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0310m>0.0200m |
| task9_ep021.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: occupant tilt change=11.0deg>10.0deg |
| task9_ep022.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0111m>0.0100m |
| task9_ep023.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep024.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep025.npz | 0 | 0 | 1 | 0 | 0 | -- |
| task9_ep026.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep027.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0109m>0.0100m |
| task9_ep028.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0179m>0.0100m |
| task9_ep029.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0139m>0.0100m |
| task9_ep030.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0140m>0.0100m |
| task9_ep031.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0100m>0.0100m |
| task9_ep032.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0201m>0.0200m |
| task9_ep033.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep034.npz | 1 | 1 | 0 | 0 | 0 | -- |
| task9_ep035.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0264m<0.0500m |
| task9_ep036.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant tilt change=10.1deg>10.0deg |
| task9_ep037.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0126m>0.0100m |
| task9_ep038.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0107m>0.0100m |
| task9_ep039.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0124m>0.0100m |
| task9_ep040.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0219m>0.0200m |
| task9_ep041.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0201m>0.0200m |
| task9_ep042.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0285m>0.0200m |
| task9_ep043.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0234m>0.0200m |
| task9_ep044.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep045.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0241m>0.0200m |
| task9_ep046.npz | 1 | 0 | 0 | 0 | 0 | -- |
| task9_ep047.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0204m>0.0200m |
| task9_ep048.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target xy displacement=0.0246m>0.0200m |
| task9_ep049.npz | 1 | 0 | 0 | 0 | 0 | -- |
