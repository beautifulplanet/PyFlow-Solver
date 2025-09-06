# Extended Learning Report
Generated: 2025-08-29T23:19:06.571061+00:00

Python files analyzed: 290
Suppressed SyntaxWarnings during parsing: 0
Total functions extracted: 12732
Complexity distribution: <=10:1701, <=15:536, <=25:908, <=5:8652, >25:935
Top CFD tags: velocity(289), pressure(277), boundary(212), solver(171), residual(126), flux(80), mesh(65), cfd(29), navier(26), stokes(26), timestep(8)
Top signature stems: execute(self,node)(450), interpret(self,ast)(446), evaluate(self,node)(446), __init__(self)(432), __init__(self,tokens)(422), parse(self)(422), parse_expression(self)(422), parse_statement(self)(420), current(self)(398), consume(self,type)(398)
Evolution chains (top 5 by length):
- pycfdflow2_: pycfdflow2_v2.py -> pycfdflow2_v2.py -> pycfdflow2_v3.py -> pycfdflow2_v3.py -> pycfdflow2_v4.py -> pycfdflow2_v4.py -> pycfdflow2_v5.py -> pycfdflow2_v5.py -> pycfdflow2_v6.py -> pycfdflow2_v6.py -> pycfdflow2_v7.py -> pycfdflow2_v7.py
- grid: grid.py -> grid.py -> grid.py -> grid.py -> grid.py -> grid.py
- cfd_: cfd_v021.py -> cfd_v021.py -> cfd_v023.py -> cfd_v023.py -> cfd_v023.py
- : _12.py -> _12.py -> _12.py -> _12.py
- test_core: test_core.py -> test_core.py -> test_core.py -> test_core.py
Failure clusters (top 10):
- except FileNotFoundError: (count 130)
- "    print(f\"An error occurred while reading {file_path}: {e}\")" (count 93)
- print(f"Error: File not found at {file_path}") (count 73)
- As per your directive for "10 rounds between showings" for "everything is done nothing to check moments," I will present the initial blueprint, and then imme... (count 56)
- Modern physics is haunted by the specter of the infinite. In fluid dynamics, this manifests as the prediction of infinite forces and pressures within turbule... (count 55)
- This arrangement naturally handles the pressure-velocity coupling and prevents common numerical errors. (count 55)
- // Calculate the residuals (a measure of error) for the momentum and mass (count 55)
- "except FileNotFoundError:\n", (count 50)
- "    print(f\"Error: File not found at {file_path}\")\n", (count 50)
- Now, Dmitry, the extreme internal pre-debugging protocol has begun for Round 10. The Formalizer has generated this initial blueprint. Skeptical Alpaca is alr... (count 48)

See cfd_scaffold_recommendation.md for scaffold guidance.