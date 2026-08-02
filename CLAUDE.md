# On inconsistencies
- General:This document is going to underly our data curation. It needs to be 100% correct and up-to-date. If you find a disrepancy as you work, fix it unless it's due to a major methodological choice, in which case report it to the use immediately and together you will decide on how to resolve it and update the methods.
- External inconsistencies: If while you search for things outside of this repo, you find contradicting information to what is INSIDE the repo, flag it to the user and together we will update things accordingly.
- Internal inconsistencies: If two or more sources inside this repo contradict each other, check which one is most recently changed (perhaps something is stale) and perhaps easy to fix. If two methodologies lead to the generation of internal inconsistencies, then you need to revisit the methodologies with the user and resolve the internal inconsistencies.

# Checkability
- To ensure correctness, everything should be easily sourced. Results, findings, etc., should all point to the data and code that was used to generate them. We should be able to independently reproduce the results.

# Coding style
- Everything is typed.
- Return type is always stated explicitly (use `None` too).
- Use `jaxtyped` (with `beartype`) for any tensor-typed arguments or returns.

# Skills
- This repo uses pynapple package a lot. There is a Claude skill for that. Leverage it.
