# Changelog
All notable changes to this project will be documented in this file.

## [1.0.0] - 15-07-2025
Refactor of the entire project! 

### Added
- New interfaces folder for setting up optimisation problems on e.g. slurm
- It is possible to save the optimiser again, now in a readable, stable json file
- New setting file (also json) where optimiser configuration can be saved
- New constructor functions that can be called instead of creating classes directly
- veropt is now typed and checked by mypy

### Changed
- Internal structure
- Interfaces
- Examples

### Removed
- The GUI is not currently available but will hopefully return in the future

## [0.6.0] - 28-02-2025
### Added
- Changelog :))
- New visualisation tools in plotly
- Test folder!
  - First tests added for normalisation, more will follow in veropt 1.0

### Changed
- Dependencies are updated to newest versions
- Normalisation
  - Should work more correctly and be more robust now
- Sequence optimiser (proximity punish)
  - Now measuring scale globally to remove assumption of acq func range from 0 to a positive number
  - Furthermore checking for multiple disjoint distributions of acquisition function values and (if found) uses std of 
    top distribution
  - All this should ensure correct behaviour and avoid bugs that caused optimisation to 1) choose the same point 
    multiple times or 2) making the punishment dominate the landscape

### Removed
- Temporary:
  - Saver (will come back in later release!)
- Possibly permanent:
  - UCB with noise