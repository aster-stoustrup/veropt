# Changelog
All notable changes to this project will be documented in this file.

## [1.2.0] - 10-12-2025
Major improvements of visual tools and added new kernels.

### Added
- New built-in kernels
- 3d prediction plot
- Ability to run all graphs without normalisation (now default)
- Template for writing a new experiment
- Two visualisation examples
- General improvement to most visual tools
- 

### Changed
- Moved some internal visual methods to their own folders
  - 'visualisation.py' is now where users should go to find visual methods
- Changed naming of public visual methods
- Suggested points are now reset after loading new data instead of 
  after saving suggestions
- Learning rate setting has been fixed and is now
  - a) functional and
  - b) residing in the model optimiser where it belongs
- Model optimiser has been cleaned up and now follows same system as similar objects
- Fixed issue from pydantic with saving nan's to json
- jsons are pretty-printed
- Objective values will not be re-calculated if they're already in exp state
- Fixed minor bug when saving suggested steps

## [1.1.2] - 17-11-2025
Added the ability to use existing run with a new objective

### Added
- New experiment constructor that will use new ability to create 
  new version of existing experiment.

### Changed
- Name and location of optimiser and experiment state jsons

## [1.1.0] - 25-10-2025
Updated interfaces to allow pausing and resuming runs.

### Added
- Ability to resume runs that have been stopped
- Support for optimiser configuration
- Experiment will save optimiser state and reload it

### Changed
- Some internal refactoring on the Experiment class

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