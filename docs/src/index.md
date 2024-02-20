# Passivity-preserving model reduction for descriptor systems via spectral factorization

## Citing

If you use this project for academic work, please consider citing our
[publication](https://TODO)

    TODO

## How to install

Clone the project and navigate to the folder:

```bash
shell> git clone https://github.com/steff-mueller/spectralFactorMORdescriptor.git
shell> cd spectralFactorMORdescriptor
```

Activate and instantiate the Julia environment using the
[Julia package manager](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project)
to install the required packages:

```julia
pkg> activate .
pkg> instantiate
```

## How to reproduce

The `scripts/` folder contains TOML configuration files for
the different experiments from our paper:

| TOML configuration file   | Experiment                     |
| ------------------------- | ------------------------------ |
| `scripts/RCL-1-SISO.toml` | Index-1 SISO descriptor system |
| `scripts/RCL-1-MIMO.toml` | Index-1 MIMO descriptor system |
| `scripts/RCL-2-SISO.toml` | Index-2 SISO descriptor system |
| `scripts/RCL-2-MIMO.toml` | Index-2 MIMO descriptor system |

Point the `RCL_CONFIG` environment variable to
the experiment you want to run. Use the `scripts/rcl.jl` script to run an
experiment.  For example, to run `scripts/RCL-2-SISO.toml`,
execute the following commands:

```bash
shell> cd spectralFactorMORdescriptor
shell> export RCL_CONFIG="scripts/RCL-2-SISO.toml"
shell> julia --project=. scripts/rcl.jl
```

The experiment results are stored under `data/`.

## Julia package

The project contains a Julia package under `src/SpectralFactorMOR`
which you can include in your own Julia projects to use the methods.
See [`SpectralFactorMOR`](SpectralFactorMOR.md).

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Steffen Müller - steffen.mueller@simtech.uni-stuttgart.de\
Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de
