# Passivity-preserving model reduction for descriptor systems via spectral factorization

[![CI][ci-shield]][ci-url]
[![Docs][docs-shield]][docs-url]
[![MIT License][license-shield]][license-url]

## Citing

If you use this project for academic work, please consider citing our
[publication][arxiv-url]:

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
the different experiments. Point the `RCL_CONFIG` environment variable to
the experiment you want to run. Use the `scripts/rcl.jl` script to run an
experiment.  For example, for `scripts/RCL-2-SISO.toml`,
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
See the instructions [here][docs-url-pkg].

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Steffen Müller - steffen.mueller@simtech.uni-stuttgart.de\
Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de

[arxiv-url]: TODO
[ci-shield]: https://github.com/steff-mueller/spectralFactorMORdescriptor/workflows/CI/badge.svg
[ci-url]: https://github.com/steff-mueller/spectralFactorMORdescriptor/actions
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg
[docs-url]: https://steff-mueller.github.io/spectralFactorMORdescriptor/
[docs-url-pkg]: https://steff-mueller.github.io/spectralFactorMORdescriptor/SpectralFactorMOR
[license-shield]: https://img.shields.io/github/license/steff-mueller/spectralFactorMORdescriptor.svg
[license-url]: https://github.com/steff-mueller/spectralFactorMORdescriptor/blob/main/LICENSE
