# evomap - A Python Toolbox for Mapping Evolving Relationship Data

`evomap` offers a comprehensive toolbox to create, explore and analyze spatial representations ('maps') from relationship data. Common applications include Marketing (market structure analysis), Network Analysis (e.g., social, economic, or biological networks), Political Science, or High-Dimensional Data Analysis in general. 

Often, relationship data is retrievable over time, as markets and networks tend to evolve. `evomap` provides all necessary tools to analyze such data in maps either in static snapshots at a single point in time, or in evolving maps across multiple periods. `evomap` provides an all-in-one solution and integrates many steps of the analysis into an easy-to-use API. Specifically, `evomap` includes modules for 

- preprocessing
- mapping (static/dynamic)
- evaluation
- plotting

Note: As of now, `evomap` is available as a ***pre-release version*** and parts of `evomap` are still under active development. For any bug reports or feature requests, <a href = 'mailto:matthe@wiwi.uni-frankfurt.de'>please get in touch</a>.

## Installation

This pre-release is available via GitHub. Stay tuned for a release on PyPi, which is coming soon! 

To install `evomap` run
```bash
pip install git+https://github.com/mpmatthe/evomap
```

`evomap` requires Python version 3.7 (or higher). We recommend Python version 3.9 within a virtual environment, for instance via conda:
```bash
conda create -n evomap python=3.9
conda activate evomap
pip install git+https://github.com/mpmatthe/evomap
```

**Note:** Currently, `evomap` builds its C extensions upon installation on the system. Thus, it requires a C compiler to be present. The right C compiler depends upon your system, e.g. GCC on Linux or MSVC on Windows. For details, see <a href = 'https://cython.readthedocs.io/en/latest/src/quickstart/install.html'>the Cython documentation</a>. In future versions, extensions will be pre-compiled.

## Usage

The following tutorials provide a good starting point for using `evomap`. 

For a simple introduction to a typical market structure application, see <a href = 'https://evomap.readthedocs.io/en/latest/car%20application.html'>this example</a>.

If you want to dive deaper into what `evomap` has to offer, check out the following examples on

1. <a href = 'https://evomap.readthedocs.io/en/latest/static%20mapping.html'>Static Mapping</a>
2. <a href = 'https://evomap.readthedocs.io/en/latest/dynamic%20mapping.html'>Dynamic Mapping</a>

Updated versions of these examples will be available as new features are released. 

## Mapping Methods

As of now, `evomap` provides implementations of the following mapping methods:
- MDS (Multidimensional Scaling)
- Sammon Mapping (non-linear MDS)
- t-SNE (t-distributed Stochastic Neighborhood Embedding)

You can apply all methods statically and dynamically. Moreover, `evomap` follows the syntax conventions of `scikit-learn`, such that other 
machine-learning techniques (such as LLE, Isomap, ... ) can easily be integrated. For more background, see <a href = 'https://scikit-learn.org/stable/modules/manifold.html'> here</a>.

## References

This package is based on the authors' work in 

```
[1] Matthe, M., Ringel, D. M., Skiera, B. (2022), Mapping Market Structure Evolution. Working Paper.
```

<b><i>Please cite our paper if you use this package or part of its code</i></b>

`evomap` also builds upon the work of others, including
```
[2] Ringel, D. M., & Skiera, B. (2016). Visualizing asymmetric competition among more than 1,000 products using big search data. Marketing Science, 35(3), 511-534.

[3] Torgerson, W. S. (1952). Multidimensional Scaling: I. Theory and method. Psychometrika, 17(4), 401-419.

[4] Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11).

[5] Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.

[6] Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.
```

If you use the respective methods implemented in `evomap`, consider also citing the original references.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`evomap` is licensed under the terms of the MIT license. It is free to use, however, <i>please cite our work</i>.

## Credits

`evomap` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).