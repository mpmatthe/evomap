# evomap

A Python toolbox for evolving mapping of relationship data.

`evomap` offers a comprehensive toolbox to create, explore and analyze spatial representations ('maps') from relationship data. Such maps are commonly applied in market structure analysis, study different kinds of networks (e.g., social, economic, or biological), political ideology, or more broadly to detect structure in high-dimensional data. 

Often, relationship data is retrievable over time, as markets and networks tend to evolve. `evomap` provides all necessary tools to analyze such data in maps either at a single period, or over time. Thereby, `evomap` provides an all-in-one solution and integrates many steps of the analysis into an easy-to-use API. Specifically, `evomap` includes modules for 

- preprocessing
- mapping
- evaluation
- plotting
- export

Note: Currently, `evomap` is available as a ***pre-release version***. Thus, parts of `evomap` are still under active development. For any bug reports or feature requests, <a href = 'mailto:matthe@wiwi.uni-frankfurt.de'>please get in touch</a>.

## Installation

As of now, `marketmaps` is available via GitHub. Stay tuned for a release on PyPi, which is coming soon! 


To install `evomap` run
```bash
conda create -n evomap python=3.9
conda activate evomap
pip install git+https://github.com/mpmatthe/evomap
```

## Usage

The following tutorial provide a good starting point for using `evomap`. 

For a quick overview on a typical market structure application, see <a href = 'https://evomap.readthedocs.io/en/latest/car%20application.html'>this example</a>.

If you want to dive deaper into what `evomap` has to offer, check out the following examples on

1. <a href = 'https://evomap.readthedocs.io/en/latest/static%20mapping.html'>Static Mapping</a>
2. <a href = 'https://evomap.readthedocs.io/en/latest/dynamic%20mapping.html'>Dynamic Mapping</a>

These examples are updated as new features are released!

## Mapping Methods

As of now, `evomap` provides implementations of the following mapping methods:
- MDS (Metric/Non Metric Multidimensional Scaling)
- Sammon Mapping (non-linear MDS)
- t-SNE (t-distributed Stochastic Neighborhood Embedding)

All methods can be applied both statically and dynamically. Moreover, `evomap` follows the syntax conventions of `scikit-learn`, such that other 
machine-learning techniques (such as LLE, Isomap, ... ) can easily be integrated. For more background, see <a href = 'https://scikit-learn.org/stable/modules/manifold.html'> here</a>.

## References

This package is based on the authors' work in 

```
[1] Matthe, M., Ringel, D. M., Skiera, B. (2022), Mapping Market Structure Evolution. Working Paper.
```

<b><i>Please remember to cite this paper if you use any code from this package!</i></b>

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

`evomap` was created by Maximilian Matthe based on joint research with Daniel M. Ringel and Bernd Skiera. It is licensed under the terms of the MIT license. It is free to use, however, <b><i>please cite our work</i></b>.

## Credits

`evomap` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
