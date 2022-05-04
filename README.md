# marketmaps

A Python package for exploring markets in maps.

`marketmaps` offers a comprehensive toolbox to create, explore and analyze market maps - that is, spatial representation of market actors' positions. 
Such maps allow researchers and decision makers to identify key competitors, analyze their relative positioning, discover submarkets, or monitor a market's evolution over time. As such, they are an important ingredient of modern marketing analytics.

Creating market maps, however, is challenging. Doing so can require extensive preprocessing, . Moreover, implementations of different mapping algorithms are scattered across different platforms (e.g., MatLab, R, Python or R). Finally, exploring the resultant maps can provide further challenges - especially in larger or more complex markets. 

To alleviate these challenges, `marketmaps` provides an all-in-one solution and integrates many steps of the analysis into a easy-to-use API. 

Note: `marketmaps` is still under active development. For any bug reports or feature requests, <a href = 'mailto:matthe@wiwi.uni-frankfurt.de'>please get in touch</a>.

## Installation

As of now, `marketmaps` is available from GitHub. Stay tuned for a release on PyPi, which is coming soon! 

```bash
$ pip install marketmaps
```

## Usage

`marketmaps` entails different modules, corresponding to individual tasks of the market mapping process (such as preprocessing, mapping, ploting or evaluation).

For a quickstart on how to use `marketmaps`, see <a href = 'car_application.ipynb'>this example</a>.

If you want to dive deaper into what `marketmaps` has to offer, see the following examples:

1. <a href = 'static mapping.html'>Static Mapping</a>
2. <a href = 'dynamic mapping.html'>Dynamic Mapping</a>

These examples are updated as new features are released!

## Mapping Methods

Implementations for the following mapping methods are currently available:
- CMDS (Classic Multidimensional Scaling)
- MDS (Metric/Non Metric Multidimensional Scaling)
- Sammon Mapping (non-linear MDS)
- VOS (Visualization of Similarities)
- EvoMap (implemented for t-SNE, MDS and Sammon Mapping)

Moreover, `marketmaps` follows the syntax conventions of `scikit-learn`, such that 
machine-learning techniques (such as t-SNE, LLE, Isomap, ... ) can easily be integrated. For more background, see <a href = 'https://scikit-learn.org/stable/modules/manifold.html'> here</a>.

## Work in progress

The following features are under active development und will be available in the near future:

- estimation of consumers' ideal points (e.g., via multidimensional unfolding)
- integration into scikit-learn pipelines

## References

This package is based on lots of own prepratory work and heavily inspired by the work of others. Moreover, many of the included implementations greatly benefited from extant implementations (e.g., in different programming languages). Thus, if you use this package, please consider citing the corresponding references.

The following work either directly contributed or heavily inspired parts of this package: 

```
[1] Matthe, M., Ringel, D. M., Skiera, B. (2022), Mapping Market Structure Evolution. Working Paper.

[2] Ringel, D. M., & Skiera, B. (2016). Visualizing asymmetric competition among more than 1,000 products using big search data. Marketing Science, 35(3), 511-534.
```

Finally, don't forget to give credit to different mapping algorithms' original authors when using them: 

```
[4] Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11).

[5] Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.

[6] Eck, N. J. V., & Waltman, L. (2007). VOS: A new method for visualizing similarities between objects. In Advances in data analysis (pp. 299-306). Springer, Berlin, Heidelberg.

[7] Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`marketmaps` was created by Maximilian Matthe. It is licensed under the terms of the MIT license.

## Credits

`marketmaps` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
