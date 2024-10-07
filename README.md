# evomap - A Toolbox for Dynamic Mapping in Python

`evomap` is a comprehensive Python toolbox for creating, exploring, and analyzing spatial representations ('maps') from complex data, including high-dimensional feature vectors and pairwise relationship matrices. Such spatial representations find frequent applications in Marketing (e.g., market structure or positioning analysis), Network Analysis (e.g., social, economic, bibliographic, or biological networks), Political Science (e.g., ideological scaling), and High-Dimensional Data Analysis.

A key use case of `evomap` is creating such maps from time-evolving data by processing longitudinal sequences of relationship matrices or high-dimensional feature vectors. The resultant maps allow users to track changes in complex systems, such as evolving markets or networks, and visualize their underlying evolution.

`evomap` offers an all-in-one solution by integrating several essential steps into an easy-to-use API, including:

- preprocessing
- mapping (static/dynamic)
- evaluation
- plotting

For any bug reports or feature requests, <a href = 'mailto:mpmatthe@iu.edu'>please get in touch</a>.

## Installation

`evomap` is available via PyPi. 

To install `evomap` run
```bash
pip install evomap
```

`evomap` requires Python version>=3.9. We recommend using Python within a virtual environment, for instance via conda:
```bash
conda create -n evomap python
conda activate evomap
pip install evomap
```

## Usage

The following tutorials provide a good starting point for using `evomap`.

For a simple introduction to a typical market structure application, see <a href='https://evomap.readthedocs.io/en/latest/car%20application.html'>this example</a>.

If you want to explore more of what evomap has to offer, check out the following examples on:

- <a href='https://evomap.readthedocs.io/en/latest/static%20mapping.html'>Static Mapping</a>
- <a href='https://evomap.readthedocs.io/en/latest/dynamic%20mapping.html'>Dynamic Mapping</a>

## Mapping Methods

As of now, `evomap` provides implementations of the following mapping methods:
- MDS (Multidimensional Scaling)
- Sammon Mapping (non-linear MDS)
- t-SNE (t-distributed Stochastic Neighbor Embedding)

You can apply all methods statically and dynamically. Moreover, `evomap` follows the syntax conventions of `scikit-learn`, such that other 
machine-learning techniques (such as LLE, Isomap, ... ) can easily be integrated. For more background, see <a href = 'https://scikit-learn.org/stable/modules/manifold.html'> here</a>.

## References

This package is based on the authors' work in 

```
[1] Matthe, M., Ringel, D. M., Skiera, B. (2023), Mapping Market Structure Evolution. Marketing Science, Vol. 42, Issue 3, 589-613.
```
Read the full paper here (open access): <a href = 'https://doi.org/10.1287/mksc.2022.1385'>https://doi.org/10.1287/mksc.2022.1385</a> 

<b><i>Please cite our paper if you use this package or part of its code</i></b>

`evomap` also builds upon the work of others, including
```
[2] Torgerson, W. S. (1952). Multidimensional Scaling: I. Theory and method. Psychometrika, 17(4), 401-419.

[3] Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.

[4] Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.

[5] Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11).

[6] Ringel, D. M., & Skiera, B. (2016). Visualizing asymmetric competition among more than 1,000 products using big search data. Marketing Science, 35(3), 511-534.
```

If you use any of the methods implemented in 'evomap', please consider citing the original references alongside this package.

## Contributing

Interested in contributing? <a href = 'mailto:mpmatthe@iu.edu'>Get in touch</a>!
## License

`evomap` is licensed under the terms of the MIT license. It is free to use, however, <i>please cite our work</i>.

## Credits

`evomap` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
