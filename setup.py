# -*- coding: utf-8 -*-

package_dir = \
{'': 'src'}

packages = \
['evomap',
 'evomap.data',
 'evomap.data.cars',
 'evomap.data.tnic_sample_small',
 'evomap.data.tnic_snapshot',
 'evomap.data.tnic_snapshot_small',
 'evomap.mapping',
 'evomap.mapping.evomap']

package_data = \
{'': ['*']}

install_requires = \
['Cython>=0.29.28',
 'ipykernel>=6.13.0',
 'matplotlib>=3.5.1',
 'numba>=0.55.1,<0.56.0',
 'numpy=1.21.6',
 'pandas>=1.3.3',
 'scipy>=1.7.0',
 'seaborn>=0.11.2',
 'statsmodels>=0.13.2,<0.14.0',
 'setuptools>=62.3.2']

setup_kwargs = {
    'name': 'evomap',
    'version': '0.1.0',
    'description': 'A Python Toolbox for Evolving Mapping of Relationship Data',
    'long_description': "# evomap\n\nA Python toolbox for evolving mapping of relationship data.\n\n`evomap` offers a comprehensive toolbox to create, explore and analyze spatial representations ('maps') from relationship data. Such maps are commonly applied in market structure analysis, study different kinds of networks (e.g., social, economic, or biological), political ideology, or more broadly to detect structure in high-dimensional data. \n\nOften, relationship data is retrievable over time, as markets and networks tend to evolve. `evomap` provides all necessary tools to analyze such data in maps either at a single period, or over time. Thereby, `evomap` provides an all-in-one solution and integrates many steps of the analysis into an easy-to-use API. Specifically, `evomap` includes modules for \n\n- preprocessing\n- mapping\n- evaluation\n- plotting\n- export\n\nNote: Parts of `evomap` are still under active development. For any bug reports or feature requests, <a href = 'mailto:matthe@wiwi.uni-frankfurt.de'>please get in touch</a>.\n\n## Installation\n\nAs of now, `marketmaps` is available via GitHub. Stay tuned for a release on PyPi, which is coming soon! \n\n```bash\n# Add GitHub Installation Code\n```\n\n## Usage\n\nThe following tutorial provide a good starting point for using `evomap`. \n\nFor a quick overview on a typical market structure application, see <a href = 'car_application.ipynb'>this example</a>.\n\nIf you want to dive deaper into what `evomap` has to offer, check out the following examples on\n\n1. <a href = 'static mapping.html'>Static Mapping</a>\n2. <a href = 'dynamic mapping.html'>Dynamic Mapping</a>\n\nThese examples are updated as new features are released!\n\n## Mapping Methods\n\nAs of now, `evomap` provides implementations of the following mapping methods:\n- MDS (Metric/Non Metric Multidimensional Scaling)\n- Sammon Mapping (non-linear MDS)\n- t-SNE (t-distributed Stochastic Neighborhood Embedding)\n\nAll methods can be applied both statically and dynamically. Moreover, `evomap` follows the syntax conventions of `scikit-learn`, such that other \nmachine-learning techniques (such as LLE, Isomap, ... ) can easily be integrated. For more background, see <a href = 'https://scikit-learn.org/stable/modules/manifold.html'> here</a>.\n\n## Work in progress\n\n`evomap` is under active development. Thus, additional features will be available in the future, such as\n\n- fitting consumers' ideal points \n- further mapping methods\n- ...\n\n## References\n\nThis package is based on own work in \n\n```\n[1] Matthe, M., Ringel, D. M., Skiera, B. (2022), Mapping Market Structure Evolution. Working Paper.\n```\n\n<b><i>Please remember to cite this paper if you use any code from this package!</i></b>\n\n`evomap` also builds upon the work of others, including\n```\n[2] Ringel, D. M., & Skiera, B. (2016). Visualizing asymmetric competition among more than 1,000 products using big search data. Marketing Science, 35(3), 511-534.\n\n[3] Torgerson, W. S. (1952). Multidimensional Scaling: I. Theory and method. Psychometrika, 17(4), 401-419.\n\n[4] Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11).\n\n[5] Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.\n\n[6] Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.\n```\n\nIf you use the respective methods implemented in `evomap`, consider also citing the original references.\n\n## Contributing\n\nInterested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.\n\n## License\n\n`evomap` was created by Maximilian Matthe based on joint research with Daniel M. Ringel and Bernd Skiera. It is licensed under the terms of the MIT license. It is free to use, however, <b><i>please cite our work</i></b>.\n\n## Credits\n\n`evomap` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).\n",
    'author': 'Maximilian Matthe',
    'author_email': 'matthe@wiwi.uni-frankfurt.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.11',
}
from build import *
build(setup_kwargs)

from setuptools import setup
setup(**setup_kwargs)
